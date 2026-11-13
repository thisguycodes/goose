//! What a failed response says about when to try again.
//!
//! `http_status` reads the `Retry-After` of a 429. Providers say more than that:
//! capacity errors (529, or any 5xx carrying `Retry-After`) ask for the same
//! wait-then-retry as a rate limit, some send the wait in milliseconds or as the
//! time until a `x-ratelimit-*` window resets, and the `x-ratelimit-remaining-*`
//! headers name the limit that ran out.

use std::time::Duration;

use reqwest::header::{HeaderMap, RETRY_AFTER};
use reqwest::StatusCode;
use serde_json::Value;

use crate::errors::ProviderError;
use crate::http_status::{extract_retry_after, parse_request_id_suffix, request_id_message_suffix};

/// Not in the IANA registry; Baseten and Anthropic use it for "out of capacity".
const STATUS_OVERLOADED: u16 = 529;

/// Matches the cap `http_status` puts on `Retry-After`: a malformed hint should
/// not freeze the agent.
const MAX_HINT: Duration = Duration::from_secs(3600);

const RATE_LIMIT_KINDS: [&str; 2] = ["requests", "tokens"];

pub(crate) fn apply(
    error: ProviderError,
    status: StatusCode,
    headers: &HeaderMap,
    payload: Option<&Value>,
) -> ProviderError {
    match error {
        ProviderError::RateLimitExceeded {
            details,
            retry_delay,
        } => ProviderError::RateLimitExceeded {
            details: with_remaining_allowance(details, headers),
            retry_delay: retry_delay.or_else(|| delay_from_headers(headers)),
        },
        // Out of capacity is not an account limit, but it clears the same way:
        // wait as long as the server asks, then send the request again.
        ProviderError::ServerError(details) => {
            let retry_delay =
                extract_retry_after(headers, payload).or_else(|| delay_from_headers(headers));
            if status.as_u16() == STATUS_OVERLOADED || retry_delay.is_some() {
                ProviderError::RateLimitExceeded {
                    details,
                    retry_delay,
                }
            } else {
                ProviderError::ServerError(details)
            }
        }
        other => other,
    }
}

fn header_text<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    let value = headers.get(name)?.to_str().ok()?.trim();
    (!value.is_empty()).then_some(value)
}

fn bounded(seconds: f64) -> Option<Duration> {
    (seconds.is_finite() && seconds >= 0.0)
        .then(|| Duration::from_secs_f64(seconds.min(MAX_HINT.as_secs_f64())))
}

/// Retry hints `extract_retry_after` does not read: `retry-after-ms`, a
/// fractional `Retry-After`, and the reset time of an exhausted rate limit.
fn delay_from_headers(headers: &HeaderMap) -> Option<Duration> {
    header_text(headers, "retry-after-ms")
        .and_then(|value| value.parse::<f64>().ok())
        .and_then(|millis| bounded(millis / 1000.0))
        .or_else(|| {
            header_text(headers, RETRY_AFTER.as_str())
                .and_then(|value| value.parse::<f64>().ok())
                .and_then(bounded)
        })
        .or_else(|| exhausted_limit_reset(headers))
}

fn exhausted_limit_reset(headers: &HeaderMap) -> Option<Duration> {
    RATE_LIMIT_KINDS
        .iter()
        .filter(|kind| {
            header_text(headers, &format!("x-ratelimit-remaining-{kind}"))
                .and_then(|value| value.parse::<f64>().ok())
                .is_some_and(|remaining| remaining < 1.0)
        })
        .filter_map(|kind| {
            header_text(headers, &format!("x-ratelimit-reset-{kind}")).and_then(parse_reset)
        })
        .max()
}

/// `x-ratelimit-reset-*` is either plain seconds or a Go-style duration such as
/// `250ms`, `1.5s` or `6m0s`.
fn parse_reset(value: &str) -> Option<Duration> {
    if let Ok(seconds) = value.parse::<f64>() {
        return bounded(seconds);
    }

    let is_numeric = |c: char| c.is_ascii_digit() || c == '.';
    let mut total = 0.0;
    let mut rest = value;
    while !rest.is_empty() {
        let (amount, after_amount) = split_where(rest, |c| !is_numeric(c));
        let (unit, after_unit) = split_where(after_amount, is_numeric);
        let unit_seconds = match unit {
            "ms" => 0.001,
            "s" => 1.0,
            "m" => 60.0,
            "h" => 3600.0,
            _ => return None,
        };
        total += amount.parse::<f64>().ok()? * unit_seconds;
        rest = after_unit;
    }
    bounded(total)
}

fn split_where(text: &str, boundary: impl Fn(char) -> bool) -> (&str, &str) {
    text.split_at(text.find(boundary).unwrap_or(text.len()))
}

fn remaining_allowance(headers: &HeaderMap) -> Option<String> {
    let parts: Vec<String> = RATE_LIMIT_KINDS
        .iter()
        .filter_map(|kind| {
            let remaining = header_text(headers, &format!("x-ratelimit-remaining-{kind}"))?;
            Some(
                match header_text(headers, &format!("x-ratelimit-limit-{kind}")) {
                    Some(limit) => format!("{kind} left: {remaining}/{limit}"),
                    None => format!("{kind} left: {remaining}"),
                },
            )
        })
        .collect();
    (!parts.is_empty()).then(|| parts.join(", "))
}

/// Says which limit ran out. The request id stays last, where
/// `parse_request_id_suffix` looks for it.
fn with_remaining_allowance(details: String, headers: &HeaderMap) -> String {
    let Some(allowance) = remaining_allowance(headers) else {
        return details;
    };
    let request_id = parse_request_id_suffix(&details);
    let id_suffix = request_id_message_suffix(request_id.as_deref());
    let message = details.strip_suffix(&id_suffix).unwrap_or(&details);
    format!("{message} ({allowance}){id_suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(
                reqwest::header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                value.parse().unwrap(),
            );
        }
        map
    }

    fn rate_limited(details: &str, retry_delay: Option<Duration>) -> ProviderError {
        ProviderError::RateLimitExceeded {
            details: details.to_string(),
            retry_delay,
        }
    }

    fn overloaded() -> StatusCode {
        StatusCode::from_u16(STATUS_OVERLOADED).unwrap()
    }

    #[test]
    fn overloaded_backs_off_like_a_rate_limit() {
        let error = apply(
            ProviderError::ServerError("no capacity".to_string()),
            overloaded(),
            &HeaderMap::new(),
            None,
        );

        assert_eq!(error, rate_limited("no capacity", None));
    }

    #[test]
    fn overloaded_waits_as_long_as_retry_after_asks() {
        let error = apply(
            ProviderError::ServerError("no capacity".to_string()),
            overloaded(),
            &headers(&[("retry-after", "7")]),
            None,
        );

        assert_eq!(
            error,
            rate_limited("no capacity", Some(Duration::from_secs(7)))
        );
    }

    #[test]
    fn server_error_with_retry_after_honors_the_wait() {
        let error = apply(
            ProviderError::ServerError("draining".to_string()),
            StatusCode::SERVICE_UNAVAILABLE,
            &headers(&[("retry-after", "3")]),
            None,
        );

        assert_eq!(
            error,
            rate_limited("draining", Some(Duration::from_secs(3)))
        );
    }

    #[test]
    fn server_error_without_a_hint_is_unchanged() {
        let error = ProviderError::ServerError("boom".to_string());

        assert_eq!(
            apply(
                error.clone(),
                StatusCode::INTERNAL_SERVER_ERROR,
                &HeaderMap::new(),
                None
            ),
            error
        );
    }

    #[test]
    fn existing_retry_delay_is_kept() {
        let error = apply(
            rate_limited("slow down", Some(Duration::from_secs(17))),
            StatusCode::TOO_MANY_REQUESTS,
            &headers(&[("retry-after-ms", "250")]),
            None,
        );

        assert_eq!(
            error,
            rate_limited("slow down", Some(Duration::from_secs(17)))
        );
    }

    #[test_case(&[("retry-after-ms", "1500")], Some(Duration::from_millis(1500)) ; "milliseconds")]
    #[test_case(&[("retry-after", "1.5")], Some(Duration::from_millis(1500)) ; "fractional seconds")]
    #[test_case(
        &[("x-ratelimit-remaining-requests", "0"), ("x-ratelimit-reset-requests", "6m0s"),
          ("x-ratelimit-remaining-tokens", "900"), ("x-ratelimit-reset-tokens", "20ms")],
        Some(Duration::from_secs(360)) ; "reset of the exhausted limit"
    )]
    #[test_case(
        &[("x-ratelimit-remaining-requests", "5"), ("x-ratelimit-reset-requests", "1s")],
        None ; "reset of a limit with allowance left is not a wait"
    )]
    #[test_case(&[("x-ratelimit-reset-requests", "soon")], None ; "unparseable reset")]
    #[test_case(&[], None ; "no hint")]
    fn rate_limit_delay_comes_from_headers(pairs: &[(&str, &str)], expected: Option<Duration>) {
        let error = apply(
            rate_limited("slow down", None),
            StatusCode::TOO_MANY_REQUESTS,
            &headers(pairs),
            None,
        );

        let ProviderError::RateLimitExceeded { retry_delay, .. } = error else {
            panic!("expected a rate limit, got {error:?}");
        };
        assert_eq!(retry_delay, expected);
    }

    #[test_case("250ms", Duration::from_millis(250))]
    #[test_case("1.5s", Duration::from_millis(1500))]
    #[test_case("1m2s", Duration::from_secs(62))]
    #[test_case("12", Duration::from_secs(12))]
    #[test_case("99999h", MAX_HINT)]
    fn reset_durations_parse(value: &str, expected: Duration) {
        assert_eq!(parse_reset(value), Some(expected));
    }

    #[test]
    fn exhausted_limit_is_named_ahead_of_the_request_id() {
        let details = format!(
            "Rate limit exceeded{}",
            request_id_message_suffix(Some("req_123"))
        );

        let error = apply(
            rate_limited(&details, None),
            StatusCode::TOO_MANY_REQUESTS,
            &headers(&[
                ("x-ratelimit-limit-requests", "120"),
                ("x-ratelimit-remaining-requests", "0"),
                ("x-ratelimit-limit-tokens", "500000"),
                ("x-ratelimit-remaining-tokens", "4321"),
            ]),
            None,
        );

        let ProviderError::RateLimitExceeded { details, .. } = error else {
            panic!("expected a rate limit");
        };
        assert_eq!(
            details,
            "Rate limit exceeded (requests left: 0/120, tokens left: 4321/500000) \
             (provider request id: req_123)"
        );
        assert_eq!(
            parse_request_id_suffix(&details).as_deref(),
            Some("req_123")
        );
    }
}
