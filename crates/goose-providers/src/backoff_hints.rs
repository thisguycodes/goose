//! What a failed response says about when to try again.
//!
//! `http_status` reads the `Retry-After` of a 429. Providers say more than that:
//! capacity errors (529, or a 5xx carrying `Retry-After`) ask for the same
//! wait-then-retry as a rate limit, some send the wait in milliseconds or as the
//! time a `x-ratelimit-*` window resets, and the `x-ratelimit-remaining-*`
//! headers name the limit that ran out. Baseten also identifies the request in
//! a header of its own, which support asks for.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::header::{HeaderMap, RETRY_AFTER};
use reqwest::StatusCode;
use serde_json::Value;

use crate::errors::ProviderError;
use crate::http_status::{extract_retry_after, parse_request_id_suffix, request_id_message_suffix};

/// Not in the IANA registry; Baseten and Anthropic use it for "out of capacity".
const STATUS_OVERLOADED: u16 = 529;

const BASETEN_REQUEST_ID_HEADER: &str = "x-baseten-request-id";

/// A shorter hint is dropped so that exponential backoff applies: "retry now"
/// from several clients at once is how an overloaded server stays overloaded.
const MIN_HINT: Duration = Duration::from_secs(1);

/// A longer hint from the headers read here is taken as malformed, matching the
/// cap `http_status` puts on `Retry-After`.
const MAX_HINT: Duration = Duration::from_secs(3600);

/// A server error is only worth waiting out when the wait is short; a
/// maintenance window should end the turn rather than hold it open.
const MAX_SERVER_ERROR_WAIT: Duration = Duration::from_secs(60);

/// Numeric `x-ratelimit-reset-*` values at least this large are Unix timestamps
/// (in seconds, or in milliseconds from `EPOCH_MILLIS`) rather than seconds from now.
const EPOCH_SECONDS: f64 = 1e9;
const EPOCH_MILLIS: f64 = 1e11;

const RATE_LIMIT_KINDS: [&str; 2] = ["requests", "tokens"];

pub(crate) fn apply(
    error: ProviderError,
    status: StatusCode,
    headers: &HeaderMap,
    payload: Option<&Value>,
) -> ProviderError {
    match with_baseten_request_id(error, headers) {
        ProviderError::RateLimitExceeded {
            details,
            retry_delay,
        } => ProviderError::RateLimitExceeded {
            details: with_remaining_allowance(details, headers),
            retry_delay: retry_delay
                .or_else(|| delay_from_headers(headers))
                .filter(|delay| *delay >= MIN_HINT),
        },
        // Out of capacity is not an account limit, but it clears the same way:
        // wait as long as the server asks, then send the request again.
        ProviderError::ServerError(details) => {
            let retry_delay = extract_retry_after(headers, payload)
                .or_else(|| delay_from_headers(headers))
                .filter(|delay| *delay >= MIN_HINT);
            if status.as_u16() == STATUS_OVERLOADED {
                ProviderError::RateLimitExceeded {
                    details: format!("the provider is overloaded. {details}"),
                    retry_delay,
                }
            } else if retry_delay.is_some_and(|delay| delay <= MAX_SERVER_ERROR_WAIT) {
                ProviderError::RateLimitExceeded {
                    details: format!("the provider asked to retry later. {details}"),
                    retry_delay,
                }
            } else {
                ProviderError::ServerError(details)
            }
        }
        other => other,
    }
}

fn with_baseten_request_id(mut error: ProviderError, headers: &HeaderMap) -> ProviderError {
    let request_id = header_text(headers, BASETEN_REQUEST_ID_HEADER);
    if let (Some(details), Some(_)) = (error.details_mut(), request_id) {
        if parse_request_id_suffix(details).is_none() {
            details.push_str(&request_id_message_suffix(request_id));
        }
    }
    error
}

fn header_text<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    let value = headers.get(name)?.to_str().ok()?.trim();
    (!value.is_empty()).then_some(value)
}

fn in_range(seconds: f64) -> Option<Duration> {
    (seconds.is_finite() && (0.0..=MAX_HINT.as_secs_f64()).contains(&seconds))
        .then(|| Duration::from_secs_f64(seconds))
}

/// Retry hints `extract_retry_after` does not read: `retry-after-ms`, a
/// fractional `Retry-After`, and the reset time of an exhausted rate limit.
fn delay_from_headers(headers: &HeaderMap) -> Option<Duration> {
    header_text(headers, "retry-after-ms")
        .and_then(|value| value.parse::<f64>().ok())
        .and_then(|millis| in_range(millis / 1000.0))
        .or_else(|| {
            header_text(headers, RETRY_AFTER.as_str())
                .and_then(|value| value.parse::<f64>().ok())
                .and_then(in_range)
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

/// `x-ratelimit-reset-*` has no standard form. Providers send seconds from now,
/// a Go-style duration such as `250ms`, `1.5s` or `6m0s`, or a Unix timestamp.
fn parse_reset(value: &str) -> Option<Duration> {
    if let Ok(number) = value.parse::<f64>() {
        return in_range(seconds_until(number));
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
    in_range(total)
}

fn seconds_until(number: f64) -> f64 {
    let epoch_seconds = if number >= EPOCH_MILLIS {
        number / 1000.0
    } else if number >= EPOCH_SECONDS {
        number
    } else {
        return number;
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    epoch_seconds - now.as_secs_f64()
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

    fn server_error(status: StatusCode, pairs: &[(&str, &str)]) -> ProviderError {
        apply(
            ProviderError::ServerError("boom".to_string()),
            status,
            &headers(pairs),
            None,
        )
    }

    #[test]
    fn overloaded_backs_off_like_a_rate_limit() {
        assert_eq!(
            server_error(overloaded(), &[]),
            rate_limited("the provider is overloaded. boom", None)
        );
    }

    #[test]
    fn overloaded_waits_as_long_as_retry_after_asks() {
        assert_eq!(
            server_error(overloaded(), &[("retry-after", "7")]),
            rate_limited(
                "the provider is overloaded. boom",
                Some(Duration::from_secs(7))
            )
        );
    }

    #[test]
    fn server_error_with_a_short_retry_after_is_waited_out() {
        assert_eq!(
            server_error(StatusCode::SERVICE_UNAVAILABLE, &[("retry-after", "3")]),
            rate_limited(
                "the provider asked to retry later. boom",
                Some(Duration::from_secs(3))
            )
        );
    }

    #[test_case(&[("retry-after", "3600")] ; "long wait")]
    #[test_case(&[("retry-after", "0")] ; "retry now")]
    #[test_case(&[] ; "no hint")]
    fn other_server_errors_are_unchanged(pairs: &[(&str, &str)]) {
        assert_eq!(
            server_error(StatusCode::SERVICE_UNAVAILABLE, pairs),
            ProviderError::ServerError("boom".to_string())
        );
    }

    #[test]
    fn existing_retry_delay_is_kept() {
        let error = apply(
            rate_limited("slow down", Some(Duration::from_secs(17))),
            StatusCode::TOO_MANY_REQUESTS,
            &headers(&[("retry-after-ms", "2500")]),
            None,
        );

        assert_eq!(
            error,
            rate_limited("slow down", Some(Duration::from_secs(17)))
        );
    }

    #[test]
    fn retry_now_is_left_to_exponential_backoff() {
        let error = apply(
            rate_limited("slow down", Some(Duration::ZERO)),
            StatusCode::TOO_MANY_REQUESTS,
            &headers(&[("retry-after", "0")]),
            None,
        );

        assert_eq!(error, rate_limited("slow down", None));
    }

    #[test_case(&[("retry-after-ms", "1500")], Some(Duration::from_millis(1500)) ; "milliseconds")]
    #[test_case(&[("retry-after", "1.5")], Some(Duration::from_millis(1500)) ; "fractional seconds")]
    #[test_case(
        &[("x-ratelimit-remaining-requests", "0"), ("x-ratelimit-reset-requests", "6m0s"),
          ("x-ratelimit-remaining-tokens", "900"), ("x-ratelimit-reset-tokens", "20ms")],
        Some(Duration::from_secs(360)) ; "reset of the exhausted limit"
    )]
    #[test_case(
        &[("x-ratelimit-remaining-requests", "5"), ("x-ratelimit-reset-requests", "2s")],
        None ; "reset of a limit with allowance left is not a wait"
    )]
    #[test_case(
        &[("x-ratelimit-remaining-requests", "0"), ("x-ratelimit-reset-requests", "soon")],
        None ; "unparseable reset"
    )]
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

    #[test_case("250ms", Some(Duration::from_millis(250)))]
    #[test_case("1.5s", Some(Duration::from_millis(1500)))]
    #[test_case("1m2s", Some(Duration::from_secs(62)))]
    #[test_case("12", Some(Duration::from_secs(12)))]
    #[test_case("99999h", None ; "longer than any real window")]
    #[test_case("5m30", None ; "missing unit")]
    fn reset_durations_parse(value: &str, expected: Option<Duration>) {
        assert_eq!(parse_reset(value), expected);
    }

    #[test_case(1.0 ; "unix seconds")]
    #[test_case(1000.0 ; "unix milliseconds")]
    fn reset_timestamp_is_the_time_left_until_it(scale: f64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let delay = parse_reset(&format!("{:.0}", (now + 30.0) * scale)).unwrap();

        assert!(
            (28.0..=30.5).contains(&delay.as_secs_f64()),
            "expected about 30s, got {delay:?}"
        );
    }

    #[test]
    fn reset_timestamp_in_the_past_is_not_a_wait() {
        assert_eq!(parse_reset("1700000000"), None);
    }

    #[test]
    fn baseten_request_id_is_attached_unless_the_error_already_names_one() {
        let from_baseten = headers(&[(BASETEN_REQUEST_ID_HEADER, "req_baseten")]);

        let error = apply(
            ProviderError::RequestFailed("Bad request (400): nope".to_string()),
            StatusCode::BAD_REQUEST,
            &from_baseten,
            None,
        );
        assert_eq!(
            parse_request_id_suffix(&error.to_string()).as_deref(),
            Some("req_baseten")
        );

        let already_identified = format!("nope{}", request_id_message_suffix(Some("req_other")));
        let error = apply(
            ProviderError::RequestFailed(already_identified.clone()),
            StatusCode::BAD_REQUEST,
            &from_baseten,
            None,
        );
        assert_eq!(error, ProviderError::RequestFailed(already_identified));
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
