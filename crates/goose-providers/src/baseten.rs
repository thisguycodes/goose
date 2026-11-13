//! How Baseten's inference API wants to be retried and resumed.
//!
//! Baseten is reached through the OpenAI-compatible provider, under whatever
//! name the user gave their custom provider, so it is recognised by host.
//! Behaviour here follows <https://docs.baseten.co/inference/errors> and
//! <https://docs.baseten.co/reference/inference-api/chat-completions>.

use std::collections::HashMap;

use serde_json::json;

use crate::retry::RetryConfig;
use crate::stream_resume::StreamResumePolicy;

const BASETEN_DOMAIN: &str = "baseten.co";
const MODEL_APIS_HOST: &str = "inference.baseten.co";

/// Chat Completions applies this when a request names no `max_tokens`.
const DEFAULT_MAX_TOKENS: i32 = 4096;

fn host_of(base_url: &str) -> Option<String> {
    url::Url::parse(base_url)
        .ok()?
        .host_str()
        .map(str::to_ascii_lowercase)
}

/// Whether `base_url` is a Baseten endpoint: Model APIs or a dedicated
/// deployment (`model-<id>.api.baseten.co`).
pub fn is_baseten_url(base_url: &str) -> bool {
    host_of(base_url).is_some_and(|host| {
        host == BASETEN_DOMAIN
            || host
                .strip_suffix(BASETEN_DOMAIN)
                .is_some_and(|prefix| prefix.ends_with('.'))
    })
}

/// Whether `base_url` is Model APIs. A dedicated deployment runs whatever server
/// its owner deployed, so the stream contract below is not known to hold there.
pub fn is_model_apis_url(base_url: &str) -> bool {
    host_of(base_url).is_some_and(|host| host == MODEL_APIS_HOST)
}

/// Baseten asks clients to retry 429, 5xx and connection errors but never
/// 400, 401, 403, 404 or 422.
pub fn retry_config() -> RetryConfig {
    RetryConfig::default().transient_only()
}

/// A Model APIs stream can end early without an error event, so one that never
/// reports a `finish_reason` was cut off.
///
/// The chat template only leaves a trailing assistant message open when
/// `continue_final_message` is set, which in turn requires
/// `add_generation_prompt` to be off. The Model APIs reference lists only the
/// latter; if the endpoint rejects the request the response is not resumed.
pub fn stream_resume_policy() -> StreamResumePolicy {
    StreamResumePolicy {
        prefill: true,
        prefill_request_params: HashMap::from([
            ("add_generation_prompt".to_string(), json!(false)),
            ("continue_final_message".to_string(), json!(true)),
        ]),
        requires_finish_reason: true,
        default_max_tokens: Some(DEFAULT_MAX_TOKENS),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    #[test_case("https://inference.baseten.co", true ; "model apis")]
    #[test_case("https://inference.baseten.co/v1", true ; "model apis with path")]
    #[test_case("https://model-abc123.api.baseten.co", true ; "dedicated deployment")]
    #[test_case("https://INFERENCE.BASETEN.CO", true ; "case insensitive")]
    #[test_case("https://notbaseten.co", false ; "suffix of another domain")]
    #[test_case("https://baseten.co.example.com", false ; "prefix of another domain")]
    #[test_case("http://localhost:11434", false ; "local server")]
    #[test_case("not a url", false ; "unparseable")]
    fn recognises_baseten_hosts(base_url: &str, expected: bool) {
        assert_eq!(is_baseten_url(base_url), expected);
    }

    #[test_case("https://inference.baseten.co/v1", true ; "model apis")]
    #[test_case("https://model-abc123.api.baseten.co", false ; "dedicated deployment")]
    #[test_case("https://api.example.com", false ; "another provider")]
    fn recognises_model_apis(base_url: &str, expected: bool) {
        assert_eq!(is_model_apis_url(base_url), expected);
    }
}
