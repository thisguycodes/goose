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

/// Chat Completions applies this when a request names no `max_tokens`.
const DEFAULT_MAX_TOKENS: i32 = 4096;

/// Whether `base_url` is a Baseten endpoint: Model APIs (`inference.baseten.co`)
/// or a dedicated deployment (`model-<id>.api.baseten.co`).
pub fn is_baseten_url(base_url: &str) -> bool {
    url::Url::parse(base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase))
        .is_some_and(|host| {
            host == BASETEN_DOMAIN
                || host
                    .strip_suffix(BASETEN_DOMAIN)
                    .is_some_and(|prefix| prefix.ends_with('.'))
        })
}

/// Baseten asks clients to retry 429, 5xx and connection errors but never
/// 400, 401, 403, 404 or 422.
pub fn retry_config() -> RetryConfig {
    RetryConfig::default().transient_only()
}

/// A stream can end early without an error event, so one that never reports a
/// `finish_reason` was cut off.
///
/// `add_generation_prompt: false` is the documented switch for completing the
/// conversation as given rather than opening a new assistant turn.
/// `continue_final_message` is what Baseten's vLLM and SGLang engines use to
/// leave that turn open; the Model APIs reference does not list it, so it is
/// tried first and dropped if the endpoint rejects it.
pub fn stream_resume_policy() -> StreamResumePolicy {
    let completion_style = ("add_generation_prompt".to_string(), json!(false));
    StreamResumePolicy {
        prefill: true,
        prefill_request_params: vec![
            HashMap::from([
                completion_style.clone(),
                ("continue_final_message".to_string(), json!(true)),
            ]),
            HashMap::from([completion_style]),
        ],
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
}
