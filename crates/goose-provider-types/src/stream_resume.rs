//! Continues a provider response that stopped part-way.
//!
//! Once text has reached the user it cannot be taken back, so a stream that
//! fails mid-response cannot simply be requested again. Where the provider can
//! continue a partial assistant message, the text so far is sent back as a
//! trailing assistant message (a prefill) and the caller sees one uninterrupted
//! response. A response that ran out of output tokens is continued the same way.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_stream::try_stream;
use futures::StreamExt;
use rmcp::model::Tool;
use serde_json::Value;

use crate::base::{MessageStream, Provider};
use crate::conversation::message::{Message, MessageContent};
use crate::conversation::token_usage::ProviderUsage;
use crate::errors::ProviderError;
use crate::model::ModelConfig;
use crate::retry::{should_retry, RetryConfig};
use crate::retry_notice::{self, RetryNotice};

// wasm32 hosts run without a tokio runtime, so back off on the host's timer.
#[cfg(target_arch = "wasm32")]
use gloo_timers::future::sleep;
#[cfg(not(target_arch = "wasm32"))]
use tokio::time::sleep;

pub const DEFAULT_MAX_STREAM_RESUMES: usize = 3;
pub const STREAM_RESUME_ENV: &str = "GOOSE_PROVIDER_STREAM_RESUME";
pub const MAX_STREAM_RESUMES_ENV: &str = "GOOSE_PROVIDER_MAX_STREAM_RESUMES";
pub const STREAM_RESUME_PARAMS_ENV: &str = "GOOSE_PROVIDER_STREAM_RESUME_PARAMS";

/// Ceiling for the output budget a continuation asks for after an output-limit stop.
const MAX_CONTINUATION_TOKENS: i32 = 65_536;

const INCOMPLETE_STREAM_ERROR: &str =
    "The provider's response ended before it was complete — check your network connection.";

/// How a provider's interrupted response can be picked up again.
///
/// The default resumes nothing once content has been produced: it only asks
/// again while the response is still empty, which is safe for any provider.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StreamResumePolicy {
    /// The provider continues a trailing assistant message rather than answering it.
    pub prefill: bool,
    /// Request parameters that make the provider continue the prefill.
    pub prefill_request_params: HashMap<String, Value>,
    /// Every complete response reports a finish reason, so a stream that ends
    /// without one was cut off.
    pub requires_finish_reason: bool,
    /// Output budget the provider applies when a request names none, so that a
    /// continuation after an output-limit stop can ask for more.
    pub default_max_tokens: Option<i32>,
}

type StreamItem = (Option<Message>, Option<ProviderUsage>);

enum AttemptEnd {
    Complete,
    OutputLimit,
    Interrupted(ProviderError),
}

/// Why the response is being asked for again.
enum Continuing {
    AfterInterruption(ProviderError),
    AfterOutputLimit,
}

/// What has already been handed to the caller, which a continuation must build on.
#[derive(Default)]
struct Delivered {
    started: bool,
    /// Whether the caller has been handed any item at all, even an empty one.
    yielded: bool,
    content: bool,
    text: String,
    tool_request: bool,
    /// Id of the message the text belongs to; continuations reuse it so their
    /// chunks extend that message instead of starting a second one.
    anchor_id: Option<String>,
}

impl Delivered {
    fn record(&mut self, message: Option<&Message>) {
        self.yielded = true;
        let Some(message) = message else {
            return;
        };
        self.content |= !message.content.is_empty();
        for content in &message.content {
            match content {
                MessageContent::Text(text) => {
                    if self.anchor_id.is_none() {
                        self.anchor_id = message.id.clone();
                    }
                    self.text.push_str(&text.text);
                }
                MessageContent::ToolRequest(_) => self.tool_request = true,
                _ => {}
            }
        }
    }

    /// A tool call cannot be prefilled, and without prefill support only an
    /// empty response can be asked for again.
    fn can_resume(&self, policy: &StreamResumePolicy) -> bool {
        self.started && !self.tool_request && (policy.prefill || !self.content)
    }
}

fn reports_finish_reason(usage: Option<&ProviderUsage>) -> bool {
    usage
        .and_then(|usage| usage.finish_reasons.as_ref())
        .is_some_and(|reasons| !reasons.is_empty())
}

fn has_tool_request(message: Option<&Message>) -> bool {
    message.is_some_and(|message| {
        message
            .content
            .iter()
            .any(|content| matches!(content, MessageContent::ToolRequest(_)))
    })
}

fn has_usable_tool_request(message: &Message) -> bool {
    message.content.iter().any(
        |content| matches!(content, MessageContent::ToolRequest(request) if request.tool_call.is_ok()),
    )
}

fn has_text(message: &Message) -> bool {
    message
        .content
        .iter()
        .any(|content| matches!(content, MessageContent::Text(text) if !text.text.is_empty()))
}

fn output_limit_marker(id: Option<String>) -> Message {
    let mut marker = Message::assistant();
    marker.id = id;
    marker.metadata.output_token_limit_reached = true;
    marker
}

fn resume_setting() -> Option<String> {
    std::env::var(STREAM_RESUME_ENV)
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
}

fn configured_prefill_params() -> Option<HashMap<String, Value>> {
    let value = std::env::var(STREAM_RESUME_PARAMS_ENV).ok()?;
    match serde_json::from_str(&value) {
        Ok(params) => Some(params),
        Err(error) => {
            tracing::warn!("Ignoring {STREAM_RESUME_PARAMS_ENV}: not a JSON object: {error}");
            None
        }
    }
}

fn max_stream_resumes() -> usize {
    std::env::var(MAX_STREAM_RESUMES_ENV)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(DEFAULT_MAX_STREAM_RESUMES)
}

fn effective_policy(
    provider: &dyn Provider,
    model_config: &ModelConfig,
) -> Option<StreamResumePolicy> {
    // Asking a provider that keeps its own conversation state again would
    // record the turn twice.
    if provider.manages_own_context() {
        return None;
    }
    let mut policy = provider.stream_resume_policy(model_config);
    match resume_setting().as_deref() {
        Some("off" | "false" | "0") => return None,
        Some("prefill") => policy.prefill = true,
        _ => {}
    }
    if let Some(params) = configured_prefill_params() {
        policy.prefill_request_params = params;
    }
    Some(policy)
}

fn backoff_delay(error: &ProviderError, config: &RetryConfig, attempt: usize) -> Duration {
    match error {
        ProviderError::RateLimitExceeded {
            retry_delay: Some(provider_delay),
            ..
        } => *provider_delay,
        _ => config.delay_for_attempt(attempt),
    }
}

fn skip_backoff() -> bool {
    std::env::var("GOOSE_PROVIDER_SKIP_BACKOFF")
        .unwrap_or_default()
        .parse::<bool>()
        .unwrap_or(false)
}

fn raise_output_budget(current: Option<i32>, policy: &StreamResumePolicy) -> Option<i32> {
    let budget = current.or(policy.default_max_tokens)?;
    Some(
        budget
            .saturating_mul(2)
            .min(MAX_CONTINUATION_TOKENS)
            .max(budget),
    )
}

/// The request being answered, kept so it can be issued again.
struct Request {
    provider: Arc<dyn Provider>,
    model_config: ModelConfig,
    system: String,
    messages: Vec<Message>,
    tools: Vec<Tool>,
}

impl Request {
    async fn continuation(
        &self,
        max_tokens: Option<i32>,
        delivered: &Delivered,
        policy: &StreamResumePolicy,
    ) -> Result<MessageStream, ProviderError> {
        let mut model_config = self.model_config.clone();
        model_config.max_tokens = max_tokens;
        if delivered.text.is_empty() {
            return self
                .provider
                .stream(&model_config, &self.system, &self.messages, &self.tools)
                .await;
        }

        let model_config =
            model_config.with_merged_request_params(policy.prefill_request_params.clone());
        let mut messages = self.messages.clone();
        messages.push(Message::assistant().with_text(&delivered.text));
        self.provider
            .stream(&model_config, &self.system, &messages, &self.tools)
            .await
    }
}

/// `Provider::stream`, except that a response which stops part-way is continued
/// where the provider's [`StreamResumePolicy`] allows it.
///
/// A failure before the first item is returned as is: the caller's own retry
/// handling already covers a response that never started.
pub async fn stream(
    provider: Arc<dyn Provider>,
    model_config: &ModelConfig,
    system: &str,
    messages: &[Message],
    tools: &[Tool],
) -> Result<MessageStream, ProviderError> {
    let first = provider
        .stream(model_config, system, messages, tools)
        .await?;
    let Some(policy) = effective_policy(provider.as_ref(), model_config) else {
        return Ok(first);
    };
    let request = Request {
        provider,
        model_config: model_config.clone(),
        system: system.to_owned(),
        messages: messages.to_vec(),
        tools: tools.to_vec(),
    };

    Ok(Box::pin(try_stream! {
        let retry_config = request.provider.retry_config().transient_only();
        let max_resumes = max_stream_resumes();
        let mut stream = first;
        let mut delivered = Delivered::default();
        let mut interruptions = 0;
        let mut output_limit_continuations = 0;
        let mut max_tokens = request.model_config.max_tokens;
        let mut continuing: Option<Continuing> = None;

        loop {
            let mut finish_seen = false;
            let mut output_limit_reached = false;
            let progress_before = (delivered.text.len(), delivered.tool_request);
            // A cut-off stream still yields the tool calls it was part-way through,
            // with truncated arguments. They are held until the response proves complete.
            let mut held: Vec<StreamItem> = Vec::new();

            let end = loop {
                let (mut message, usage) = match stream.next().await {
                    Some(Ok(item)) => item,
                    Some(Err(error)) => break AttemptEnd::Interrupted(error),
                    None if policy.requires_finish_reason && !finish_seen => {
                        break AttemptEnd::Interrupted(ProviderError::NetworkError(
                            INCOMPLETE_STREAM_ERROR.to_string(),
                        ));
                    }
                    None if output_limit_reached => break AttemptEnd::OutputLimit,
                    None => break AttemptEnd::Complete,
                };
                delivered.started = true;
                finish_seen |= reports_finish_reason(usage.as_ref());

                if let Some(limited) = message
                    .as_mut()
                    .filter(|message| message.metadata.output_token_limit_reached)
                {
                    // With no text to continue from, asking again only helps if
                    // the next request gets more room than this one had.
                    let budget = max_tokens.or(policy.default_max_tokens);
                    let can_continue = policy.prefill
                        && output_limit_continuations < max_resumes
                        && !delivered.tool_request
                        && !has_usable_tool_request(limited)
                        && (!delivered.text.is_empty()
                            || has_text(limited)
                            || raise_output_budget(max_tokens, &policy) > budget);
                    if can_continue {
                        output_limit_reached = true;
                        limited.metadata.output_token_limit_reached = false;
                        limited
                            .content
                            .retain(|content| !matches!(content, MessageContent::ToolRequest(_)));
                    }
                }
                // Only the truncated tool calls were left in it; there is nothing to deliver.
                let emptied = output_limit_reached
                    && message.as_ref().is_some_and(|message| message.content.is_empty());
                if emptied {
                    message = None;
                }
                if continuing.is_some() {
                    if let (Some(message), Some(anchor_id)) =
                        (message.as_mut(), &delivered.anchor_id)
                    {
                        message.id = Some(anchor_id.clone());
                    }
                }

                let hold = policy.requires_finish_reason
                    && !finish_seen
                    && (!held.is_empty() || has_tool_request(message.as_ref()));
                if hold {
                    held.push((message, usage));
                    continue;
                }
                for (held_message, held_usage) in held.drain(..) {
                    delivered.record(held_message.as_ref());
                    yield (held_message, held_usage);
                }
                if !emptied || usage.is_some() {
                    delivered.record(message.as_ref());
                    yield (message, usage);
                }
            };

            let prefilled = continuing.is_some() && progress_before.0 > 0;
            let no_progress =
                progress_before == (delivered.text.len(), delivered.tool_request);
            let next = match end {
                // A prefilled request that adds nothing was not a continuation:
                // the model saw a finished turn and had nothing to say.
                AttemptEnd::Complete if prefilled && no_progress => {
                    match continuing {
                        Some(Continuing::AfterInterruption(error)) => {
                            Err::<(), ProviderError>(error)?
                        }
                        _ => {
                            let marker = output_limit_marker(delivered.anchor_id.clone());
                            yield (Some(marker), None);
                        }
                    }
                    break;
                }
                AttemptEnd::Complete => break,
                AttemptEnd::OutputLimit => {
                    output_limit_continuations += 1;
                    retry_notice::notify(RetryNotice::ContinuingAfterOutputLimit {
                        attempt: output_limit_continuations,
                        max_attempts: max_resumes,
                    });
                    max_tokens = raise_output_budget(max_tokens, &policy);
                    Continuing::AfterOutputLimit
                }
                AttemptEnd::Interrupted(error) => {
                    let resumable = should_retry(&error, &retry_config)
                        && interruptions < max_resumes
                        && delivered.can_resume(&policy);
                    if !resumable {
                        Err::<(), ProviderError>(error)?;
                        break;
                    }
                    interruptions += 1;
                    tracing::warn!(
                        "Provider stream interrupted, resuming ({}/{}): {:?}",
                        interruptions,
                        max_resumes,
                        error
                    );
                    retry_notice::notify(RetryNotice::ResumingStream {
                        error: error.clone(),
                        attempt: interruptions,
                        max_attempts: max_resumes,
                    });
                    if !skip_backoff() {
                        sleep(backoff_delay(&error, &retry_config, interruptions)).await;
                    }
                    Continuing::AfterInterruption(error)
                }
            };

            // A caller that has seen nothing yet would take a later failure for
            // a response that never started and ask for the whole thing again.
            if !delivered.yielded {
                delivered.yielded = true;
                yield (None, None);
            }
            stream = match request.continuation(max_tokens, &delivered, &policy).await {
                Ok(stream) => stream,
                Err(continuation_error) => {
                    tracing::warn!(
                        "Could not continue the provider response: {:?}",
                        continuation_error
                    );
                    // The interruption is what the user needs to hear about; a
                    // response that merely ran out of tokens still stands.
                    match next {
                        Continuing::AfterInterruption(error) => {
                            Err::<(), ProviderError>(error)?
                        }
                        Continuing::AfterOutputLimit => {
                            let marker = output_limit_marker(delivered.anchor_id.clone());
                            yield (Some(marker), None);
                        }
                    }
                    break;
                }
            };
            continuing = Some(next);
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::token_usage::Usage;
    use crate::retry_notice::RetryObserver;
    use async_trait::async_trait;
    use rmcp::model::{CallToolRequestParams, ErrorCode, ErrorData};
    use serde_json::json;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    type Scripted = Result<Vec<Result<StreamItem, ProviderError>>, ProviderError>;

    struct Call {
        max_tokens: Option<i32>,
        request_params: Option<HashMap<String, Value>>,
        messages: Vec<Message>,
    }

    struct ScriptedProvider {
        policy: StreamResumePolicy,
        manages_own_context: bool,
        responses: Mutex<VecDeque<Scripted>>,
        calls: Mutex<Vec<Call>>,
    }

    impl ScriptedProvider {
        fn new(policy: StreamResumePolicy, responses: Vec<Scripted>) -> Arc<Self> {
            Arc::new(Self {
                policy,
                manages_own_context: false,
                responses: Mutex::new(responses.into()),
                calls: Mutex::new(Vec::new()),
            })
        }

        fn call_count(&self) -> usize {
            self.calls.lock().unwrap().len()
        }

        fn last_text_sent(&self, call: usize) -> String {
            let calls = self.calls.lock().unwrap();
            let last = calls[call].messages.last().unwrap();
            assert_eq!(last.role, rmcp::model::Role::Assistant);
            last.as_concat_text()
        }
    }

    #[async_trait]
    impl Provider for ScriptedProvider {
        fn get_name(&self) -> &str {
            "scripted"
        }

        async fn stream(
            &self,
            model_config: &ModelConfig,
            _system: &str,
            messages: &[Message],
            _tools: &[Tool],
        ) -> Result<MessageStream, ProviderError> {
            self.calls.lock().unwrap().push(Call {
                max_tokens: model_config.max_tokens,
                request_params: model_config.request_params.clone(),
                messages: messages.to_vec(),
            });
            let items = self
                .responses
                .lock()
                .unwrap()
                .pop_front()
                .expect("provider was asked for more responses than the test scripted")?;
            Ok(Box::pin(futures::stream::iter(items)))
        }

        fn retry_config(&self) -> RetryConfig {
            RetryConfig::new(3, 0, 1.0, 0)
        }

        fn stream_resume_policy(&self, _model_config: &ModelConfig) -> StreamResumePolicy {
            self.policy.clone()
        }

        fn manages_own_context(&self) -> bool {
            self.manages_own_context
        }
    }

    fn prefill_policy() -> StreamResumePolicy {
        StreamResumePolicy {
            prefill: true,
            prefill_request_params: HashMap::from([(
                "add_generation_prompt".to_string(),
                json!(false),
            )]),
            requires_finish_reason: true,
            default_max_tokens: Some(4096),
        }
    }

    fn text(id: &str, text: &str) -> Result<StreamItem, ProviderError> {
        Ok((Some(Message::assistant().with_id(id).with_text(text)), None))
    }

    fn thinking(id: &str, text: &str) -> Result<StreamItem, ProviderError> {
        Ok((
            Some(
                Message::assistant()
                    .with_id(id)
                    .with_content(MessageContent::thinking(text, "")),
            ),
            None,
        ))
    }

    fn finished(reason: &str) -> Result<StreamItem, ProviderError> {
        let usage = ProviderUsage::new("model".to_string(), Usage::default())
            .with_finish_reasons(vec![reason.to_string()]);
        Ok((None, Some(usage)))
    }

    fn tool_call(id: &str, name: &str) -> Result<StreamItem, ProviderError> {
        Ok((
            Some(
                Message::assistant()
                    .with_id(id)
                    .with_tool_request("call_1", Ok(CallToolRequestParams::new(name.to_string()))),
            ),
            None,
        ))
    }

    fn truncated_tool_call(id: &str, limit_reached: bool) -> Result<StreamItem, ProviderError> {
        let mut message = Message::assistant().with_id(id).with_tool_request(
            "call_1",
            Err(ErrorData {
                code: ErrorCode::INVALID_PARAMS,
                message: "arguments were truncated".into(),
                data: None,
            }),
        );
        message.metadata.output_token_limit_reached = limit_reached;
        Ok((Some(message), None))
    }

    fn limit_marker(id: &str) -> Result<StreamItem, ProviderError> {
        Ok((Some(output_limit_marker(Some(id.to_string()))), None))
    }

    fn connection_reset() -> ProviderError {
        ProviderError::NetworkError("connection reset".to_string())
    }

    struct Outcome {
        messages: Vec<Message>,
        error: Option<ProviderError>,
    }

    impl Outcome {
        fn text(&self) -> String {
            self.messages
                .iter()
                .map(Message::as_concat_text)
                .collect::<String>()
        }

        fn tool_requests(&self) -> Vec<bool> {
            self.messages
                .iter()
                .flat_map(|message| message.content.iter())
                .filter_map(MessageContent::as_tool_request)
                .map(|request| request.tool_call.is_ok())
                .collect()
        }

        fn output_limit_reached(&self) -> bool {
            self.messages
                .iter()
                .any(|message| message.metadata.output_token_limit_reached)
        }
    }

    /// Pins the settings these tests depend on, whatever the developer's shell
    /// has exported, for as long as the guard lives.
    fn settings() -> env_lock::EnvGuard<'static> {
        env_lock::lock_env([
            (STREAM_RESUME_ENV, None::<&str>),
            (MAX_STREAM_RESUMES_ENV, None),
            (STREAM_RESUME_PARAMS_ENV, None),
            ("GOOSE_PROVIDER_SKIP_BACKOFF", None),
        ])
    }

    async fn run(provider: &Arc<ScriptedProvider>) -> Outcome {
        let _settings = settings();
        let mut outcome = Outcome {
            messages: Vec::new(),
            error: None,
        };
        let stream = stream(
            provider.clone(),
            &ModelConfig::new("model"),
            "system",
            &[Message::user().with_text("hello")],
            &[],
        )
        .await;
        let mut stream = match stream {
            Ok(stream) => stream,
            Err(error) => {
                outcome.error = Some(error);
                return outcome;
            }
        };
        while let Some(item) = stream.next().await {
            match item {
                Ok((message, _)) => outcome.messages.extend(message),
                Err(error) => outcome.error = Some(error),
            }
        }
        outcome
    }

    #[tokio::test]
    async fn interrupted_text_is_continued_from_a_prefill() {
        let provider = ScriptedProvider::new(
            prefill_policy(),
            vec![
                Ok(vec![text("first", "Hello, "), Err(connection_reset())]),
                Ok(vec![text("second", "world"), finished("stop")]),
            ],
        );

        let outcome = run(&provider).await;

        assert_eq!(outcome.error, None);
        assert_eq!(outcome.text(), "Hello, world");
        assert!(
            outcome
                .messages
                .iter()
                .all(|message| message.id.as_deref() == Some("first")),
            "the continuation extends the message the user is already reading"
        );
        assert_eq!(provider.last_text_sent(1), "Hello, ");
        let calls = provider.calls.lock().unwrap();
        assert_eq!(calls[0].request_params, None);
        assert_eq!(
            calls[1].request_params.as_ref().unwrap()["add_generation_prompt"],
            json!(false)
        );
    }

    #[tokio::test]
    async fn stream_that_ends_without_a_finish_reason_is_resumed() {
        let provider = ScriptedProvider::new(
            prefill_policy(),
            vec![
                Ok(vec![text("first", "Hel")]),
                Ok(vec![text("second", "lo"), finished("stop")]),
            ],
        );

        let outcome = run(&provider).await;

        assert_eq!(outcome.error, None);
        assert_eq!(outcome.text(), "Hello");
    }

    #[tokio::test]
    async fn stream_without_a_finish_reason_is_complete_unless_the_policy_requires_one() {
        let provider = ScriptedProvider::new(
            StreamResumePolicy::default(),
            vec![Ok(vec![text("first", "Hello")])],
        );

        let outcome = run(&provider).await;

        assert_eq!(outcome.error, None);
        assert_eq!(outcome.text(), "Hello");
        assert_eq!(provider.call_count(), 1);
    }

    #[tokio::test]
    async fn tool_call_from_a_cut_off_stream_is_withheld_and_requested_again() {
        let provider = ScriptedProvider::new(
            prefill_policy(),
            vec![
                Ok(vec![
                    text("first", "Let me check. "),
                    truncated_tool_call("first", false),
                ]),
                Ok(vec![tool_call("second", "shell"), finished("tool_calls")]),
            ],
        );

        let outcome = run(&provider).await;

        assert_eq!(outcome.error, None);
        assert_eq!(outcome.tool_requests(), vec![true]);
        assert_eq!(provider.last_text_sent(1), "Let me check. ");
    }

    #[tokio::test]
    async fn complete_tool_call_is_delivered_once_the_finish_reason_arrives() {
        let provider = ScriptedProvider::new(
            prefill_policy(),
            vec![Ok(vec![
                tool_call("first", "shell"),
                finished("tool_calls"),
            ])],
        );

        let outcome = run(&provider).await;

        assert_eq!(outcome.error, None);
        assert_eq!(outcome.tool_requests(), vec![true]);
        assert_eq!(provider.call_count(), 1);
    }

    #[tokio::test]
    async fn interruption_after_text_is_an_error_without_prefill_support() {
        let provider = ScriptedProvider::new(
            StreamResumePolicy::default(),
            vec![Ok(vec![text("first", "Hello, "), Err(connection_reset())])],
        );

        let outcome = run(&provider).await;

        assert_eq!(outcome.error, Some(connection_reset()));
        assert_eq!(outcome.text(), "Hello, ");
        assert_eq!(provider.call_count(), 1);
    }

    #[tokio::test]
    async fn interruption_before_any_content_asks_again_for_any_provider() {
        let provider = ScriptedProvider::new(
            StreamResumePolicy::default(),
            vec![
                Ok(vec![Ok((None, None)), Err(connection_reset())]),
                Ok(vec![text("second", "Hello"), finished("stop")]),
            ],
        );

        let outcome = run(&provider).await;

        assert_eq!(outcome.error, None);
        assert_eq!(outcome.text(), "Hello");
        assert_eq!(provider.calls.lock().unwrap()[1].messages.len(), 1);
    }

    #[tokio::test]
    async fn failure_before_the_first_item_is_left_to_the_caller() {
        let provider =
            ScriptedProvider::new(prefill_policy(), vec![Ok(vec![Err(connection_reset())])]);

        let outcome = run(&provider).await;

        assert_eq!(outcome.error, Some(connection_reset()));
        assert_eq!(provider.call_count(), 1);
    }

    #[tokio::test]
    async fn interruption_during_reasoning_asks_again_without_a_prefill() {
        let provider = ScriptedProvider::new(
            prefill_policy(),
            vec![
                Ok(vec![
                    thinking("first", "Considering"),
                    Err(connection_reset()),
                ]),
                Ok(vec![
                    thinking("second", "Considering the question."),
                    text("second", "Hello"),
                    finished("stop"),
                ]),
            ],
        );

        let outcome = run(&provider).await;

        assert_eq!(outcome.error, None);
        assert_eq!(outcome.text(), "Hello");
        let calls = provider.calls.lock().unwrap();
        assert_eq!(calls[1].messages.len(), 1);
        assert_eq!(calls[1].request_params, None);
        assert_eq!(
            outcome.messages.last().unwrap().id.as_deref(),
            Some("second")
        );
    }

    #[tokio::test]
    async fn interruption_that_cannot_be_retried_is_an_error() {
        let refusal = ProviderError::Refusal {
            details: "declined".to_string(),
            category: None,
        };
        let provider = ScriptedProvider::new(
            prefill_policy(),
            vec![Ok(vec![text("first", "Hello, "), Err(refusal.clone())])],
        );

        let outcome = run(&provider).await;

        assert_eq!(outcome.error, Some(refusal));
        assert_eq!(provider.call_count(), 1);
    }

    #[tokio::test]
    async fn repeated_interruptions_give_up_after_the_resume_limit() {
        let interrupted = || Ok(vec![text("chunk", "a"), Err(connection_reset())]);
        let provider = ScriptedProvider::new(
            prefill_policy(),
            vec![interrupted(), interrupted(), interrupted(), interrupted()],
        );

        let outcome = run(&provider).await;

        assert_eq!(outcome.error, Some(connection_reset()));
        assert_eq!(outcome.text(), "aaaa");
        assert_eq!(provider.call_count(), 1 + DEFAULT_MAX_STREAM_RESUMES);
    }

    #[tokio::test]
    async fn refused_continuation_reports_the_interruption() {
        let provider = ScriptedProvider::new(
            prefill_policy(),
            vec![
                Ok(vec![text("first", "Hello, "), Err(connection_reset())]),
                Err(ProviderError::RequestFailed(
                    "Bad request (400)".to_string(),
                )),
            ],
        );

        let outcome = run(&provider).await;

        assert_eq!(outcome.error, Some(connection_reset()));
        assert_eq!(outcome.text(), "Hello, ");
    }

    #[tokio::test]
    async fn prefilled_request_that_adds_nothing_reports_the_interruption() {
        let provider = ScriptedProvider::new(
            prefill_policy(),
            vec![
                Ok(vec![text("first", "Hello, "), Err(connection_reset())]),
                Ok(vec![finished("stop")]),
            ],
        );

        let outcome = run(&provider).await;

        assert_eq!(outcome.error, Some(connection_reset()));
        assert_eq!(outcome.text(), "Hello, ");
    }

    #[tokio::test]
    async fn cut_off_tool_call_with_nothing_delivered_is_not_mistaken_for_an_unstarted_response() {
        let cut_off = || Ok(vec![truncated_tool_call("chunk", false)]);
        let provider = ScriptedProvider::new(
            prefill_policy(),
            vec![cut_off(), cut_off(), cut_off(), cut_off()],
        );

        let _settings = settings();
        let stream = stream(
            provider.clone(),
            &ModelConfig::new("model"),
            "system",
            &[Message::user().with_text("hello")],
            &[],
        )
        .await
        .unwrap();
        let items: Vec<_> = stream.collect().await;

        assert!(
            matches!(items.first(), Some(Ok((None, None)))),
            "the caller retries a stream whose first item is an error: {items:?}"
        );
        assert!(matches!(
            items.last(),
            Some(Err(ProviderError::NetworkError(_)))
        ));
        assert_eq!(provider.call_count(), 1 + DEFAULT_MAX_STREAM_RESUMES);
    }

    #[tokio::test]
    async fn output_limit_is_continued_with_a_larger_budget() {
        let provider = ScriptedProvider::new(
            prefill_policy(),
            vec![
                Ok(vec![
                    text("first", "Part one, "),
                    truncated_tool_call("first", true),
                    finished("length"),
                ]),
                Ok(vec![text("second", "part two."), finished("stop")]),
            ],
        );

        let outcome = run(&provider).await;

        assert_eq!(outcome.error, None);
        assert_eq!(outcome.text(), "Part one, part two.");
        assert!(!outcome.output_limit_reached());
        assert!(outcome.tool_requests().is_empty());
        assert_eq!(provider.last_text_sent(1), "Part one, ");
        let calls = provider.calls.lock().unwrap();
        assert_eq!(calls[0].max_tokens, None);
        assert_eq!(calls[1].max_tokens, Some(8192));
    }

    #[tokio::test]
    async fn output_limit_is_reported_when_continuations_run_out() {
        let limited = || {
            Ok(vec![
                text("chunk", "more "),
                limit_marker("chunk"),
                finished("length"),
            ])
        };
        let provider = ScriptedProvider::new(
            prefill_policy(),
            vec![limited(), limited(), limited(), limited()],
        );

        let outcome = run(&provider).await;

        assert_eq!(outcome.error, None);
        assert_eq!(outcome.text(), "more more more more ");
        assert!(outcome.output_limit_reached());
        let budgets: Vec<_> = provider
            .calls
            .lock()
            .unwrap()
            .iter()
            .map(|call| call.max_tokens)
            .collect();
        assert_eq!(budgets, vec![None, Some(8192), Some(16_384), Some(32_768)]);
    }

    #[tokio::test]
    async fn output_limit_without_text_is_only_asked_again_with_a_larger_budget() {
        let mut policy = prefill_policy();
        policy.default_max_tokens = None;
        let provider = ScriptedProvider::new(
            policy,
            vec![Ok(vec![
                thinking("first", "Considering"),
                limit_marker("first"),
                finished("length"),
            ])],
        );

        let outcome = run(&provider).await;

        assert!(outcome.output_limit_reached());
        assert_eq!(provider.call_count(), 1);
    }

    #[tokio::test]
    async fn prefilled_request_that_adds_nothing_still_reports_the_output_limit() {
        let provider = ScriptedProvider::new(
            prefill_policy(),
            vec![
                Ok(vec![
                    text("first", "Part one, "),
                    limit_marker("first"),
                    finished("length"),
                ]),
                Ok(vec![finished("stop")]),
            ],
        );

        let outcome = run(&provider).await;

        assert_eq!(outcome.error, None);
        assert_eq!(outcome.text(), "Part one, ");
        assert!(outcome.output_limit_reached());
    }

    #[tokio::test]
    async fn output_limit_is_reported_as_is_without_prefill_support() {
        let provider = ScriptedProvider::new(
            StreamResumePolicy::default(),
            vec![Ok(vec![
                text("first", "Part one, "),
                limit_marker("first"),
                finished("length"),
            ])],
        );

        let outcome = run(&provider).await;

        assert!(outcome.output_limit_reached());
        assert_eq!(provider.call_count(), 1);
    }

    #[tokio::test]
    async fn refused_output_limit_continuation_still_reports_the_limit() {
        let provider = ScriptedProvider::new(
            prefill_policy(),
            vec![
                Ok(vec![
                    text("first", "Part one, "),
                    limit_marker("first"),
                    finished("length"),
                ]),
                Err(ProviderError::RequestFailed(
                    "Bad request (400)".to_string(),
                )),
            ],
        );

        let outcome = run(&provider).await;

        assert_eq!(outcome.error, None);
        assert_eq!(outcome.text(), "Part one, ");
        assert!(outcome.output_limit_reached());
        assert_eq!(
            outcome.messages.last().unwrap().id.as_deref(),
            Some("first")
        );
    }

    #[tokio::test]
    async fn provider_that_manages_its_own_context_is_never_asked_again() {
        let provider = Arc::new(ScriptedProvider {
            policy: prefill_policy(),
            manages_own_context: true,
            responses: Mutex::new(
                vec![Ok(vec![text("first", "Hello, "), Err(connection_reset())])].into(),
            ),
            calls: Mutex::new(Vec::new()),
        });

        let outcome = run(&provider).await;

        assert_eq!(outcome.error, Some(connection_reset()));
        assert_eq!(provider.call_count(), 1);
    }

    #[tokio::test]
    async fn resuming_and_continuing_are_announced() {
        let provider = ScriptedProvider::new(
            prefill_policy(),
            vec![
                Ok(vec![text("first", "One, "), Err(connection_reset())]),
                Ok(vec![
                    text("second", "two, "),
                    limit_marker("second"),
                    finished("length"),
                ]),
                Ok(vec![text("third", "three."), finished("stop")]),
            ],
        );
        let notices = Arc::new(Mutex::new(Vec::new()));
        let sink = notices.clone();
        let observer: RetryObserver = Arc::new(move |notice: &RetryNotice| {
            sink.lock().unwrap().push(notice.clone());
        });

        let outcome = retry_notice::observe(observer, run(&provider)).await;

        assert_eq!(outcome.text(), "One, two, three.");
        assert_eq!(
            *notices.lock().unwrap(),
            vec![
                RetryNotice::ResumingStream {
                    error: connection_reset(),
                    attempt: 1,
                    max_attempts: DEFAULT_MAX_STREAM_RESUMES,
                },
                RetryNotice::ContinuingAfterOutputLimit {
                    attempt: 1,
                    max_attempts: DEFAULT_MAX_STREAM_RESUMES,
                },
            ]
        );
    }
}
