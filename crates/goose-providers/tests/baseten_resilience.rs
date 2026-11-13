//! How the OpenAI-compatible provider behaves against the failures Baseten
//! documents: rate limits, overload, and streams that stop without finishing.

use async_trait::async_trait;
use futures::StreamExt;
use goose_providers::{
    api_client::{ApiClient, AuthMethod},
    base::{MessageStream, Provider},
    baseten,
    conversation::message::Message,
    errors::ProviderError,
    model::ModelConfig,
    openai::{OpenAiProvider, OpenAiProviderBuilder},
    retry::RetryConfig,
    retry_notice::{self, RetryNotice, RetryObserver},
    stream_resume::{self, StreamResumePolicy},
};
use rmcp::model::Tool;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use wiremock::{
    matchers::{body_partial_json, method, path},
    Mock, MockServer, ResponseTemplate,
};

const MODEL: &str = "zai-org/GLM-5.3";

fn provider_at(base_url: &str) -> OpenAiProvider {
    let api_client =
        ApiClient::new_with_tls(base_url.to_string(), AuthMethod::NoAuth, None).unwrap();
    OpenAiProviderBuilder::new(api_client).build()
}

/// Baseten is recognised by host, which a mock server cannot have, so this
/// gives the provider under test the policy it would pick for itself there.
struct AsBaseten(OpenAiProvider);

#[async_trait]
impl Provider for AsBaseten {
    fn get_name(&self) -> &str {
        self.0.get_name()
    }

    async fn stream(
        &self,
        model_config: &ModelConfig,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        self.0.stream(model_config, system, messages, tools).await
    }

    fn retry_config(&self) -> RetryConfig {
        RetryConfig::new(3, 0, 1.0, 0).transient_only()
    }

    fn stream_resume_policy(&self, _model_config: &ModelConfig) -> StreamResumePolicy {
        baseten::stream_resume_policy()
    }
}

fn content_chunk(text: &str) -> String {
    format!(
        "data: {}\n\n",
        json!({"id": "chatcmpl-1", "model": MODEL,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]})
    )
}

fn finished(reason: &str) -> String {
    format!(
        "data: {}\n\ndata: [DONE]\n\n",
        json!({"id": "chatcmpl-1", "model": MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}})
    )
}

fn sse(body: String) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_raw(body, "text/event-stream")
}

fn chat_completions() -> wiremock::MockBuilder {
    Mock::given(method("POST")).and(path("/v1/chat/completions"))
}

async fn collect_text(mut stream: MessageStream) -> Result<String, ProviderError> {
    let mut text = String::new();
    while let Some(item) = stream.next().await {
        if let (Some(message), _) = item? {
            text.push_str(&message.as_concat_text());
        }
    }
    Ok(text)
}

fn user_prompt() -> Vec<Message> {
    vec![Message::user().with_text("Say hello")]
}

fn recording_observer() -> (RetryObserver, Arc<Mutex<Vec<RetryNotice>>>) {
    let notices = Arc::new(Mutex::new(Vec::new()));
    let sink = notices.clone();
    let observer: RetryObserver = Arc::new(move |notice: &RetryNotice| {
        sink.lock().unwrap().push(notice.clone());
    });
    (observer, notices)
}

async fn request_bodies(server: &MockServer) -> Vec<Value> {
    server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .map(|request| serde_json::from_slice(&request.body).unwrap())
        .collect()
}

#[tokio::test]
async fn stream_cut_off_mid_answer_is_continued_from_a_prefill() {
    let server = MockServer::start().await;
    chat_completions()
        .and(body_partial_json(json!({"continue_final_message": true})))
        .respond_with(sse(format!(
            "{}{}",
            content_chunk("world"),
            finished("stop")
        )))
        .expect(1)
        .mount(&server)
        .await;
    // Ends with neither a finish_reason nor [DONE].
    chat_completions()
        .respond_with(sse(content_chunk("Hello, ")))
        .expect(1)
        .mount(&server)
        .await;
    let provider = Arc::new(AsBaseten(provider_at(&server.uri())));
    let (observer, notices) = recording_observer();

    let text = retry_notice::observe(observer, async {
        let stream = stream_resume::stream(
            provider,
            &ModelConfig::new(MODEL),
            "system",
            &user_prompt(),
            &[],
        )
        .await?;
        collect_text(stream).await
    })
    .await
    .unwrap();

    assert_eq!(text, "Hello, world");
    let bodies = request_bodies(&server).await;
    assert_eq!(bodies[0].get("add_generation_prompt"), None);
    assert_eq!(bodies[1]["add_generation_prompt"], json!(false));
    assert_eq!(
        bodies[1]["messages"].as_array().unwrap().last().unwrap(),
        &json!({"role": "assistant", "content": "Hello, "})
    );
    assert!(matches!(
        notices.lock().unwrap().as_slice(),
        [RetryNotice::ResumingStream { attempt: 1, .. }]
    ));
}

#[tokio::test]
async fn output_limit_is_continued_with_twice_the_default_budget() {
    let server = MockServer::start().await;
    chat_completions()
        .and(body_partial_json(json!({"max_tokens": 8192})))
        .respond_with(sse(format!(
            "{}{}",
            content_chunk("part two."),
            finished("stop")
        )))
        .expect(1)
        .mount(&server)
        .await;
    chat_completions()
        .respond_with(sse(format!(
            "{}{}",
            content_chunk("Part one, "),
            finished("length")
        )))
        .expect(1)
        .mount(&server)
        .await;
    let provider = Arc::new(AsBaseten(provider_at(&server.uri())));

    let stream = stream_resume::stream(
        provider,
        &ModelConfig::new(MODEL),
        "system",
        &user_prompt(),
        &[],
    )
    .await
    .unwrap();

    assert_eq!(collect_text(stream).await.unwrap(), "Part one, part two.");
    let bodies = request_bodies(&server).await;
    assert_eq!(bodies[0].get("max_tokens"), None);
}

#[tokio::test]
async fn overloaded_response_is_retried_after_the_wait_it_asks_for() {
    let server = MockServer::start().await;
    chat_completions()
        .respond_with(
            ResponseTemplate::new(529)
                .insert_header("retry-after", "0")
                .insert_header("x-baseten-request-id", "req_overloaded")
                .set_body_json(json!({"error": "no capacity"})),
        )
        .up_to_n_times(1)
        .expect(1)
        .mount(&server)
        .await;
    chat_completions()
        .respond_with(sse(format!(
            "{}{}",
            content_chunk("Hello"),
            finished("stop")
        )))
        .expect(1)
        .mount(&server)
        .await;
    let provider = provider_at(&server.uri());
    let (observer, notices) = recording_observer();

    let text = retry_notice::observe(observer, async {
        let stream = provider
            .stream(&ModelConfig::new(MODEL), "system", &user_prompt(), &[])
            .await?;
        collect_text(stream).await
    })
    .await
    .unwrap();

    assert_eq!(text, "Hello");
    let notices = notices.lock().unwrap();
    let [RetryNotice::Retrying {
        error:
            ProviderError::RateLimitExceeded {
                details,
                retry_delay,
            },
        attempt: 1,
        delay,
        ..
    }] = notices.as_slice()
    else {
        panic!("expected one retry notice for the overload, got {notices:?}");
    };
    assert_eq!(*retry_delay, Some(Duration::ZERO));
    assert_eq!(*delay, Duration::ZERO);
    assert!(details.contains("529"), "{details}");
    assert!(
        details.ends_with("(provider request id: req_overloaded)"),
        "{details}"
    );
}

#[tokio::test]
async fn rate_limit_notice_names_the_limit_that_ran_out() {
    let server = MockServer::start().await;
    chat_completions()
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "0")
                .insert_header("x-ratelimit-limit-requests", "120")
                .insert_header("x-ratelimit-remaining-requests", "0")
                .insert_header("x-ratelimit-limit-tokens", "500000")
                .insert_header("x-ratelimit-remaining-tokens", "4321")
                .set_body_json(json!({"error": {"message": "Rate limit exceeded"}})),
        )
        .up_to_n_times(2)
        .expect(2)
        .mount(&server)
        .await;
    chat_completions()
        .respond_with(sse(format!(
            "{}{}",
            content_chunk("Hello"),
            finished("stop")
        )))
        .expect(1)
        .mount(&server)
        .await;
    let provider = provider_at(&server.uri());
    let (observer, notices) = recording_observer();

    retry_notice::observe(observer, async {
        let stream = provider
            .stream(&ModelConfig::new(MODEL), "system", &user_prompt(), &[])
            .await?;
        collect_text(stream).await
    })
    .await
    .unwrap();

    let messages: Vec<String> = notices
        .lock()
        .unwrap()
        .iter()
        .map(RetryNotice::message)
        .collect();
    assert_eq!(
        messages,
        vec![
            "Rate limit exceeded: Rate limit exceeded (requests left: 0/120, tokens left: \
             4321/500000). Retrying in under a second (attempt 1 of 6).",
            "Rate limit exceeded: Rate limit exceeded (requests left: 0/120, tokens left: \
             4321/500000). Retrying in under a second (attempt 2 of 6).",
        ]
    );
}

#[test]
fn baseten_host_selects_its_retry_and_resume_behaviour() {
    let model = ModelConfig::new(MODEL);

    let baseten = provider_at("https://inference.baseten.co");
    assert!(baseten.retry_config().transient_only);
    assert_eq!(
        baseten.stream_resume_policy(&model),
        baseten::stream_resume_policy()
    );
    // Routed to the Responses API, which has its own completion events.
    assert_eq!(
        baseten.stream_resume_policy(&ModelConfig::new("openai/gpt-5")),
        StreamResumePolicy::default()
    );

    let elsewhere = provider_at("https://api.example.com");
    assert!(!elsewhere.retry_config().transient_only);
    assert_eq!(
        elsewhere.stream_resume_policy(&model),
        StreamResumePolicy::default()
    );
}
