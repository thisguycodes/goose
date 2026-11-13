//! Checks against the real Baseten Model APIs what the mock-server tests can
//! only assume: that a prefilled request is continued rather than answered
//! afresh.
//!
//! ```bash
//! BASETEN_API_KEY=... cargo test -p goose-providers --test baseten_live -- --ignored --nocapture
//! ```
//!
//! `BASETEN_MODEL` picks the model slug.

use async_trait::async_trait;
use futures::StreamExt;
use goose_providers::{
    api_client::{ApiClient, AuthMethod},
    base::{MessageStream, Provider},
    conversation::message::{Message, MessageContent},
    errors::ProviderError,
    model::ModelConfig,
    openai::{OpenAiProvider, OpenAiProviderBuilder},
    retry::RetryConfig,
    retry_notice::{self, RetryNotice, RetryObserver},
    stream_resume::{self, StreamResumePolicy},
};
use rmcp::model::Tool;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const MODEL_APIS: &str = "https://inference.baseten.co";
const DEFAULT_MODEL: &str = "zai-org/GLM-5.3";
const TEXT_CHUNKS_BEFORE_THE_CUT: usize = 8;

/// Ends the first response after a few chunks of text, the way a dropped
/// connection would: no finish reason, no error.
struct CutOffOnce {
    inner: OpenAiProvider,
    requests: AtomicUsize,
}

fn has_text(message: &Message) -> bool {
    message
        .content
        .iter()
        .any(|content| matches!(content, MessageContent::Text(text) if !text.text.is_empty()))
}

#[async_trait]
impl Provider for CutOffOnce {
    fn get_name(&self) -> &str {
        self.inner.get_name()
    }

    async fn stream(
        &self,
        model_config: &ModelConfig,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let stream = self
            .inner
            .stream(model_config, system, messages, tools)
            .await?;
        if self.requests.fetch_add(1, Ordering::SeqCst) > 0 {
            return Ok(stream);
        }
        let mut text_chunks = 0;
        Ok(Box::pin(
            stream
                .take_while(move |item| {
                    let keep = text_chunks < TEXT_CHUNKS_BEFORE_THE_CUT;
                    if matches!(item, Ok((Some(message), _)) if has_text(message)) {
                        text_chunks += 1;
                    }
                    std::future::ready(keep)
                })
                .map(|item| item.map(|(message, _)| (message, None))),
        ))
    }

    fn retry_config(&self) -> RetryConfig {
        self.inner.retry_config()
    }

    fn stream_resume_policy(&self, model_config: &ModelConfig) -> StreamResumePolicy {
        self.inner.stream_resume_policy(model_config)
    }
}

#[tokio::test]
#[ignore = "calls the live Baseten API; needs BASETEN_API_KEY"]
async fn interrupted_answer_is_continued_by_model_apis() {
    let api_key = std::env::var("BASETEN_API_KEY").expect("BASETEN_API_KEY is not set");
    let model = std::env::var("BASETEN_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let api_client = ApiClient::new_with_tls(
        MODEL_APIS.to_string(),
        AuthMethod::BearerToken(api_key),
        None,
    )
    .unwrap();
    let provider = Arc::new(CutOffOnce {
        inner: OpenAiProviderBuilder::new(api_client).build(),
        requests: AtomicUsize::new(0),
    });
    let notices = Arc::new(Mutex::new(Vec::new()));
    let sink = notices.clone();
    let observer: RetryObserver = Arc::new(move |notice: &RetryNotice| {
        sink.lock().unwrap().push(notice.message());
    });

    let (text, error) = retry_notice::observe(observer, async {
        let mut stream = stream_resume::stream(
            provider.clone(),
            &ModelConfig::new(&model),
            "You are a helpful assistant.",
            &[Message::user().with_text("Count from one to forty in words, separated by commas.")],
            &[],
        )
        .await
        .expect("the first request should be accepted");
        let mut text = String::new();
        let mut error = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok((message, _)) => text.extend(message.map(|message| message.as_concat_text())),
                Err(failure) => error = Some(failure),
            }
        }
        (text, error)
    })
    .await;

    println!("notices: {:#?}", notices.lock().unwrap());
    println!("answer: {text}");
    assert_eq!(
        error, None,
        "the continuation was rejected or did not repeat the prefill"
    );
    assert_eq!(provider.requests.load(Ordering::SeqCst), 2);
    let text = text.to_lowercase();
    assert!(
        text.contains("forty") && text.matches("twenty,").count() <= 1,
        "the answer should run through to forty without starting over: {text}"
    );
}
