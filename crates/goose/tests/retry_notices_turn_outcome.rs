//! A turn's outcome is read from its last message, so a retry notification
//! must never follow the error that ended the turn.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use goose::agents::{
    AgentEvent, GoosePlatform, SessionConfig, StateMachineServices, StateMachineServicesConfig,
};
use goose::config::permission::PermissionManager;
use goose::config::GooseMode;
use goose::conversation::message::{Message, MessageContent};
use goose::providers::base::{MessageStream, Provider, ProviderDef, ProviderMetadata};
use goose::session::{SessionManager, SessionType};
use goose_providers::errors::ProviderError;
use goose_providers::model::ModelConfig;
use goose_providers::retry_notice::{self, RetryNotice};
use rmcp::model::{CallToolRequestParams, CallToolResult, ContentBlock, Tool};

/// Fails every request: a summarization call the way a provider does after
/// announcing a retry, the reply itself with an error that is not retried.
struct FailingProvider;

impl goose::providers::base::ProviderDescriptor for FailingProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata {
            name: "mock-failing".to_string(),
            display_name: "Mock Failing Provider".to_string(),
            description: "Fails every request".to_string(),
            default_model: "mock-model".to_string(),
            known_models: vec![],
            model_doc_link: "".to_string(),
            config_keys: vec![],
            setup_steps: vec![],
            setup: None,
            deprecated: None,
        }
    }
}

impl ProviderDef for FailingProvider {
    type Provider = Self;

    fn from_env(
        _extensions: Vec<goose::config::ExtensionConfig>,
        _tls_config: Option<goose::providers::api_client::TlsConfig>,
    ) -> futures::future::BoxFuture<'static, anyhow::Result<Self>> {
        Box::pin(async { Ok(Self) })
    }
}

#[async_trait]
impl Provider for FailingProvider {
    async fn stream(
        &self,
        _model_config: &ModelConfig,
        system_prompt: &str,
        _messages: &[Message],
        _tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        if !system_prompt.contains("summarize a tool call") {
            return Err(ProviderError::RequestFailed("bad request".to_string()));
        }
        let error = ProviderError::ServerError("overloaded".to_string());
        retry_notice::notify(RetryNotice::Retrying {
            error: error.clone(),
            attempt: 1,
            max_attempts: 3,
            delay: Duration::ZERO,
        });
        Err(error)
    }

    fn get_name(&self) -> &str {
        "mock-failing"
    }
}

fn is_notification(message: &Message) -> bool {
    matches!(
        message.content.as_slice(),
        [MessageContent::SystemNotification(_)]
    )
}

#[tokio::test]
async fn failed_turn_ends_on_its_error_not_on_a_retry_notification() -> Result<()> {
    let _guard = env_lock::lock_env([
        ("GOOSE_TOOL_PAIR_SUMMARIZATION", Some("true")),
        ("GOOSE_TOOL_CALL_CUTOFF", Some("2")),
    ]);

    let temp_dir = tempfile::tempdir()?;
    let session_manager = Arc::new(SessionManager::new(temp_dir.path().join("data")));
    let services = StateMachineServices::with_config(StateMachineServicesConfig::new(
        Arc::clone(&session_manager),
        Arc::new(PermissionManager::new(temp_dir.path().join("config"))),
        None,
        true,
        GoosePlatform::GooseCli,
    ));
    let session = session_manager
        .create_session(
            PathBuf::from("."),
            "retry-notices-turn-outcome".to_string(),
            SessionType::Hidden,
            GooseMode::Auto,
        )
        .await?;
    services
        .update_provider(
            Arc::new(FailingProvider),
            ModelConfig::new("mock-model"),
            &session.id,
        )
        .await?;

    // Enough finished tool calls for summarization to have work on every pass.
    let base_ts = chrono::Utc::now().timestamp() - 100;
    let mut opening = Message::user().with_text("help me read some files");
    opening.created = base_ts;
    session_manager.add_message(&session.id, &opening).await?;
    for i in 0..13 {
        let call_id = format!("precall_{i}");
        let mut request = Message::assistant()
            .with_tool_request(&call_id, Ok(CallToolRequestParams::new("read_file")))
            .with_generated_id();
        request.created = base_ts + i + 1;
        session_manager.add_message(&session.id, &request).await?;

        let mut response = Message::user()
            .with_tool_response(
                &call_id,
                Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                    "content of file {i}"
                ))])),
            )
            .with_generated_id();
        response.created = base_ts + i + 1;
        session_manager.add_message(&session.id, &response).await?;
    }

    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: Some(1),
    };
    let reply = services
        .reply(
            Message::user().with_text("summarize what you found"),
            session_config,
            None,
        )
        .await?;
    tokio::pin!(reply);
    let mut messages = Vec::new();
    while let Some(event) = reply.next().await {
        if let AgentEvent::Message(message) = event? {
            messages.push(message);
        }
    }

    assert!(
        messages.iter().any(is_notification),
        "summarization should have announced a retry before the reply was attempted"
    );
    let last = messages.last().expect("the turn emits messages");
    assert!(
        last.error_kind().is_some(),
        "the turn should end on its error, got {:?}",
        last.content
    );
    Ok(())
}
