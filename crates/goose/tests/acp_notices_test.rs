//! Provider failures reach an ACP client as notices when it can present them,
//! and as agent messages only when it cannot.

#![recursion_limit = "256"]
#[allow(dead_code)]
#[path = "acp_common_tests/mod.rs"]
mod common_tests;

use agent_client_protocol::schema::v1::{
    ContentBlock, InitializeRequest, NewSessionRequest, NoticeSeverity, PromptRequest,
    PromptResponse, SessionNotification, SessionUpdate, StopReason, TextContent,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{Agent, ByteStreams, Client, ConnectionTo};
use common_tests::fixtures::run_test;
use goose::acp::custom_notifications::{
    GooseSessionNotification, GooseSessionUpdate, StatusMessage,
};
use goose::acp::server::{
    acp_services_config, serve, AcpBuiltinSelection, AcpConnection, AcpConnectionOptions,
    AcpProviderFactory, LiveVoiceService,
};
use goose::agents::{GoosePlatform, StateMachineServices};
use goose::config::paths::Paths;
use goose::session::SessionManager;
use goose_test_support::TEST_MODEL;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const RATE_LIMIT_MAX_RETRIES: usize = 6;

/// Stands in for an OpenAI-compatible endpoint, answering each chat completion
/// with the next scripted response and repeating the last one after that.
struct Backend {
    server: MockServer,
}

impl Backend {
    async fn start(responses: Vec<ResponseTemplate>) -> Self {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_string(include_str!("acp_test_data/openai_models.json")),
            )
            .mount(&server)
            .await;
        let responses = Mutex::new(VecDeque::from(responses));
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(move |_request: &wiremock::Request| {
                let mut responses = responses.lock().unwrap();
                match responses.len() {
                    1 => responses.front().cloned().unwrap(),
                    _ => responses
                        .pop_front()
                        .expect("the test scripted no responses"),
                }
            })
            .mount(&server)
            .await;
        Self { server }
    }

    async fn chat_requests(&self) -> Vec<Value> {
        self.server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .filter(|request| request.method.as_str() == "POST")
            .map(|request| serde_json::from_slice(&request.body).unwrap())
            .collect()
    }
}

fn chunk(delta: Value, finish_reason: Option<&str>) -> String {
    format!(
        "data: {}\n\n",
        json!({"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1,
            "model": TEST_MODEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]})
    )
}

fn sse(body: String) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_raw(body, "text/event-stream")
}

fn answer(text: &str) -> ResponseTemplate {
    sse(format!(
        "{}{}data: [DONE]\n\n",
        chunk(json!({"content": text}), None),
        chunk(json!({}), Some("stop"))
    ))
}

fn answer_cut_off_by_a_stream_error(text: &str) -> ResponseTemplate {
    sse(format!(
        "{}data: {}\n\n",
        chunk(json!({"content": text}), None),
        json!({"error": {"message": "upstream exploded"}})
    ))
}

fn output_limit_before_any_content() -> ResponseTemplate {
    sse(format!(
        "{}data: [DONE]\n\n",
        chunk(json!({}), Some("length"))
    ))
}

fn rate_limited() -> ResponseTemplate {
    ResponseTemplate::new(429).set_body_json(json!({"error": {"message": "slow down"}}))
}

fn bad_request() -> ResponseTemplate {
    ResponseTemplate::new(400).set_body_json(json!({"error": {"message": "unsupported parameter"}}))
}

#[derive(Default)]
struct Received {
    updates: Vec<SessionUpdate>,
    status_messages: Vec<StatusMessage>,
}

impl Received {
    fn agent_text(&self) -> String {
        self.updates
            .iter()
            .filter_map(|update| match update {
                SessionUpdate::AgentMessageChunk(chunk) => match &chunk.content {
                    ContentBlock::Text(text) => Some(text.text.as_str()),
                    _ => None,
                },
                _ => None,
            })
            .collect()
    }

    fn notices(&self) -> Vec<(NoticeSeverity, String, Option<String>)> {
        self.updates
            .iter()
            .filter_map(|update| match update {
                SessionUpdate::Notice(notice) => Some((
                    notice.severity.clone(),
                    notice.title.clone(),
                    notice.description.clone(),
                )),
                _ => None,
            })
            .collect()
    }

    fn status_notices(&self) -> Vec<&str> {
        self.status_messages
            .iter()
            .filter_map(|status| match status {
                StatusMessage::Notice { message } => Some(message.as_str()),
                _ => None,
            })
            .collect()
    }
}

struct TestClient {
    cx: ConnectionTo<Agent>,
    received: Arc<Mutex<Received>>,
    _data_root: tempfile::TempDir,
}

impl TestClient {
    async fn connect(backend: &Backend, client_capabilities: Value) -> Self {
        // Retries are under test, not the time spent waiting between them.
        std::env::set_var("GOOSE_PROVIDER_SKIP_BACKOFF", "true");
        std::env::set_var("OPENAI_API_KEY", "test-key");

        let data_root = tempfile::tempdir().unwrap();
        let config = format!(
            "GOOSE_MODEL: {TEST_MODEL}\nGOOSE_PROVIDER: openai\nOPENAI_HOST: {}\n\
             GOOSE_TOOL_PAIR_SUMMARIZATION: false\n",
            backend.server.uri()
        );
        for config_dir in [data_root.path().to_path_buf(), Paths::config_dir()] {
            std::fs::create_dir_all(&config_dir).unwrap();
            std::fs::write(
                config_dir.join(goose::config::base::CONFIG_YAML_NAME),
                &config,
            )
            .unwrap();
        }
        std::fs::create_dir_all(Paths::in_state_dir("logs")).unwrap();

        let services = StateMachineServices::with_config(acp_services_config(
            Arc::new(SessionManager::new(data_root.path().to_path_buf())),
            data_root.path().to_path_buf(),
            true,
            GoosePlatform::GooseCli,
        ));
        let live_voice = Arc::new(LiveVoiceService::from_config(services.runtime.runs.clone()));
        let provider_factory: AcpProviderFactory = Arc::new(|provider_name| {
            Box::pin(async move { Err(anyhow::anyhow!("no provider named {provider_name}")) })
        });
        let connection = AcpConnection::new(AcpConnectionOptions {
            provider_factory,
            builtin_selection: AcpBuiltinSelection::default(),
            config_dir: data_root.path().to_path_buf(),
            services,
            additional_source_roots: Vec::new(),
            session_cwd: None,
            live_voice,
        })
        .await
        .unwrap();

        let (client_read, server_write) = tokio::io::duplex(64 * 1024);
        let (server_read, client_write) = tokio::io::duplex(64 * 1024);
        tokio::spawn(async move {
            let _ = serve(
                Arc::new(connection),
                server_read.compat(),
                server_write.compat_write(),
            )
            .await;
        });

        let received = Arc::new(Mutex::new(Received::default()));
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let initialize: InitializeRequest = serde_json::from_value(json!({
            "protocolVersion": ProtocolVersion::V1,
            "clientCapabilities": client_capabilities,
        }))
        .unwrap();
        tokio::spawn({
            let received = received.clone();
            async move {
                let _ = Client
                    .builder()
                    .on_receive_notification(
                        {
                            let received = received.clone();
                            async move |notification: SessionNotification, _cx| {
                                received.lock().unwrap().updates.push(notification.update);
                                Ok(())
                            }
                        },
                        agent_client_protocol::on_receive_notification!(),
                    )
                    .on_receive_notification(
                        {
                            let received = received.clone();
                            async move |notification: GooseSessionNotification, _cx| {
                                if let GooseSessionUpdate::StatusMessage(update) =
                                    notification.update
                                {
                                    received.lock().unwrap().status_messages.push(update.status);
                                }
                                Ok(())
                            }
                        },
                        agent_client_protocol::on_receive_notification!(),
                    )
                    .connect_with(
                        ByteStreams::new(client_write.compat_write(), client_read.compat()),
                        async move |cx: ConnectionTo<Agent>| {
                            cx.send_request(initialize).block_task().await.unwrap();
                            let _ = ready_tx.send(cx.clone());
                            std::future::pending::<Result<(), agent_client_protocol::Error>>().await
                        },
                    )
                    .await;
            }
        });

        Self {
            cx: ready_rx.await.unwrap(),
            received,
            _data_root: data_root,
        }
    }

    /// Sends one prompt and returns its outcome with everything the client was
    /// sent while the turn ran.
    async fn prompt(
        &self,
    ) -> (
        Result<PromptResponse, agent_client_protocol::Error>,
        Received,
    ) {
        let work_dir = tempfile::tempdir().unwrap();
        let session = self
            .cx
            .send_request(NewSessionRequest::new(work_dir.path()))
            .block_task()
            .await
            .unwrap();
        // Session setup sends its own updates; only the turn is under test.
        tokio::time::sleep(Duration::from_millis(200)).await;
        *self.received.lock().unwrap() = Received::default();

        let response = self
            .cx
            .send_request(PromptRequest::new(
                session.session_id,
                vec![ContentBlock::Text(TextContent::new("hello"))],
            ))
            .block_task()
            .await;
        tokio::time::sleep(Duration::from_millis(200)).await;
        (
            response,
            std::mem::take(&mut *self.received.lock().unwrap()),
        )
    }
}

fn session_notices() -> Value {
    json!({ "session": { "notices": {} } })
}

fn goose_custom_notifications() -> Value {
    json!({ "_meta": { "goose": { "customNotifications": true } } })
}

fn failure_reason(error: &agent_client_protocol::Error) -> Option<&str> {
    error.data.as_ref()?.get("reason")?.as_str()
}

const FAILURE_TITLE: &str =
    "Ran into this error: Request failed: Bad request (400): unsupported parameter.";
const FAILURE_GUIDANCE: &str =
    "Please retry if you think this is a transient or recoverable error.";

#[test]
fn rate_limit_is_announced_as_a_notice_and_the_turn_completes() {
    run_test(async {
        let backend = Backend::start(vec![rate_limited(), answer("Hello")]).await;
        let client = TestClient::connect(&backend, session_notices()).await;

        let (response, received) = client.prompt().await;

        assert_eq!(response.unwrap().stop_reason, StopReason::EndTurn);
        assert_eq!(received.agent_text(), "Hello");
        assert_eq!(
            received.notices(),
            vec![(
                NoticeSeverity::Info,
                format!(
                    "Rate limit exceeded: slow down. Retrying in under a second \
                     (attempt 1 of {RATE_LIMIT_MAX_RETRIES})."
                ),
                None
            )]
        );
    });
}

#[test]
fn interrupted_response_is_resumed_and_announced() {
    run_test(async {
        // Prefill is otherwise only on for providers known to support it.
        std::env::set_var("GOOSE_PROVIDER_STREAM_RESUME", "prefill");
        let backend = Backend::start(vec![
            answer_cut_off_by_a_stream_error("Hello, "),
            answer("world"),
        ])
        .await;
        let client = TestClient::connect(&backend, session_notices()).await;

        let (response, received) = client.prompt().await;
        std::env::remove_var("GOOSE_PROVIDER_STREAM_RESUME");

        assert_eq!(response.unwrap().stop_reason, StopReason::EndTurn);
        assert_eq!(received.agent_text(), "Hello, world");
        let notices = received.notices();
        assert_eq!(notices.len(), 1, "{notices:?}");
        assert!(
            notices[0].1.contains("Resuming where it left off"),
            "{notices:?}"
        );
        let requests = backend.chat_requests().await;
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[1]["messages"].as_array().unwrap().last().unwrap(),
            &json!({"role": "assistant", "content": "Hello, "})
        );
    });
}

#[test]
fn turn_failure_is_a_notice_rather_than_an_agent_message() {
    run_test(async {
        let backend = Backend::start(vec![bad_request()]).await;
        let client = TestClient::connect(&backend, session_notices()).await;

        let (response, received) = client.prompt().await;

        assert_eq!(failure_reason(&response.unwrap_err()), Some("turn_failed"));
        assert_eq!(received.agent_text(), "");
        assert_eq!(
            received.notices().last(),
            Some(&(
                NoticeSeverity::Error,
                FAILURE_TITLE.to_string(),
                Some(FAILURE_GUIDANCE.to_string())
            ))
        );
    });
}

#[test]
fn goose_client_is_shown_a_turn_failure_as_a_status_message() {
    run_test(async {
        let backend = Backend::start(vec![bad_request()]).await;
        let client = TestClient::connect(&backend, goose_custom_notifications()).await;

        let (response, received) = client.prompt().await;

        assert_eq!(failure_reason(&response.unwrap_err()), Some("turn_failed"));
        assert_eq!(received.agent_text(), "");
        assert!(received.notices().is_empty());
        assert_eq!(
            received.status_notices().last().copied(),
            Some(format!("{FAILURE_TITLE}\n\n{FAILURE_GUIDANCE}").as_str())
        );
    });
}

#[test]
fn client_without_a_notice_channel_gets_the_failure_as_an_agent_message() {
    run_test(async {
        let backend = Backend::start(vec![bad_request()]).await;
        let client = TestClient::connect(&backend, json!({})).await;

        let (response, received) = client.prompt().await;

        assert_eq!(failure_reason(&response.unwrap_err()), Some("turn_failed"));
        assert_eq!(
            received.agent_text(),
            format!("{FAILURE_TITLE}\n\n{FAILURE_GUIDANCE}")
        );
        assert!(received.notices().is_empty());
        assert!(received.status_notices().is_empty());
    });
}

#[test]
fn output_limit_without_content_is_a_warning_notice() {
    run_test(async {
        let backend = Backend::start(vec![output_limit_before_any_content()]).await;
        let client = TestClient::connect(&backend, session_notices()).await;

        let (response, received) = client.prompt().await;

        assert_eq!(response.unwrap().stop_reason, StopReason::MaxTokens);
        assert_eq!(received.agent_text(), "");
        assert_eq!(
            received.notices(),
            vec![(
                NoticeSeverity::Warning,
                "The model reached its output-token limit".to_string(),
                Some("The response stopped early and may be incomplete.".to_string())
            )]
        );
    });
}
