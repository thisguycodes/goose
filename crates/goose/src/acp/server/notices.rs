//! Shows errors and status to the user without making them conversation content.
//!
//! An ACP session notice is advisory and live-only, and the client says at
//! initialize whether it can present one. A client that does not still gets an
//! error as an agent message, which is the fallback the protocol gives agents
//! and what goose's own clients are built around; `GOOSE_ACP_ERRORS_AS_MESSAGES`
//! turns that fallback off.

use agent_client_protocol::schema::v1::{
    InitializeRequest, Notice, NoticeSeverity, SessionId, SessionNotification, SessionUpdate,
};
use agent_client_protocol::{Client, ConnectionTo};

use super::AcpConnection;
use crate::acp::custom_notifications::{
    GooseSessionNotification, GooseSessionUpdate, StatusMessage, StatusMessageUpdate,
};
use crate::conversation::message::{
    ErrorContent, Message, MessageContent, SystemNotificationContent, SystemNotificationType,
};

pub(super) const ERRORS_AS_MESSAGES_ENV: &str = "GOOSE_ACP_ERRORS_AS_MESSAGES";

const OUTPUT_LIMIT_TITLE: &str = "The model reached its output-token limit";
const OUTPUT_LIMIT_DESCRIPTION: &str = "The response stopped early and may be incomplete.";

/// A notice title has to stand alone; a provider error can carry a whole
/// response body.
const MAX_TITLE_CHARS: usize = 240;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NoticeChannel {
    /// ACP `notice` session updates.
    Session,
    /// goose's `status_message` custom notification.
    GooseStatus,
    Unsupported,
}

pub(super) fn client_supports_session_notices(args: &InitializeRequest) -> bool {
    args.client_capabilities
        .session
        .as_ref()
        .and_then(|session| session.notices.as_ref())
        .is_some()
}

impl AcpConnection {
    pub(super) fn notice_channel(&self) -> NoticeChannel {
        if self
            .client_supports_session_notices
            .get()
            .copied()
            .unwrap_or(false)
        {
            NoticeChannel::Session
        } else if self.supports_goose_custom_notifications() {
            NoticeChannel::GooseStatus
        } else {
            NoticeChannel::Unsupported
        }
    }
}

fn errors_as_messages() -> bool {
    !std::env::var(ERRORS_AS_MESSAGES_ENV).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "false" | "0" | "off"
        )
    })
}

/// goose's error text leads with what happened and follows with guidance, which
/// is the split a notice wants.
fn notice(severity: NoticeSeverity, text: &str) -> Notice {
    let text = text.trim();
    let (title, description) = match text.split_once("\n\n") {
        Some((title, description)) => (title.trim(), Some(description.trim())),
        None => (text, None),
    };
    if title.chars().count() <= MAX_TITLE_CHARS {
        return Notice::new(severity, title).description(description.map(str::to_string));
    }
    let shortened: String = title.chars().take(MAX_TITLE_CHARS).collect();
    Notice::new(severity, format!("{}…", shortened.trim_end())).description(text.to_string())
}

fn send_session_notice(
    cx: &ConnectionTo<Client>,
    session_id: &SessionId,
    notice: Notice,
) -> Result<(), agent_client_protocol::Error> {
    cx.send_notification(SessionNotification::new(
        session_id.clone(),
        SessionUpdate::Notice(notice),
    ))
}

/// Returns whether the error has been dealt with, so the caller knows not to
/// send it as an agent message.
pub(super) fn send_error(
    cx: &ConnectionTo<Client>,
    channel: NoticeChannel,
    session_id: &SessionId,
    error: &ErrorContent,
) -> Result<bool, agent_client_protocol::Error> {
    match channel {
        NoticeChannel::Session => {
            send_session_notice(
                cx,
                session_id,
                notice(NoticeSeverity::Error, &error.message),
            )?;
        }
        _ if errors_as_messages() => return Ok(false),
        NoticeChannel::GooseStatus => {
            cx.send_notification(GooseSessionNotification {
                session_id: session_id.0.to_string(),
                update: GooseSessionUpdate::StatusMessage(StatusMessageUpdate {
                    status: StatusMessage::Notice {
                        message: error.message.clone(),
                    },
                }),
            })?;
        }
        // The prompt's own error response still reports a failed turn.
        NoticeChannel::Unsupported => {}
    }
    Ok(true)
}

/// Returns whether the notification went out as a session notice. Progress
/// text is a spinner label rather than something to notify the user about, so
/// it is left to the status-message path.
pub(super) fn send_system_notification(
    cx: &ConnectionTo<Client>,
    channel: NoticeChannel,
    session_id: &SessionId,
    notification: &SystemNotificationContent,
) -> Result<bool, agent_client_protocol::Error> {
    if channel != NoticeChannel::Session
        || notification.notification_type != SystemNotificationType::InlineMessage
    {
        return Ok(false);
    }
    send_session_notice(
        cx,
        session_id,
        notice(NoticeSeverity::Info, &notification.msg),
    )?;
    Ok(true)
}

fn is_output_limit_marker(message: &Message) -> bool {
    message.role == rmcp::model::Role::Assistant
        && message.content.is_empty()
        && message.metadata.output_token_limit_reached
}

/// Returns whether the text that otherwise stands in for a response cut off
/// before it produced any content should be left out. goose's own clients read
/// the limit from that message's metadata, so they always get it.
pub(super) fn send_output_limit(
    cx: &ConnectionTo<Client>,
    channel: NoticeChannel,
    session_id: &SessionId,
    message: &Message,
) -> Result<bool, agent_client_protocol::Error> {
    if !is_output_limit_marker(message) {
        return Ok(false);
    }
    match channel {
        NoticeChannel::Session => {
            send_session_notice(
                cx,
                session_id,
                Notice::new(NoticeSeverity::Warning, OUTPUT_LIMIT_TITLE)
                    .description(OUTPUT_LIMIT_DESCRIPTION.to_string()),
            )?;
            Ok(true)
        }
        // The prompt's stop reason still reports the limit.
        NoticeChannel::Unsupported => Ok(!errors_as_messages()),
        NoticeChannel::GooseStatus => Ok(false),
    }
}

/// Whether `message` only reports status. Such a message arrives at any point
/// in a turn and says nothing about how the turn went.
pub(super) fn is_status_only(message: &Message) -> bool {
    !message.content.is_empty()
        && message.content.iter().all(|content| {
            matches!(
                content,
                MessageContent::SystemNotification(notification)
                    if notification.notification_type != SystemNotificationType::CreditsExhausted
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::message::MessageErrorKind;
    use agent_client_protocol::schema::ProtocolVersion;

    fn initialize(client_capabilities: serde_json::Value) -> InitializeRequest {
        serde_json::from_value(serde_json::json!({
            "protocolVersion": ProtocolVersion::V1,
            "clientCapabilities": client_capabilities,
        }))
        .unwrap()
    }

    fn output_limit_marker() -> Message {
        let mut marker = Message::assistant();
        marker.metadata.output_token_limit_reached = true;
        marker
    }

    #[test]
    fn client_advertises_notices_under_session_capabilities() {
        assert!(client_supports_session_notices(&initialize(
            serde_json::json!({ "session": { "notices": {} } })
        )));
        assert!(!client_supports_session_notices(&initialize(
            serde_json::json!({ "session": {} })
        )));
        assert!(!client_supports_session_notices(&initialize(
            serde_json::json!({})
        )));
    }

    #[test]
    fn error_text_splits_into_a_title_and_guidance() {
        let notice = notice(
            NoticeSeverity::Error,
            "Ran into this error: Rate limit exceeded: slow down.\n\n\
             Please retry if you think this is a transient or recoverable error.",
        );

        assert_eq!(
            notice.title,
            "Ran into this error: Rate limit exceeded: slow down."
        );
        assert_eq!(
            notice.description.as_deref(),
            Some("Please retry if you think this is a transient or recoverable error.")
        );
    }

    #[test]
    fn single_paragraph_text_is_only_a_title() {
        let notice = notice(NoticeSeverity::Info, "Compaction complete\n");

        assert_eq!(notice.title, "Compaction complete");
        assert_eq!(notice.description, None);
    }

    #[test]
    fn overlong_title_is_shortened_and_kept_whole_in_the_description() {
        let text = format!("Server error: {}\n\nPlease retry.", "é".repeat(1_000));

        let notice = notice(NoticeSeverity::Error, &text);

        assert_eq!(notice.title.chars().count(), MAX_TITLE_CHARS + 1);
        assert!(notice.title.ends_with('…'));
        assert_eq!(notice.description.as_deref(), Some(text.as_str()));
    }

    #[test]
    fn only_an_empty_assistant_message_stands_for_the_output_limit() {
        assert!(is_output_limit_marker(&output_limit_marker()));

        let mut with_content = output_limit_marker();
        with_content.content.push(MessageContent::text("partial"));
        assert!(!is_output_limit_marker(&with_content));

        assert!(!is_output_limit_marker(&Message::assistant()));
    }

    #[test]
    fn status_notifications_are_status_only_and_turn_outcomes_are_not() {
        let status = |notification_type| {
            Message::assistant().with_system_notification(notification_type, "status")
        };

        assert!(is_status_only(&status(
            SystemNotificationType::InlineMessage
        )));
        assert!(is_status_only(&status(
            SystemNotificationType::ProgressMessage
        )));
        assert!(!is_status_only(&status(
            SystemNotificationType::CreditsExhausted
        )));
        assert!(!is_status_only(
            &Message::assistant().with_error(MessageErrorKind::Other, "provider failed")
        ));
        assert!(!is_status_only(&Message::assistant().with_text("answer")));
        assert!(!is_status_only(&output_limit_marker()));
    }
}
