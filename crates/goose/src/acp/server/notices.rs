//! Shows errors and status to the user without making them conversation content.
//!
//! An ACP session notice is advisory and live-only, and the client says at
//! initialize whether it can present one. goose's own clients take a status
//! message instead. A client that offers neither still gets the text as an agent
//! message, which is the fallback the protocol gives agents.

use std::borrow::Cow;

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
use crate::conversation::Conversation;
use crate::session::Session;

const OUTPUT_LIMIT_TITLE: &str = "The model reached its output-token limit";
const OUTPUT_LIMIT_DESCRIPTION: &str = "The response stopped early and may be incomplete.";

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

/// A notice needs a title that stands alone; goose's error text leads with one
/// and follows with guidance.
fn notice(severity: NoticeSeverity, text: &str) -> Notice {
    let text = text.trim();
    match text.split_once("\n\n") {
        Some((title, description)) if !title.trim().is_empty() => {
            Notice::new(severity, title.trim()).description(description.trim().to_string())
        }
        _ => Notice::new(severity, text),
    }
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

/// Returns whether the client was told, so the caller knows not to fall back to
/// an agent message.
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
        NoticeChannel::Unsupported => return Ok(false),
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

/// Returns whether a notice replaced the text that otherwise stands in for a
/// response cut off before it produced any content.
pub(super) fn send_output_limit(
    cx: &ConnectionTo<Client>,
    channel: NoticeChannel,
    session_id: &SessionId,
    message: &Message,
) -> Result<bool, agent_client_protocol::Error> {
    if channel != NoticeChannel::Session || !is_output_limit_marker(message) {
        return Ok(false);
    }
    send_session_notice(
        cx,
        session_id,
        Notice::new(NoticeSeverity::Warning, OUTPUT_LIMIT_TITLE)
            .description(OUTPUT_LIMIT_DESCRIPTION.to_string()),
    )?;
    Ok(true)
}

/// Notices are live events, not history: a client that was shown an error as a
/// notice does not get it back as an agent message when the session is loaded.
fn history_for_replay(conversation: &Conversation, channel: NoticeChannel) -> Option<Conversation> {
    let was_notice = |message: &Message| {
        message.error().is_some()
            || (channel == NoticeChannel::Session && is_output_limit_marker(message))
    };
    if channel == NoticeChannel::Unsupported || !conversation.messages().iter().any(was_notice) {
        return None;
    }

    let mut conversation = conversation.clone();
    for message in conversation.messages_mut() {
        message
            .content
            .retain(|content| !matches!(content, MessageContent::Error(_)));
        if channel == NoticeChannel::Session && is_output_limit_marker(message) {
            message.metadata.output_token_limit_reached = false;
        }
    }
    Some(conversation)
}

pub(super) fn session_for_replay(session: &Session, channel: NoticeChannel) -> Cow<'_, Session> {
    let history = session
        .conversation
        .as_ref()
        .and_then(|conversation| history_for_replay(conversation, channel));
    match history {
        Some(conversation) => Cow::Owned(Session {
            conversation: Some(conversation),
            ..session.clone()
        }),
        None => Cow::Borrowed(session),
    }
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

    fn failed_turn() -> Conversation {
        Conversation::new_unvalidated(vec![
            Message::user().with_text("hello"),
            Message::assistant().with_error(MessageErrorKind::Other, "provider failed"),
            output_limit_marker(),
        ])
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
    fn replay_keeps_errors_for_a_client_that_was_sent_them_as_messages() {
        assert!(history_for_replay(&failed_turn(), NoticeChannel::Unsupported).is_none());
    }

    #[test]
    fn replay_is_untouched_when_nothing_was_shown_as_a_notice() {
        let conversation = Conversation::new_unvalidated(vec![Message::user().with_text("hello")]);

        assert!(history_for_replay(&conversation, NoticeChannel::Session).is_none());
    }

    #[test]
    fn replay_drops_errors_that_were_shown_as_notices() {
        let goose = history_for_replay(&failed_turn(), NoticeChannel::GooseStatus).unwrap();
        assert!(goose.messages()[1].content.is_empty());
        assert!(
            goose.messages()[2].metadata.output_token_limit_reached,
            "goose clients read the limit from the replayed message"
        );

        let standard = history_for_replay(&failed_turn(), NoticeChannel::Session).unwrap();
        assert!(standard.messages()[1].content.is_empty());
        assert!(!standard.messages()[2].metadata.output_token_limit_reached);
    }

    #[test]
    fn only_an_empty_assistant_message_stands_for_the_output_limit() {
        assert!(is_output_limit_marker(&output_limit_marker()));

        let mut with_content = output_limit_marker();
        with_content.content.push(MessageContent::text("partial"));
        assert!(!is_output_limit_marker(&with_content));

        assert!(!is_output_limit_marker(&Message::assistant()));
    }
}
