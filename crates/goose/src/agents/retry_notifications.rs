//! Surfaces provider retries to the client as status notifications.

use std::future::Future;
use std::sync::Arc;

use goose_providers::retry_notice::{self, Observed, RetryNotice};
use tokio::sync::mpsc;

use crate::agents::AgentEvent;
use crate::conversation::message::{Message, SystemNotificationType};

/// The notification is emitted rather than appended to the conversation: it is
/// about the turn in progress and means nothing once the turn is over.
fn notification(notice: &RetryNotice) -> Message {
    Message::assistant()
        .with_system_notification(SystemNotificationType::InlineMessage, notice.message())
        .with_generated_id_if_missing()
}

/// Runs `turn`, sending a notification on `events` whenever a provider request
/// made inside it is retried, resumed or continued.
pub(super) fn forward_to<F: Future>(
    events: mpsc::UnboundedSender<AgentEvent>,
    turn: F,
) -> Observed<F> {
    retry_notice::observe(
        Arc::new(move |notice: &RetryNotice| {
            let _ = events.send(AgentEvent::Message(notification(notice)));
        }),
        turn,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::message::MessageContent;

    #[tokio::test]
    async fn retries_reach_the_client_as_user_only_notifications() {
        let (events, mut received) = mpsc::unbounded_channel();

        forward_to(events, async {
            retry_notice::notify(RetryNotice::ContinuingAfterOutputLimit {
                attempt: 1,
                max_attempts: 3,
            });
        })
        .await;

        let Some(AgentEvent::Message(message)) = received.recv().await else {
            panic!("expected a notification message");
        };
        assert!(message.is_user_visible());
        assert!(!message.is_agent_visible());
        assert!(message.id.is_some());
        let [MessageContent::SystemNotification(notification)] = message.content.as_slice() else {
            panic!("expected a system notification, got {:?}", message.content);
        };
        assert_eq!(
            notification.notification_type,
            SystemNotificationType::InlineMessage
        );
        assert!(notification.msg.contains("output-token limit"));
        assert!(received.recv().await.is_none());
    }
}
