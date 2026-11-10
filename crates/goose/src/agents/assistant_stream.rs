use goose_providers::conversation::message::{Message, MessageContent};
use goose_providers::errors::ProviderError;
use rmcp::model::Role;

pub(crate) struct AssistantStreamState {
    prefill_attempts: usize,
    prefill_prefix: String,
    prefill_eligible: bool,
    prefill_message_id: Option<String>,
    prefill_message_id_to_resume: Option<String>,
    active_mergeable_assistant_id: Option<String>,
    active_mergeable_assistant_is_text: Option<bool>,
}

impl AssistantStreamState {
    pub(crate) fn new(supports_assistant_prefill: bool) -> Self {
        Self {
            prefill_attempts: 0,
            prefill_prefix: String::new(),
            prefill_eligible: supports_assistant_prefill,
            prefill_message_id: None,
            prefill_message_id_to_resume: None,
            active_mergeable_assistant_id: None,
            active_mergeable_assistant_is_text: None,
        }
    }

    pub(crate) fn can_continue_after_rate_limit(
        &self,
        error: &ProviderError,
        provider_manages_context: bool,
        max_retries: usize,
    ) -> bool {
        self.prefill_eligible
            && !provider_manages_context
            && !self.prefill_prefix.is_empty()
            && matches!(error, ProviderError::RateLimitExceeded { .. })
            && self.prefill_attempts < max_retries
    }

    pub(crate) fn begin_continuation(&mut self) -> (usize, String) {
        self.prefill_attempts += 1;
        self.prefill_message_id_to_resume = self.prefill_message_id.clone();
        self.active_mergeable_assistant_is_text = Some(true);
        (self.prefill_attempts, self.prefill_prefix.clone())
    }

    pub(crate) fn process_message(&mut self, mut message: Message) -> Message {
        if self.prefill_eligible {
            self.prefill_eligible = extend_assistant_prefill(&mut self.prefill_prefix, &message);
        }

        let is_text = is_text_only_assistant_chunk(&message);
        if !is_text {
            self.prefill_message_id_to_resume = None;
        }
        if message.id.is_some() {
            self.active_mergeable_assistant_id = None;
            self.active_mergeable_assistant_is_text = None;
        } else if is_mergeable_assistant_chunk(&message) {
            if self.active_mergeable_assistant_is_text != Some(is_text) {
                self.active_mergeable_assistant_id = None;
            }
            let id = self
                .active_mergeable_assistant_id
                .get_or_insert_with(|| format!("msg_{}", uuid::Uuid::new_v4()))
                .clone();
            self.active_mergeable_assistant_is_text = Some(is_text);
            message = message.with_id(id);
        } else {
            self.active_mergeable_assistant_id = None;
            self.active_mergeable_assistant_is_text = None;
            message = message.with_generated_id();
        }

        if is_mergeable_assistant_chunk(&message) && is_text {
            if let Some(id) = self.prefill_message_id_to_resume.as_ref() {
                message.id = Some(id.clone());
                self.active_mergeable_assistant_id = Some(id.clone());
                self.active_mergeable_assistant_is_text = Some(true);
            }
            self.prefill_message_id = message.id.clone();
        }
        message
    }
}

fn is_mergeable_assistant_chunk(message: &Message) -> bool {
    message.role == Role::Assistant
        && !message.content.is_empty()
        && message.content.iter().all(|content| {
            matches!(
                content,
                MessageContent::Text(_)
                    | MessageContent::Thinking(_)
                    | MessageContent::RedactedThinking(_)
            )
        })
}

fn is_text_only_assistant_chunk(message: &Message) -> bool {
    message.role == Role::Assistant
        && message
            .content
            .iter()
            .all(|content| matches!(content, MessageContent::Text(_)))
}

fn extend_assistant_prefill(prefix: &mut String, message: &Message) -> bool {
    if message.role != Role::Assistant || message.metadata.output_token_limit_reached {
        return false;
    }
    for content in &message.content {
        match content {
            MessageContent::Text(text) => prefix.push_str(&text.text),
            _ => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assistant_prefill_accepts_only_visible_complete_text() {
        let mut prefix = String::new();
        assert!(extend_assistant_prefill(
            &mut prefix,
            &Message::assistant().with_text("1 2 ")
        ));
        assert_eq!(prefix, "1 2 ");

        let thinking = Message::assistant().with_content(MessageContent::Thinking(
            goose_providers::conversation::message::ThinkingContentBlock {
                thinking: "reasoning".to_string(),
                signature: String::new(),
            },
        ));
        assert!(!extend_assistant_prefill(&mut prefix, &thinking));
        assert_eq!(prefix, "1 2 ");

        let mut limited = Message::assistant().with_text("truncated");
        limited.metadata.output_token_limit_reached = true;
        assert!(!extend_assistant_prefill(&mut prefix, &limited));
        assert_eq!(prefix, "1 2 ");
    }
}
