//! Tells whoever is driving a provider request that it is being retried.
//!
//! Retry loops run inside `Provider::stream`, with no channel back to the
//! caller. A caller wraps its future in [`observe`]; anything polled inside it
//! can then [`notify`], so a client hears "rate limited, retrying in 12s"
//! instead of silence. The observer is installed for the duration of each poll
//! rather than through a tokio task-local, which keeps this usable on wasm32.

use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use crate::errors::ProviderError;

const MAX_ERROR_SUMMARY_CHARS: usize = 240;

#[derive(Debug, Clone, PartialEq)]
pub enum RetryNotice {
    /// A request failed and is sent again after `delay`.
    Retrying {
        error: ProviderError,
        attempt: usize,
        max_attempts: usize,
        delay: Duration,
    },
    /// A response stopped part-way and is being continued.
    ResumingStream {
        error: ProviderError,
        attempt: usize,
        max_attempts: usize,
    },
    /// The model ran out of output tokens and the response is being continued.
    ContinuingAfterOutputLimit { attempt: usize, max_attempts: usize },
}

impl RetryNotice {
    pub fn message(&self) -> String {
        match self {
            RetryNotice::Retrying {
                error,
                attempt,
                max_attempts,
                delay,
            } => format!(
                "{}. Retrying in {} (attempt {attempt} of {max_attempts}).",
                summarize(error),
                format_delay(*delay)
            ),
            RetryNotice::ResumingStream {
                error,
                attempt,
                max_attempts,
            } => format!(
                "The response was interrupted ({}). Resuming where it left off \
                 (attempt {attempt} of {max_attempts}).",
                summarize(error)
            ),
            RetryNotice::ContinuingAfterOutputLimit {
                attempt,
                max_attempts,
            } => format!(
                "The model reached its output-token limit. Continuing the response \
                 (attempt {attempt} of {max_attempts})."
            ),
        }
    }
}

fn summarize(error: &ProviderError) -> String {
    let text = error.to_string();
    let text = text.trim().trim_end_matches('.');
    if text.chars().count() <= MAX_ERROR_SUMMARY_CHARS {
        return text.to_string();
    }
    let truncated: String = text.chars().take(MAX_ERROR_SUMMARY_CHARS).collect();
    format!("{}…", truncated.trim_end())
}

fn format_delay(delay: Duration) -> String {
    let seconds = delay.as_secs_f64().round() as u64;
    match seconds {
        0 => "under a second".to_string(),
        1..=59 => format!("{seconds}s"),
        _ if seconds.is_multiple_of(60) => format!("{}m", seconds / 60),
        _ => format!("{}m {}s", seconds / 60, seconds % 60),
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub type RetryObserver = Arc<dyn Fn(&RetryNotice) + Send + Sync>;

#[cfg(target_arch = "wasm32")]
pub type RetryObserver = Arc<dyn Fn(&RetryNotice)>;

thread_local! {
    static OBSERVER: RefCell<Option<RetryObserver>> = const { RefCell::new(None) };
}

/// Reports `notice` to the observer of the future being polled, if there is one.
pub fn notify(notice: RetryNotice) {
    let observer = OBSERVER
        .try_with(|slot| slot.borrow().clone())
        .ok()
        .flatten();
    if let Some(observer) = observer {
        observer(&notice);
    }
}

/// Runs `future` with `observer` receiving every [`notify`] made while it is polled.
pub fn observe<F: Future>(observer: RetryObserver, future: F) -> Observed<F> {
    Observed {
        observer,
        future: Box::pin(future),
    }
}

pub struct Observed<F> {
    observer: RetryObserver,
    future: Pin<Box<F>>,
}

impl<F: Future> Future for Observed<F> {
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let _installed = Installed::new(self.observer.clone());
        self.future.as_mut().poll(cx)
    }
}

struct Installed {
    previous: Option<RetryObserver>,
}

impl Installed {
    fn new(observer: RetryObserver) -> Self {
        let previous = OBSERVER
            .try_with(|slot| slot.replace(Some(observer)))
            .ok()
            .flatten();
        Self { previous }
    }
}

impl Drop for Installed {
    fn drop(&mut self) {
        let previous = self.previous.take();
        let _ = OBSERVER.try_with(|slot| *slot.borrow_mut() = previous);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn recorder() -> (RetryObserver, Arc<Mutex<Vec<String>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let observer: RetryObserver = Arc::new(move |notice: &RetryNotice| {
            sink.lock().unwrap().push(notice.message());
        });
        (observer, seen)
    }

    fn output_limit_notice() -> RetryNotice {
        RetryNotice::ContinuingAfterOutputLimit {
            attempt: 1,
            max_attempts: 3,
        }
    }

    #[tokio::test]
    async fn observer_hears_notices_raised_inside_its_future() {
        let (observer, seen) = recorder();

        observe(observer, async {
            tokio::task::yield_now().await;
            notify(output_limit_notice());
        })
        .await;

        assert_eq!(seen.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn notices_outside_an_observed_future_are_dropped() {
        let (observer, seen) = recorder();
        observe(observer, async {}).await;

        notify(output_limit_notice());

        assert!(seen.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn inner_observer_wins_and_outer_is_restored() {
        let (outer, outer_seen) = recorder();
        let (inner, inner_seen) = recorder();

        observe(outer, async {
            observe(inner, async { notify(output_limit_notice()) }).await;
            notify(output_limit_notice());
        })
        .await;

        assert_eq!(inner_seen.lock().unwrap().len(), 1);
        assert_eq!(outer_seen.lock().unwrap().len(), 1);
    }

    #[test]
    fn retry_message_names_the_error_the_wait_and_the_attempt() {
        let notice = RetryNotice::Retrying {
            error: ProviderError::RateLimitExceeded {
                details: "Too many requests.".to_string(),
                retry_delay: None,
            },
            attempt: 2,
            max_attempts: 6,
            delay: Duration::from_millis(12_400),
        };

        assert_eq!(
            notice.message(),
            "Rate limit exceeded: Too many requests. Retrying in 12s (attempt 2 of 6)."
        );
    }

    #[test]
    fn long_errors_are_truncated_on_a_character_boundary() {
        let notice = RetryNotice::ResumingStream {
            error: ProviderError::NetworkError("é".repeat(1_000)),
            attempt: 1,
            max_attempts: 3,
        };

        let message = notice.message();

        assert!(message.contains('…'));
        assert!(message.chars().count() < 400);
    }

    #[test]
    fn delays_are_rounded_to_readable_units() {
        assert_eq!(format_delay(Duration::from_millis(200)), "under a second");
        assert_eq!(format_delay(Duration::from_secs(45)), "45s");
        assert_eq!(format_delay(Duration::from_secs(120)), "2m");
        assert_eq!(format_delay(Duration::from_secs(125)), "2m 5s");
    }
}
