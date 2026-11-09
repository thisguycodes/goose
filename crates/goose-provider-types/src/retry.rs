use crate::base::Provider;
use crate::errors::ProviderError;
use crate::maybe_send::MaybeSend;
use async_trait::async_trait;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

// wasm32 hosts run without a tokio runtime, so back off on the host's timer.
#[cfg(target_arch = "wasm32")]
use gloo_timers::future::sleep;
#[cfg(not(target_arch = "wasm32"))]
use tokio::time::sleep;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitWaitKind {
    Retry { attempt: usize, max_retries: usize },
    Proactive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateLimitWait {
    pub kind: RateLimitWaitKind,
    pub delay: Duration,
}

#[cfg(not(target_arch = "wasm32"))]
pub type RateLimitObserverRef = Arc<dyn Fn(RateLimitWait) + Send + Sync>;
#[cfg(target_arch = "wasm32")]
pub type RateLimitObserverRef = Arc<dyn Fn(RateLimitWait)>;

tokio::task_local! {
    static RATE_LIMIT_OBSERVER: RateLimitObserverRef;
}

pub async fn with_rate_limit_observer<F: Future>(
    observer: RateLimitObserverRef,
    future: F,
) -> F::Output {
    RATE_LIMIT_OBSERVER.scope(observer, future).await
}

pub async fn sleep_for_rate_limit(kind: RateLimitWaitKind, delay: Duration) -> bool {
    if skip_backoff() || delay.is_zero() {
        return false;
    }
    let mut remaining = delay;
    while !remaining.is_zero() {
        let _ = RATE_LIMIT_OBSERVER.try_with(|observer| {
            observer(RateLimitWait {
                kind,
                delay: remaining,
            })
        });
        let interval = remaining.min(Duration::from_secs(1));
        sleep(interval).await;
        remaining = remaining.saturating_sub(interval);
    }
    true
}

fn skip_backoff() -> bool {
    std::env::var("GOOSE_PROVIDER_SKIP_BACKOFF")
        .unwrap_or_default()
        .parse::<bool>()
        .unwrap_or(false)
}

pub const DEFAULT_MAX_RETRIES: usize = 3;
pub const DEFAULT_INITIAL_RETRY_INTERVAL_MS: u64 = 1000;
pub const DEFAULT_BACKOFF_MULTIPLIER: f64 = 2.0;
pub const DEFAULT_MAX_RETRY_INTERVAL_MS: u64 = 30_000;

#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: usize,
    /// Initial interval between retries in milliseconds
    pub initial_interval_ms: u64,
    /// Multiplier for backoff (exponential)
    pub backoff_multiplier: f64,
    /// Maximum interval between retries in milliseconds
    pub max_interval_ms: u64,
    /// When true, only retry on transient errors (ServerError, NetworkError,
    /// RateLimitExceeded). RequestFailed (4xx client errors) will not be retried.
    pub transient_only: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: DEFAULT_MAX_RETRIES,
            initial_interval_ms: DEFAULT_INITIAL_RETRY_INTERVAL_MS,
            backoff_multiplier: DEFAULT_BACKOFF_MULTIPLIER,
            max_interval_ms: DEFAULT_MAX_RETRY_INTERVAL_MS,
            transient_only: false,
        }
    }
}

impl RetryConfig {
    pub fn new(
        max_retries: usize,
        initial_interval_ms: u64,
        backoff_multiplier: f64,
        max_interval_ms: u64,
    ) -> Self {
        Self {
            max_retries,
            initial_interval_ms,
            backoff_multiplier,
            max_interval_ms,
            transient_only: false,
        }
    }

    pub fn transient_only(mut self) -> Self {
        self.transient_only = true;
        self
    }

    pub fn max_retries(&self) -> usize {
        self.max_retries
    }

    pub fn delay_for_attempt(&self, attempt: usize) -> Duration {
        if attempt == 0 {
            return Duration::from_millis(0);
        }

        let exponent = (attempt - 1) as u32;
        let base_delay_ms = (self.initial_interval_ms as f64
            * self.backoff_multiplier.powi(exponent as i32)) as u64;

        let capped_delay_ms = std::cmp::min(base_delay_ms, self.max_interval_ms);

        let jitter_factor_to_avoid_thundering_herd = 0.8 + (rand::random::<f64>() * 0.4);
        let jitter_delay_ms =
            (capped_delay_ms as f64 * jitter_factor_to_avoid_thundering_herd) as u64;

        Duration::from_millis(jitter_delay_ms)
    }
}

/// Substrings marking a `RequestFailed` (4xx) as deterministically permanent:
/// Anthropic rejects signed `thinking`/`redacted_thinking` blocks as immutable
/// once a thinking model's config changes mid-conversation, and the identical
/// payload is rebuilt on every retry — so retrying can never succeed.
const PERMANENT_REQUEST_FAILURE_MARKERS: &[&str] = &[
    "blocks in the latest assistant message cannot be modified",
    "must remain as they were in the original response",
    "Reasoning is mandatory for this endpoint",
];

fn is_permanent_request_failure(message: &str) -> bool {
    PERMANENT_REQUEST_FAILURE_MARKERS
        .iter()
        .any(|marker| message.contains(marker))
        || crate::formats::anthropic::is_thinking_signature_error(message)
        || crate::formats::openai_responses::is_encrypted_reasoning_error(message)
}

pub fn should_retry(error: &ProviderError, config: &RetryConfig) -> bool {
    match error {
        ProviderError::RateLimitExceeded { .. }
        | ProviderError::RateLimitExceededAfterNonText { .. }
        | ProviderError::ServerError(_)
        | ProviderError::NetworkError(_) => true,
        ProviderError::RequestFailed(message) if is_permanent_request_failure(message) => false,
        ProviderError::RequestFailed(_) => !config.transient_only,
        _ => false,
    }
}

pub async fn sleep_before_retry(error: &ProviderError, config: &RetryConfig, attempt: usize) {
    let delay = match error {
        ProviderError::RateLimitExceeded {
            retry_delay: Some(delay),
            ..
        }
        | ProviderError::RateLimitExceededAfterNonText {
            retry_delay: Some(delay),
            ..
        } => *delay,
        _ => config.delay_for_attempt(attempt),
    };

    if error.is_rate_limit_exceeded() {
        sleep_for_rate_limit(
            RateLimitWaitKind::Retry {
                attempt,
                max_retries: config.max_retries,
            },
            delay,
        )
        .await;
        return;
    }

    if skip_backoff() {
        tracing::info!("Skipping backoff due to GOOSE_PROVIDER_SKIP_BACKOFF");
    } else {
        tracing::info!("Backing off for {:?} before retry", delay);
        sleep(delay).await;
    }
}

pub async fn retry_operation<F, Fut, T>(
    config: &RetryConfig,
    operation: F,
) -> Result<T, ProviderError>
where
    F: Fn() -> Fut + MaybeSend,
    Fut: Future<Output = Result<T, ProviderError>> + MaybeSend,
    T: MaybeSend,
{
    let mut attempts = 0;

    loop {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(error) => {
                if should_retry(&error, config) && attempts < config.max_retries {
                    attempts += 1;
                    tracing::warn!(
                        "Request failed, retrying ({}/{}): {:?}",
                        attempts,
                        config.max_retries,
                        error
                    );

                    sleep_before_retry(&error, config, attempts).await;
                    continue;
                }
                return Err(error);
            }
        }
    }
}

/// Trait for retry functionality to keep Provider dyn-compatible.
///
/// All `Provider` implementors get this via the blanket impl below.
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait ProviderRetry {
    fn retry_config(&self) -> RetryConfig {
        RetryConfig::default()
    }

    async fn with_retry<F, Fut, T>(&self, operation: F) -> Result<T, ProviderError>
    where
        F: Fn() -> Fut + MaybeSend,
        Fut: Future<Output = Result<T, ProviderError>> + MaybeSend,
        T: MaybeSend,
    {
        self.with_retry_config(operation, self.retry_config()).await
    }

    async fn with_retry_config<F, Fut, T>(
        &self,
        operation: F,
        config: RetryConfig,
    ) -> Result<T, ProviderError>
    where
        F: Fn() -> Fut + MaybeSend,
        Fut: Future<Output = Result<T, ProviderError>> + MaybeSend,
        T: MaybeSend;
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl<P: Provider> ProviderRetry for P {
    fn retry_config(&self) -> RetryConfig {
        Provider::retry_config(self)
    }

    async fn with_retry_config<F, Fut, T>(
        &self,
        operation: F,
        config: RetryConfig,
    ) -> Result<T, ProviderError>
    where
        F: Fn() -> Fut + MaybeSend,
        Fut: Future<Output = Result<T, ProviderError>> + MaybeSend,
        T: MaybeSend,
    {
        let mut attempts = 0;
        let mut auth_retried = false;

        loop {
            return match operation().await {
                Ok(result) => Ok(result),
                Err(error) => {
                    // Auth retry is separate from transient-error retries: we get
                    // at most 1 credential refresh, independent of max_retries.
                    if matches!(error, ProviderError::Authentication(_)) && !auth_retried {
                        auth_retried = true;
                        match self.refresh_credentials().await {
                            Ok(()) => {
                                tracing::warn!(
                                    "Credentials refreshed after auth error, retrying: {:?}",
                                    error
                                );
                                continue;
                            }
                            Err(refresh_err) => {
                                tracing::warn!(
                                    "Credential refresh failed, returning original auth error: {:?}",
                                    refresh_err
                                );
                            }
                        }
                    }

                    if should_retry(&error, &config) && attempts < config.max_retries {
                        attempts += 1;
                        tracing::warn!(
                            "Request failed, retrying ({}/{}): {:?}",
                            attempts,
                            config.max_retries,
                            error
                        );

                        sleep_before_retry(&error, &config, attempts).await;
                        continue;
                    }

                    Err(error)
                }
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_retries_request_failed() {
        let config = RetryConfig::default();
        let error = ProviderError::RequestFailed("Bad request (400): model not found".into());
        assert!(should_retry(&error, &config));
    }

    #[test]
    fn never_retries_permanent_thinking_block_400() {
        let config = RetryConfig::default();
        let error = ProviderError::RequestFailed(
            "Bad request (400): {\"message\":\"messages.3.content.1: `thinking` or \
             `redacted_thinking` blocks in the latest assistant message cannot be \
             modified. These blocks must remain as they were in the original \
             response.\"}"
                .into(),
        );
        assert!(!should_retry(&error, &config));
    }

    #[test]
    fn permanent_request_failure_marker_detection() {
        assert!(is_permanent_request_failure(
            "messages.3.content.1: `thinking` blocks in the latest assistant message \
             cannot be modified"
        ));
        assert!(is_permanent_request_failure(
            "These blocks must remain as they were in the original response."
        ));
        assert!(is_permanent_request_failure(
            "messages.1.content.0: Invalid `signature` in `thinking` block."
        ));
        assert!(!is_permanent_request_failure(
            "Bad request (400): model not found"
        ));
    }

    #[test]
    fn transient_only_skips_request_failed() {
        let config = RetryConfig::default().transient_only();
        let error = ProviderError::RequestFailed("Bad request (400): model not found".into());
        assert!(!should_retry(&error, &config));
    }

    #[test]
    fn transient_only_still_retries_server_error() {
        let config = RetryConfig::default().transient_only();
        assert!(should_retry(
            &ProviderError::ServerError("500 internal".into()),
            &config
        ));
    }

    #[test]
    fn transient_only_still_retries_network_error() {
        let config = RetryConfig::default().transient_only();
        assert!(should_retry(
            &ProviderError::NetworkError("connection refused".into()),
            &config
        ));
    }

    #[test]
    fn transient_only_still_retries_rate_limit() {
        let config = RetryConfig::default().transient_only();
        assert!(should_retry(
            &ProviderError::RateLimitExceeded {
                details: "too many requests".into(),
                retry_delay: None,
            },
            &config
        ));
    }

    #[tokio::test]
    async fn rate_limit_retries_notify_scoped_observer() {
        let waits = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed_waits = waits.clone();
        let observer: RateLimitObserverRef = Arc::new(move |wait| {
            observed_waits.lock().unwrap().push(wait);
        });
        let config = RetryConfig {
            max_retries: 1,
            initial_interval_ms: 0,
            max_interval_ms: 0,
            ..RetryConfig::default()
        };
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let result = with_rate_limit_observer(observer, {
            let attempts = attempts.clone();
            retry_operation(&config, move || {
                let attempts = attempts.clone();
                async move {
                    if attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                        Err(ProviderError::RateLimitExceeded {
                            details: "slow down".into(),
                            retry_delay: Some(std::time::Duration::from_millis(1)),
                        })
                    } else {
                        Ok("ok")
                    }
                }
            })
        })
        .await;

        assert_eq!(result.unwrap(), "ok");
        let waits = waits.lock().unwrap();
        assert_eq!(waits.len(), 1);
        assert_eq!(
            waits[0].kind,
            RateLimitWaitKind::Retry {
                attempt: 1,
                max_retries: 1
            }
        );
    }

    #[tokio::test]
    async fn rate_limit_wait_reports_countdown() {
        let delays = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed_delays = delays.clone();
        let observer: RateLimitObserverRef = Arc::new(move |wait| {
            observed_delays.lock().unwrap().push(wait.delay);
        });

        with_rate_limit_observer(
            observer,
            sleep_for_rate_limit(
                RateLimitWaitKind::Proactive,
                std::time::Duration::from_millis(1_100),
            ),
        )
        .await;

        assert_eq!(
            *delays.lock().unwrap(),
            [
                std::time::Duration::from_millis(1_100),
                std::time::Duration::from_millis(100)
            ]
        );
    }

    #[test]
    fn never_retries_auth_errors() {
        let config = RetryConfig::default();
        assert!(!should_retry(
            &ProviderError::Authentication("invalid key".into()),
            &config
        ));
    }
}
