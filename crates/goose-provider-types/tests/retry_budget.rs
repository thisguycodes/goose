//! How many times a failed provider request is retried, and who hears about it.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use goose_provider_types::errors::ProviderError;
use goose_provider_types::retry::{
    max_retries_for, retry_operation, RetryConfig, DEFAULT_MAX_RETRIES,
    DEFAULT_RATE_LIMIT_MAX_RETRIES, RATE_LIMIT_MAX_RETRIES_ENV,
};
use goose_provider_types::retry_notice::{self, RetryNotice, RetryObserver};
use test_case::test_case;

fn rate_limited(details: &str, retry_delay: Option<Duration>) -> ProviderError {
    ProviderError::RateLimitExceeded {
        details: details.to_string(),
        retry_delay,
    }
}

fn without_backoff(max_retries: usize) -> RetryConfig {
    RetryConfig::new(max_retries, 0, 1.0, 0)
}

fn env(value: Option<&str>) -> env_lock::EnvGuard<'static> {
    env_lock::lock_env([(RATE_LIMIT_MAX_RETRIES_ENV, value)])
}

#[test]
fn rate_limit_on_the_default_config_gets_more_attempts() {
    let _env = env(None);

    assert_eq!(
        max_retries_for(&rate_limited("slow down", None), &RetryConfig::default()),
        DEFAULT_RATE_LIMIT_MAX_RETRIES
    );
}

#[test_case(ProviderError::ServerError("boom".into()) ; "server error")]
#[test_case(ProviderError::NetworkError("reset".into()) ; "network error")]
#[test_case(rate_limited("slow down", Some(Duration::from_secs(600))) ; "rate limit asking for a long wait")]
#[test_case(rate_limited("You exceeded your current quota", None) ; "exhausted quota")]
fn everything_else_keeps_the_usual_number_of_attempts(error: ProviderError) {
    let _env = env(None);

    assert_eq!(
        max_retries_for(&error, &RetryConfig::default()),
        DEFAULT_MAX_RETRIES
    );
}

#[test_case(0 ; "retries disabled")]
#[test_case(2 ; "fewer than the default")]
#[test_case(10 ; "more than the default")]
fn limit_a_provider_chose_is_kept(max_retries: usize) {
    let _env = env(None);

    assert_eq!(
        max_retries_for(
            &rate_limited("slow down", None),
            &without_backoff(max_retries)
        ),
        max_retries
    );
}

#[test_case("1", 10, 1 ; "lowers a provider limit")]
#[test_case("9", 3, 9 ; "raises the default")]
#[test_case("9", 0, 0 ; "does not enable retries a provider disabled")]
#[test_case("many", 3, DEFAULT_RATE_LIMIT_MAX_RETRIES ; "unparseable is ignored")]
fn configured_rate_limit_budget_applies_to_every_provider(
    configured: &str,
    provider_limit: usize,
    expected: usize,
) {
    let _env = env(Some(configured));

    assert_eq!(
        max_retries_for(
            &rate_limited("slow down", None),
            &without_backoff(provider_limit)
        ),
        expected
    );
}

#[test]
fn configured_budget_lowers_but_does_not_extend_a_long_wait() {
    let long_wait = rate_limited("slow down", Some(Duration::from_secs(600)));

    {
        let _env = env(Some("1"));
        assert_eq!(max_retries_for(&long_wait, &RetryConfig::default()), 1);
    }
    let _env = env(Some("9"));
    assert_eq!(
        max_retries_for(&long_wait, &RetryConfig::default()),
        DEFAULT_MAX_RETRIES
    );
}

#[tokio::test]
async fn budget_counts_every_retry_so_far_whatever_failed() {
    let _env = env(None);
    let calls = AtomicUsize::new(0);
    let config = without_backoff(DEFAULT_MAX_RETRIES);

    let result: Result<(), ProviderError> = retry_operation(&config, || async {
        match calls.fetch_add(1, Ordering::SeqCst) {
            0..=2 => Err(rate_limited("slow down", None)),
            _ => Err(ProviderError::ServerError("boom".into())),
        }
    })
    .await;

    // Three rate-limit retries have used up what a server error is allowed.
    assert_eq!(result, Err(ProviderError::ServerError("boom".into())));
    assert_eq!(calls.load(Ordering::SeqCst), 4);
}

#[tokio::test]
async fn every_retry_of_a_rate_limit_is_announced() {
    let _env = env(None);
    let calls = AtomicUsize::new(0);
    let notices = Arc::new(Mutex::new(Vec::new()));
    let sink = notices.clone();
    let observer: RetryObserver = Arc::new(move |notice: &RetryNotice| {
        sink.lock().unwrap().push(notice.clone());
    });
    let config = without_backoff(DEFAULT_MAX_RETRIES);

    let result: Result<(), ProviderError> = retry_notice::observe(
        observer,
        retry_operation(&config, || async {
            calls.fetch_add(1, Ordering::SeqCst);
            Err(rate_limited("slow down", None))
        }),
    )
    .await;

    assert_eq!(result, Err(rate_limited("slow down", None)));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1 + DEFAULT_RATE_LIMIT_MAX_RETRIES
    );
    let attempts: Vec<(usize, usize)> = notices
        .lock()
        .unwrap()
        .iter()
        .map(|notice| match notice {
            RetryNotice::Retrying {
                attempt,
                max_attempts,
                ..
            } => (*attempt, *max_attempts),
            other => panic!("unexpected notice {other:?}"),
        })
        .collect();
    assert_eq!(
        attempts,
        (1..=DEFAULT_RATE_LIMIT_MAX_RETRIES)
            .map(|attempt| (attempt, DEFAULT_RATE_LIMIT_MAX_RETRIES))
            .collect::<Vec<_>>()
    );
}
