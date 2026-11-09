use crate::http_status::{extract_retry_after, handle_status, RateLimitHeaders};
use crate::retry::{sleep_for_rate_limit, RateLimitWaitKind};
use goose_provider_types::errors::ProviderError;
use reqwest::header::HeaderMap;
use reqwest::StatusCode;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

#[derive(Debug, Default)]
pub(crate) struct RateLimitTracker {
    resume_at_by_model: Mutex<HashMap<String, Instant>>,
    project_resume_at: Mutex<Option<Instant>>,
}

impl RateLimitTracker {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn observe(&self, model: &str, headers: &HeaderMap) {
        let status = RateLimitHeaders::from_headers(headers);
        let now = Instant::now();
        if let Some(delay) = status.model_reset_delay() {
            let resume_at = now + delay;
            let mut resume_at_by_model = self.resume_at_by_model.lock().unwrap();
            resume_at_by_model
                .entry(model.to_string())
                .and_modify(|current| *current = (*current).max(resume_at))
                .or_insert(resume_at);
        }
        if let Some(delay) = status.project_reset_delay() {
            let resume_at = now + delay;
            let mut project_resume_at = self.project_resume_at.lock().unwrap();
            *project_resume_at = Some(
                project_resume_at
                    .map(|current| current.max(resume_at))
                    .unwrap_or(resume_at),
            );
        }
    }

    pub(crate) async fn handle_response(
        &self,
        model: &str,
        response: reqwest::Response,
    ) -> Result<reqwest::Response, ProviderError> {
        if response.status() == StatusCode::TOO_MANY_REQUESTS {
            self.observe(model, response.headers());
            if let Some(delay) = extract_retry_after(response.headers(), None) {
                self.observe_retry_delay(model, delay);
            }
        } else if response.status().is_success() {
            self.observe(model, response.headers());
        }

        let result = handle_status(response).await;
        if let Err(ProviderError::RateLimitExceeded {
            retry_delay: Some(delay),
            ..
        }) = &result
        {
            self.observe_retry_delay(model, *delay);
        }
        result
    }

    fn observe_retry_delay(&self, model: &str, delay: std::time::Duration) {
        let resume_at = Instant::now() + delay;
        let mut resume_at_by_model = self.resume_at_by_model.lock().unwrap();
        resume_at_by_model
            .entry(model.to_string())
            .and_modify(|current| *current = (*current).max(resume_at))
            .or_insert(resume_at);
    }

    pub(crate) async fn wait(&self, model: &str) {
        loop {
            let delay = {
                let now = Instant::now();
                let mut resume_at_by_model = self.resume_at_by_model.lock().unwrap();
                if resume_at_by_model
                    .get(model)
                    .is_some_and(|resume_at| *resume_at <= now)
                {
                    resume_at_by_model.remove(model);
                }
                let mut project_resume_at = self.project_resume_at.lock().unwrap();
                if project_resume_at.is_some_and(|resume_at| resume_at <= now) {
                    *project_resume_at = None;
                }
                let resume_at = [resume_at_by_model.get(model).copied(), *project_resume_at]
                    .into_iter()
                    .flatten()
                    .max();
                let Some(resume_at) = resume_at else {
                    return;
                };
                resume_at - now
            };

            if !sleep_for_rate_limit(RateLimitWaitKind::Proactive, delay).await {
                self.resume_at_by_model.lock().unwrap().remove(model);
                *self.project_resume_at.lock().unwrap() = None;
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn proactive_wait_emits_notification() {
        use crate::retry::{with_rate_limit_observer, RateLimitObserverRef};
        use std::sync::Arc;
        use std::time::Duration;

        let tracker = RateLimitTracker::new();
        tracker.resume_at_by_model.lock().unwrap().insert(
            "model".to_string(),
            Instant::now() + Duration::from_millis(1),
        );
        let waits = Arc::new(Mutex::new(Vec::new()));
        let observed_waits = waits.clone();
        let observer: RateLimitObserverRef = Arc::new(move |wait| {
            observed_waits.lock().unwrap().push(wait.kind);
        });

        with_rate_limit_observer(observer, tracker.wait("model")).await;

        assert_eq!(
            waits.lock().unwrap().as_slice(),
            &[RateLimitWaitKind::Proactive]
        );
    }

    #[tokio::test]
    async fn waits_only_for_exhausted_buckets_with_reset_headers() {
        let tracker = RateLimitTracker::new();
        let mut headers = HeaderMap::new();
        headers.insert("x-ratelimit-remaining-requests", "0".parse().unwrap());
        tracker.observe("model", &headers);
        tracker.wait("model").await;
        assert!(tracker.resume_at_by_model.lock().unwrap().is_empty());

        headers.insert("x-ratelimit-reset-requests", "1h".parse().unwrap());
        tracker.observe("model", &headers);
        assert!(tracker
            .resume_at_by_model
            .lock()
            .unwrap()
            .contains_key("model"));
        tracker
            .resume_at_by_model
            .lock()
            .unwrap()
            .insert("model".to_string(), Instant::now());
        tracker.wait("model").await;
        assert!(tracker.resume_at_by_model.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn wait_rechecks_a_deadline_extended_during_the_wait() {
        use std::sync::Arc;
        use std::time::Duration;

        let tracker = Arc::new(RateLimitTracker::new());
        let started = Instant::now();
        tracker
            .resume_at_by_model
            .lock()
            .unwrap()
            .insert("model".to_string(), started + Duration::from_millis(200));

        let waiter = {
            let tracker = tracker.clone();
            tokio::spawn(async move { tracker.wait("model").await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        tracker.resume_at_by_model.lock().unwrap().insert(
            "model".to_string(),
            Instant::now() + Duration::from_millis(250),
        );
        waiter.await.unwrap();

        assert!(started.elapsed() >= Duration::from_millis(200));
        assert!(tracker.resume_at_by_model.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn project_token_wait_applies_to_other_models() {
        use crate::retry::{with_rate_limit_observer, RateLimitObserverRef};
        use std::sync::Arc;

        let tracker = RateLimitTracker::new();
        let mut headers = HeaderMap::new();
        headers.insert("x-ratelimit-remaining-project-tokens", "0".parse().unwrap());
        headers.insert("x-ratelimit-reset-project-tokens", "1ms".parse().unwrap());
        tracker.observe("model-a", &headers);

        let waits = Arc::new(Mutex::new(Vec::new()));
        let observed_waits = waits.clone();
        let observer: RateLimitObserverRef = Arc::new(move |wait| {
            observed_waits.lock().unwrap().push(wait.kind);
        });
        with_rate_limit_observer(observer, tracker.wait("model-b")).await;

        assert_eq!(
            waits.lock().unwrap().as_slice(),
            &[RateLimitWaitKind::Proactive]
        );
        assert!(tracker.project_resume_at.lock().unwrap().is_none());
        assert!(!tracker
            .resume_at_by_model
            .lock()
            .unwrap()
            .contains_key("model-b"));
    }

    #[tokio::test]
    async fn rate_limited_response_shares_its_retry_deadline() {
        use crate::retry::{with_rate_limit_observer, RateLimitObserverRef};
        use std::sync::Arc;

        let tracker = RateLimitTracker::new();
        let response = http::Response::builder()
            .status(StatusCode::TOO_MANY_REQUESTS)
            .header("x-ratelimit-remaining-requests", "0")
            .header("x-ratelimit-reset-requests", "20ms")
            .body(reqwest::Body::from(
                r#"{"error":{"message":"rate limited"}}"#,
            ))
            .unwrap()
            .into();

        let error = tracker
            .handle_response("model", response)
            .await
            .unwrap_err();
        let ProviderError::RateLimitExceeded {
            retry_delay: Some(delay),
            ..
        } = error
        else {
            panic!("expected a rate-limit retry delay");
        };
        assert!(tracker
            .resume_at_by_model
            .lock()
            .unwrap()
            .contains_key("model"));

        sleep_for_rate_limit(
            RateLimitWaitKind::Retry {
                attempt: 1,
                max_retries: 1,
            },
            delay,
        )
        .await;
        let waits = Arc::new(Mutex::new(Vec::new()));
        let observed_waits = waits.clone();
        let observer: RateLimitObserverRef = Arc::new(move |wait| {
            observed_waits.lock().unwrap().push(wait.kind);
        });
        with_rate_limit_observer(observer, tracker.wait("model")).await;

        assert!(waits.lock().unwrap().is_empty());
        assert!(tracker.resume_at_by_model.lock().unwrap().is_empty());
    }

    #[test]
    fn shorter_reset_does_not_replace_longer_exhausted_bucket() {
        use std::time::Duration;

        let tracker = RateLimitTracker::new();
        let mut headers = HeaderMap::new();
        headers.insert("x-ratelimit-remaining-requests", "0".parse().unwrap());
        headers.insert("x-ratelimit-reset-requests", "60s".parse().unwrap());
        tracker.observe("model", &headers);
        let request_deadline = tracker.resume_at_by_model.lock().unwrap()["model"];

        let mut headers = HeaderMap::new();
        headers.insert("x-ratelimit-remaining-tokens", "0".parse().unwrap());
        headers.insert("x-ratelimit-reset-tokens", "1s".parse().unwrap());
        tracker.observe("model", &headers);

        let current = tracker.resume_at_by_model.lock().unwrap()["model"];
        assert_eq!(current, request_deadline);
        assert!(current > Instant::now() + Duration::from_secs(1));
    }
}
