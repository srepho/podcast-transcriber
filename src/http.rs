//! HTTP defaults and diagnostics that never expose subscription URLs.

use anyhow::Result;
use std::time::Duration;

pub fn client(timeout_secs: u64) -> Result<reqwest::blocking::Client> {
    anyhow::ensure!(timeout_secs > 0, "HTTP timeout must be greater than zero");
    reqwest::blocking::Client::builder()
        .user_agent(crate::feeds::USER_AGENT)
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .map_err(safe_error)
}

/// Tokens can occur in the path as well as the query, fragment or userinfo.
pub fn display_url(value: &str) -> String {
    url::Url::parse(value)
        .ok()
        .filter(|u| matches!(u.scheme(), "http" | "https"))
        .map(|u| format!("{}/[redacted]", u.origin().ascii_serialization()))
        .unwrap_or_else(|| "[redacted URL]".into())
}

/// Do not retain the original error as a source: nested transport errors can contain
/// redirected or signed URLs even after reqwest's top-level URL is removed.
pub fn safe_error(error: reqwest::Error) -> anyhow::Error {
    if let Some(status) = error.status() {
        anyhow::anyhow!("HTTP request failed with status {}", status.as_u16())
    } else if error.is_timeout() {
        anyhow::anyhow!("HTTP request timed out")
    } else if error.is_connect() {
        anyhow::anyhow!("HTTP connection failed")
    } else if error.is_builder() {
        anyhow::anyhow!("invalid HTTP request")
    } else {
        anyhow::anyhow!("HTTP request or response transfer failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_all_url_credentials() {
        assert_eq!(
            display_url("https://user:secret@example.com/private-token/rss?key=secret#secret"),
            "https://example.com/[redacted]"
        );
        assert_eq!(display_url("not-a-url-secret"), "[redacted URL]");
    }

    #[test]
    fn errors_do_not_retain_url_in_chain() {
        let error = client(1)
            .unwrap()
            .get("invalid://secret/path?token=secret")
            .send()
            .unwrap_err();
        let error = safe_error(error);
        assert!(!format!("{error:#}").contains("secret"));
        assert!(!format!("{error:?}").contains("secret"));
    }

    #[test]
    fn longer_download_deadline_allows_slow_transfer() {
        use crate::test_support::{response, serve};
        let (url, server) = serve(vec![(
            response(200, "audio/mpeg", "audio"),
            Duration::from_millis(1300),
        )]);
        assert!(client(1)
            .unwrap()
            .get(&url)
            .send()
            .unwrap_err()
            .is_timeout());
        server.join().unwrap();
        let (url, server) = serve(vec![(
            response(200, "audio/mpeg", "audio"),
            Duration::from_millis(1300),
        )]);
        assert_eq!(
            client(3).unwrap().get(&url).send().unwrap().text().unwrap(),
            "audio"
        );
        server.join().unwrap();
    }

    #[test]
    fn redirect_errors_do_not_expose_redirect_url() {
        use crate::test_support::{response, serve};
        let (target, target_server) = serve(vec![(
            response(403, "text/plain", "denied"),
            Duration::ZERO,
        )]);
        let redirect = format!("HTTP/1.1 302 Found\r\nLocation: {target}/secret-path?token=secret-query\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        let (url, server) = serve(vec![(redirect, Duration::ZERO)]);
        let error = crate::feeds::fetch(&client(3).unwrap(), &format!("{url}/original-secret"))
            .unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("403"));
        assert!(!message.contains("secret"));
        server.join().unwrap();
        target_server.join().unwrap();
    }
}
