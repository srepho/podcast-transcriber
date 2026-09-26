//! RSS/Atom fetching and parsing, including "rel=next" pagination for full history.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use feed_rs::model::{Entry, Feed as RawFeed};

pub const USER_AGENT: &str = concat!("podcast-transcriber/", env!("CARGO_PKG_VERSION"));

const AUDIO_EXTS: [&str; 8] = [
    ".mp3", ".m4a", ".mp4", ".aac", ".ogg", ".opus", ".wav", ".flac",
];

#[derive(Debug, Clone, PartialEq)]
pub struct FeedEpisode {
    pub guid: String,
    pub title: String,
    pub published: Option<DateTime<Utc>>,
    pub audio_url: String,
    /// Show notes with HTML stripped.
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct ParsedFeed {
    pub title: String,
    pub episodes: Vec<FeedEpisode>,
    /// URL of the next (older) page, if the feed advertises one.
    pub next_page: Option<String>,
}

/// Fetch the first page of a feed.
pub fn fetch(client: &reqwest::blocking::Client, url: &str) -> Result<ParsedFeed> {
    let bytes = client
        .get(url)
        .send()
        .map_err(crate::http::safe_error)
        .with_context(|| format!("fetching feed {}", crate::http::display_url(url)))?
        .error_for_status()
        .map_err(crate::http::safe_error)
        .with_context(|| format!("fetching feed {}", crate::http::display_url(url)))?
        .bytes()
        .map_err(crate::http::safe_error)?;
    parse(&bytes).map_err(|_| {
        anyhow::anyhow!(
            "invalid feed document from {}",
            crate::http::display_url(url)
        )
    })
}

/// Fetch every page of a feed by following `rel="next"` links. Most podcast feeds have a single
/// page containing their full history; some (notably those on WordPress or with >300 episodes)
/// paginate. Returns all episodes, newest first, de-duplicated by guid.
pub fn fetch_all(
    client: &reqwest::blocking::Client,
    url: &str,
    max_pages: usize,
    mut on_page: impl FnMut(usize, usize),
) -> Result<ParsedFeed> {
    let mut first = fetch(client, url)?;
    let mut seen: std::collections::HashSet<String> =
        first.episodes.iter().map(|e| e.guid.clone()).collect();
    let mut visited = std::collections::HashSet::from([url.to_string()]);
    let mut next = first.next_page.clone();
    let mut page = 1;
    on_page(page, first.episodes.len());
    while let Some(next_url) = next.take() {
        if page >= max_pages || !visited.insert(next_url.clone()) {
            break;
        }
        let p = fetch(client, &next_url)?;
        page += 1;
        let mut added = 0;
        for e in p.episodes {
            if seen.insert(e.guid.clone()) {
                first.episodes.push(e);
                added += 1;
            }
        }
        on_page(page, added);
        if added == 0 {
            break;
        }
        next = p.next_page;
    }
    first
        .episodes
        .sort_by_key(|e| std::cmp::Reverse(e.published));
    first.next_page = None;
    Ok(first)
}

pub fn parse(bytes: &[u8]) -> Result<ParsedFeed> {
    // Leave missing IDs empty so the fallback is chosen per entry, including mixed feeds.
    let raw: RawFeed = feed_rs::parser::Builder::new()
        .id_generator(|_, _, _| String::new())
        .build()
        .parse(bytes)?;
    let title = raw
        .title
        .as_ref()
        .map(|t| t.content.trim().to_string())
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| "Untitled feed".to_string());
    let next_page = raw
        .links
        .iter()
        .find(|l| l.rel.as_deref() == Some("next"))
        .map(|l| l.href.clone());
    let mut episodes: Vec<FeedEpisode> = raw.entries.iter().filter_map(entry_to_episode).collect();
    episodes.sort_by_key(|e| std::cmp::Reverse(e.published));
    Ok(ParsedFeed {
        title,
        episodes,
        next_page,
    })
}

fn entry_to_episode(entry: &Entry) -> Option<FeedEpisode> {
    let audio_url = find_audio_url(entry)?;
    let guid = if !entry.id.trim().is_empty() {
        entry.id.clone()
    } else {
        audio_url
            .split(['?', '#'])
            .next()
            .unwrap_or(&audio_url)
            .to_string()
    };
    let title = entry
        .title
        .as_ref()
        .map(|t| t.content.trim().to_string())
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| "Untitled episode".to_string());
    let description = entry
        .summary
        .as_ref()
        .map(|t| t.content.as_str())
        .or(entry.content.as_ref().and_then(|c| c.body.as_deref()))
        .map(crate::vocab::strip_html)
        .unwrap_or_default();
    Some(FeedEpisode {
        guid,
        title,
        published: entry.published.or(entry.updated),
        audio_url,
        description,
    })
}

fn find_audio_url(entry: &Entry) -> Option<String> {
    // RSS <enclosure> and MediaRSS <media:content> both land in entry.media[].content[].
    for m in &entry.media {
        for c in &m.content {
            if let Some(url) = &c.url {
                let mime = c.content_type.as_ref().map(|t| t.to_string());
                if looks_like_audio(url.as_str(), mime.as_deref()) {
                    return Some(url.to_string());
                }
            }
        }
    }
    // Atom <link rel="enclosure">.
    for l in &entry.links {
        if l.rel.as_deref() == Some("enclosure")
            && looks_like_audio(&l.href, l.media_type.as_deref())
        {
            return Some(l.href.clone());
        }
    }
    None
}

pub fn looks_like_audio(url: &str, mime: Option<&str>) -> bool {
    if let Some(m) = mime {
        let m = m.to_ascii_lowercase();
        if m.starts_with("audio/") || m.starts_with("video/") {
            return true;
        }
    }
    let path = url
        .split(['?', '#'])
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    AUDIO_EXTS.iter().any(|e| path.ends_with(e))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RSS: &str = r#"<?xml version="1.0"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
<channel>
  <title>Test Cast</title>
  <atom:link rel="next" href="https://example.com/feed?page=2"/>
  <item>
    <title>Older</title>
    <guid isPermaLink="false">ep-1</guid>
    <pubDate>Mon, 01 Jan 2024 10:00:00 +0000</pubDate>
    <enclosure url="https://cdn.example.com/1.mp3?x=1" type="audio/mpeg" length="123"/>
  </item>
  <item>
    <title>Newer</title>
    <guid>ep-2</guid>
    <pubDate>Tue, 02 Jan 2024 10:00:00 +0000</pubDate>
    <enclosure url="https://cdn.example.com/2" type="audio/mpeg"/>
  </item>
  <item>
    <title>No audio</title>
    <guid>ep-3</guid>
    <link>https://example.com/post</link>
  </item>
  <item>
    <title>No guid</title>
    <enclosure url="https://cdn.example.com/4.m4a" type="text/plain"/>
  </item>
</channel></rss>"#;

    #[test]
    fn parses_rss_with_enclosures_newest_first() {
        let f = parse(RSS.as_bytes()).unwrap();
        assert_eq!(f.title, "Test Cast");
        assert_eq!(
            f.next_page.as_deref(),
            Some("https://example.com/feed?page=2")
        );
        let titles: Vec<_> = f.episodes.iter().map(|e| e.title.as_str()).collect();
        assert_eq!(titles, ["Newer", "Older", "No guid"]);
        assert_eq!(f.episodes[0].guid, "ep-2");
        assert_eq!(f.episodes[1].audio_url, "https://cdn.example.com/1.mp3?x=1");
        // wrong mime but .m4a extension still counts as audio
        assert_eq!(f.episodes[2].audio_url, "https://cdn.example.com/4.m4a");
        assert!(f.episodes[2].published.is_none());
    }

    #[test]
    fn feed_without_guids_keys_by_audio_url() {
        let rss = r#"<rss version="2.0"><channel><title>NoGuid</title>
          <item><title>A</title><enclosure url="https://cdn/a.mp3?token=1" type="audio/mpeg"/></item>
          <item><title>B</title><enclosure url="https://cdn/b.mp3" type="audio/mpeg"/></item>
        </channel></rss>"#;
        let f1 = parse(rss.as_bytes()).unwrap();
        let f2 = parse(rss.as_bytes()).unwrap();
        let g1: Vec<_> = f1.episodes.iter().map(|e| e.guid.clone()).collect();
        let g2: Vec<_> = f2.episodes.iter().map(|e| e.guid.clone()).collect();
        assert_eq!(g1, g2, "ids must be stable across parses");
        assert!(g1.contains(&"https://cdn/a.mp3".to_string()));
    }

    #[test]
    fn audio_detection() {
        assert!(looks_like_audio("https://x/a", Some("audio/mpeg")));
        assert!(looks_like_audio("https://x/a.MP3?token=1", None));
        assert!(!looks_like_audio("https://x/a.html", Some("text/html")));
        assert!(!looks_like_audio("https://x/a", None));
    }

    #[test]
    fn mixed_guid_feed_is_stable_across_refreshes() {
        let first = parse(RSS.as_bytes()).unwrap();
        let second = parse(RSS.as_bytes()).unwrap();
        assert_eq!(first.episodes, second.episodes);
        assert_eq!(first.episodes[2].guid, "https://cdn.example.com/4.m4a");
        let db = crate::db::Db::open_memory().unwrap();
        let feed = crate::config::Feed {
            name: "test".into(),
            url: "https://example.com/feed".into(),
            enabled: true,
            prompt: None,
        };
        assert_eq!(
            crate::pipeline::ingest(&db, &feed, &first, 3)
                .unwrap()
                .queued,
            3
        );
        assert_eq!(
            crate::pipeline::ingest(&db, &feed, &second, 3)
                .unwrap()
                .queued,
            0
        );
    }
}
