//! Per-feed vocabulary: correct names that speech-to-text mangles.
//!
//! Vocab file format (one entry per line, `#` comments):
//!   Giannis Antetokounmpo            # fuzzy-matched against the transcript
//!   nicole the yockage => Nikola Jokić   # explicit alias, exact (case-insensitive) phrase replace
//!
//! Fuzzy matching compares letters-only lowercase strings, so "Yokic" ~ "Jokic" and
//! "Antetokumpo" ~ "Antetokounmpo". Whisper sometimes splits a name across extra words
//! ("Gilgeous Alexander" -> "Jill just Alexander"), so windows of varying word counts are tried.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct Term {
    pub canonical: String,
    /// lowercase letters/digits only, no spaces
    pub key: String,
    pub words: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Alias {
    pub from_words: Vec<String>, // normalized words
    pub to: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Vocab {
    pub terms: Vec<Term>,
    pub aliases: Vec<Alias>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Correction {
    pub from: String,
    pub to: String,
    pub count: usize,
}

pub fn normalize_word(w: &str) -> String {
    w.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .map(fold_diacritic)
        .collect()
}

fn fold_diacritic(c: char) -> char {
    match c {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' => 'a',
        'ç' => 'c',
        'è' | 'é' | 'ê' | 'ë' => 'e',
        'ì' | 'í' | 'î' | 'ï' => 'i',
        'ñ' => 'n',
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' => 'o',
        'ù' | 'ú' | 'û' | 'ü' => 'u',
        'ý' | 'ÿ' => 'y',
        'š' => 's',
        'ž' => 'z',
        'č' | 'ć' => 'c',
        'đ' => 'd',
        _ => c,
    }
}

impl Vocab {
    pub fn parse(text: &str) -> Vocab {
        let mut v = Vocab::default();
        for line in text.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            if let Some((from, to)) = line.split_once("=>") {
                let from_words: Vec<String> = from
                    .split_whitespace()
                    .map(normalize_word)
                    .filter(|w| !w.is_empty())
                    .collect();
                let to = to.trim().to_string();
                if !from_words.is_empty() && !to.is_empty() {
                    v.aliases.push(Alias { from_words, to });
                }
            } else {
                v.add_term(line);
            }
        }
        v
    }

    pub fn add_term(&mut self, canonical: &str) {
        let canonical = canonical.trim();
        let key: String = canonical.split_whitespace().map(normalize_word).collect();
        if key.is_empty() || self.terms.iter().any(|t| t.key == key) {
            return;
        }
        self.terms.push(Term {
            canonical: canonical.to_string(),
            key,
            words: canonical.split_whitespace().count(),
        });
    }

    pub fn load(path: &Path) -> Result<Vocab> {
        if !path.exists() {
            return Ok(Vocab::default());
        }
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Ok(Vocab::parse(&text))
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty() && self.aliases.is_empty()
    }

    /// Canonical names, for use in the whisper prompt.
    pub fn names(&self) -> Vec<&str> {
        self.terms.iter().map(|t| t.canonical.as_str()).collect()
    }

    /// Apply aliases then fuzzy term matching. Returns corrected text and what changed.
    pub fn correct(&self, text: &str, threshold: f64) -> (String, Vec<Correction>) {
        let mut tokens = tokenize(text);
        let mut corrections: Vec<Correction> = vec![];
        let mut record = |from: String, to: String| {
            if let Some(c) = corrections
                .iter_mut()
                .find(|c| c.from == from && c.to == to)
            {
                c.count += 1;
            } else {
                corrections.push(Correction { from, to, count: 1 });
            }
        };

        // Word indices into tokens (tokens alternate word / separator).
        let word_idx: Vec<usize> = tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| t.is_word)
            .map(|(i, _)| i)
            .collect();
        let norm: Vec<String> = word_idx
            .iter()
            .map(|&i| normalize_word(&tokens[i].text))
            .collect();
        let mut consumed = vec![false; word_idx.len()];
        // Guard rails against false positives on ordinary prose: a window must contain a
        // capitalized word (whisper capitalizes names), may not start or end on a stopword
        // ("to remain" must never become "Tre Mann"), and may not cross a sentence boundary.
        let cap: Vec<bool> = word_idx
            .iter()
            .map(|&i| {
                tokens[i]
                    .text
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_uppercase())
            })
            .collect();
        let boundary_after: Vec<bool> = word_idx
            .iter()
            .enumerate()
            .map(|(w, &i)| {
                let sep = tokens
                    .get(i + 1)
                    .filter(|t| !t.is_word)
                    .map(|t| t.text.as_str())
                    .unwrap_or("");
                sep.contains(['?', '!', ';', ':', '\n']) || (sep.contains('.') && norm[w].len() > 2)
            })
            .collect();
        let window_ok = |i: usize, width: usize| -> bool {
            cap[i..i + width].iter().any(|&c| c)
                && !is_stopword(&norm[i])
                && !is_stopword(&norm[i + width - 1])
                && !boundary_after[i..i + width - 1].iter().any(|&b| b)
        };

        // 1. explicit aliases (longest first so multi-word aliases win)
        let mut aliases: Vec<&Alias> = self.aliases.iter().collect();
        aliases.sort_by_key(|a| std::cmp::Reverse(a.from_words.len()));
        for a in aliases {
            let n = a.from_words.len();
            let mut i = 0;
            while i + n <= norm.len() {
                if !consumed[i..i + n].iter().any(|&c| c) && norm[i..i + n] == a.from_words[..] {
                    let from = join_words(&tokens, &word_idx[i..i + n]);
                    replace_span(&mut tokens, &word_idx[i..i + n], &a.to);
                    consumed[i..i + n].iter_mut().for_each(|c| *c = true);
                    record(from, a.to.clone());
                    i += n;
                } else {
                    i += 1;
                }
            }
        }

        // 2. fuzzy terms. Score every (term, window) pair, then accept the best-scoring
        // non-overlapping matches globally, so an exact "Anthony Slater" always beats a
        // near-miss "Anthony Carter" and a full "Jaren Jackson Jr" beats a bare "Jackson".
        let mut cands: Vec<(f64, usize, usize, usize)> = vec![]; // (score, start, width, term)
        for (ti, term) in self.terms.iter().enumerate() {
            if term.key.len() < 4 {
                continue;
            }
            let max_w = (term.words + 2).min(norm.len().max(1));
            for width in 1..=max_w {
                for i in 0..norm.len().saturating_sub(width - 1) {
                    if !window_ok(i, width) {
                        continue;
                    }
                    let candidate: String = norm[i..i + width].concat();
                    let mut score = match_score(&candidate, &term.key);
                    if score >= required(threshold, &term.key) && width == term.words && width > 1 {
                        // Same word count: the surname must hold up on its own, otherwise a
                        // shared first name carries a wrong surname over the line.
                        let last_cand = &norm[i + width - 1];
                        let last_term = term
                            .canonical
                            .split_whitespace()
                            .last()
                            .map(normalize_word)
                            .unwrap_or_default();
                        if match_score(last_cand, &last_term) < 0.7 {
                            score = 0.0;
                        }
                    }
                    if score >= required(threshold, &term.key) {
                        cands.push((score, i, width, ti));
                    }
                }
            }
        }
        cands.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| self.terms[b.3].key.len().cmp(&self.terms[a.3].key.len()))
        });
        for (_, i, width, ti) in cands {
            if consumed[i..i + width].iter().any(|&c| c) {
                continue;
            }
            let term = &self.terms[ti];
            let from = join_words(&tokens, &word_idx[i..i + width]);
            if from != term.canonical {
                replace_span(&mut tokens, &word_idx[i..i + width], &term.canonical);
                record(from, term.canonical.clone());
            }
            consumed[i..i + width].iter_mut().for_each(|c| *c = true);
        }

        (
            tokens.iter().map(|t| t.text.as_str()).collect(),
            corrections,
        )
    }
}

fn is_stopword(w: &str) -> bool {
    const STOP: &[&str] = &[
        "a", "an", "the", "and", "or", "but", "so", "if", "then", "than", "to", "of", "in", "on",
        "at", "by", "for", "from", "with", "as", "into", "over", "about", "up", "out", "is", "was",
        "are", "were", "be", "been", "will", "would", "can", "could", "should", "have", "has",
        "had", "do", "did", "does", "not", "no", "he", "she", "it", "they", "we", "you", "i",
        "his", "her", "their", "our", "its", "my", "your", "that", "this", "these", "those",
        "there", "here", "who", "what", "when", "where", "which", "just", "also", "very", "really",
        "like", "got", "get", "one",
    ];
    STOP.contains(&w)
}

/// Short keys need a stricter score: "Jokic" must not absorb "joking".
fn required(threshold: f64, key: &str) -> f64 {
    if key.len() < 6 {
        threshold.max(0.85)
    } else {
        threshold
    }
}

/// Fold letter confusions that speech-to-text makes constantly (Yokic/Jokic, Doncic/Donchich)
/// so they compare as equal. Both sides are folded, so this is symmetric.
fn fold_phonetic(s: &str) -> String {
    let s = s
        .replace("ph", "f")
        .replace("ck", "k")
        .replace("ch", "k")
        .replace("sh", "s")
        .replace("th", "t");
    let mut out = String::with_capacity(s.len());
    let mut prev = '\0';
    for c in s.chars() {
        let c = match c {
            'y' => 'j',
            'c' | 'q' => 'k',
            'z' => 's',
            'w' => 'v',
            'h' => continue,
            c => c,
        };
        if c != prev {
            out.push(c);
        }
        prev = c;
    }
    out
}

/// 0..=1 similarity between a normalized transcript fragment and a vocab key.
fn match_score(candidate: &str, key: &str) -> f64 {
    if candidate == key {
        return 1.0;
    }
    if candidate.is_empty() {
        return 0.0;
    }
    // A 4-letter fragment should never become a 20-letter name, nor vice versa.
    let (a, b) = (candidate.len() as f64, key.len() as f64);
    if a < b * 0.6 || a > b * 1.4 {
        return 0.0;
    }
    let pair_score = |x: &str, y: &str| -> f64 {
        let lev = strsim::normalized_levenshtein(x, y);
        let jw = strsim::jaro_winkler(x, y);
        let same_start = x.chars().next() == y.chars().next();
        // Jaro-Winkler over-rewards shared prefixes, so it only helps when Levenshtein is
        // already respectable; without a shared first letter it can only hurt.
        if same_start {
            lev.max((jw - 0.05).min(lev + 0.10))
        } else {
            lev.min(jw)
        }
    };
    let raw = pair_score(candidate, key);
    let folded = pair_score(&fold_phonetic(candidate), &fold_phonetic(key));
    // Extra or missing letters (a swallowed neighbouring word) cost proportionally.
    let len_penalty = 0.5 * (a - b).abs() / b.max(1.0);
    (raw.max(folded) - len_penalty).clamp(0.0, 1.0)
}

#[derive(Debug, Clone)]
struct Token {
    text: String,
    is_word: bool,
}

fn tokenize(text: &str) -> Vec<Token> {
    let mut out: Vec<Token> = vec![];
    for ch in text.chars() {
        // apostrophes, hyphens and periods inside words stay with the word ("Gilgeous-Alexander", "P.J.")
        let word_char = ch.is_alphanumeric() || ch == '\'' || ch == '’' || ch == '-' || ch == '.';
        match out.last_mut() {
            Some(t) if t.is_word == word_char => t.text.push(ch),
            _ => out.push(Token {
                text: ch.to_string(),
                is_word: word_char,
            }),
        }
    }
    // A trailing "." or "-" on a word is punctuation, not part of it; split it off.
    let mut fixed: Vec<Token> = vec![];
    for t in out {
        if t.is_word
            && t.text.len() > 1
            && (t.text.ends_with('.') || t.text.ends_with('-') || t.text.ends_with(','))
        {
            let (w, p) = t.text.split_at(t.text.len() - 1);
            fixed.push(Token {
                text: w.to_string(),
                is_word: true,
            });
            fixed.push(Token {
                text: p.to_string(),
                is_word: false,
            });
        } else if t.is_word && t.text.chars().all(|c| !c.is_alphanumeric()) {
            fixed.push(Token {
                text: t.text,
                is_word: false,
            });
        } else {
            fixed.push(t);
        }
    }
    fixed
}

fn join_words(tokens: &[Token], idx: &[usize]) -> String {
    let (first, last) = (idx[0], *idx.last().unwrap());
    tokens[first..=last]
        .iter()
        .map(|t| t.text.as_str())
        .collect()
}

/// Replace the words at `idx` (and separators between them) with `to`; leaves later indices
/// valid by blanking rather than removing tokens.
fn replace_span(tokens: &mut [Token], idx: &[usize], to: &str) {
    let (first, last) = (idx[0], *idx.last().unwrap());
    tokens[first].text = to.to_string();
    for t in tokens.iter_mut().take(last + 1).skip(first + 1) {
        t.text.clear();
    }
}

/// Pull capitalized word runs (likely proper nouns) out of free text, for prompting.
pub fn proper_nouns(text: &str) -> Vec<String> {
    let mut out: BTreeSet<String> = BTreeSet::new();
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut run: Vec<String> = vec![];
    let flush = |run: &mut Vec<String>, out: &mut BTreeSet<String>| {
        if run.len() >= 2 {
            out.insert(run.join(" "));
        }
        run.clear();
    };
    for w in words {
        let clean: String = w
            .trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-')
            .to_string();
        let capitalized = clean
            .chars()
            .next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false)
            && clean.len() > 1;
        if capitalized {
            run.push(clean.clone());
        } else {
            flush(&mut run, &mut out);
        }
        if w.ends_with(['.', ',', ':', ';', '!', '?']) {
            flush(&mut run, &mut out);
        }
    }
    flush(&mut run, &mut out);
    out.into_iter().collect()
}

/// Strip HTML tags and collapse whitespace (episode descriptions are usually HTML).
pub fn strip_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                out.push(' ');
            }
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    let out = out
        .replace("&amp;", "&")
        .replace("&nbsp;", " ")
        .replace("&#39;", "'")
        .replace("&quot;", "\"");
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Build the whisper initial prompt. whisper.cpp keeps only the *last* ~224 tokens, so the
/// most useful material (episode-specific names) goes last.
pub fn build_prompt(
    feed_prompt: Option<&str>,
    global_prompt: Option<&str>,
    title: &str,
    description: &str,
    vocab: &Vocab,
) -> String {
    let mut parts: Vec<String> = vec![];
    if let Some(g) = global_prompt.filter(|s| !s.trim().is_empty()) {
        parts.push(g.trim().to_string());
    }
    if let Some(f) = feed_prompt.filter(|s| !s.trim().is_empty()) {
        parts.push(f.trim().to_string());
    }
    // Vocab names that the description mentions are the most likely to be spoken.
    let desc_nouns = proper_nouns(description);
    let mut mentioned: Vec<&str> = vocab
        .names()
        .into_iter()
        .filter(|n| {
            let nl = n.to_lowercase();
            desc_nouns.iter().any(|d| d.to_lowercase().contains(&nl))
                || description.to_lowercase().contains(&nl)
        })
        .collect();
    mentioned.truncate(40);
    let mut other_nouns: Vec<String> = desc_nouns
        .into_iter()
        .filter(|d| {
            !mentioned
                .iter()
                .any(|m| d.to_lowercase().contains(&m.to_lowercase()))
        })
        .collect();
    other_nouns.truncate(25);
    if !other_nouns.is_empty() {
        parts.push(other_nouns.join(", ") + ".");
    }
    if !mentioned.is_empty() {
        parts.push(mentioned.join(", ") + ".");
    }
    parts.push(title.trim().to_string() + ".");
    parts.join(" ")
}

/// NBA roster import from ESPN's (undocumented but public, keyless) site API.
pub mod nba {
    use anyhow::{Context, Result};

    const TEAMS: &str = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams";

    pub fn fetch_names() -> Result<Vec<String>> {
        // ESPN's edge returns 403 for unknown and for browser-like user agents sent without full
        // browser headers; it accepts curl's.
        let client = reqwest::blocking::Client::builder()
            .user_agent("curl/8.7.1")
            .timeout(std::time::Duration::from_secs(30))
            .build()?;
        let client = &client;
        let teams: serde_json::Value = client
            .get(TEAMS)
            .send()?
            .error_for_status()?
            .json()
            .context("ESPN teams list")?;
        let team_list = teams
            .pointer("/sports/0/leagues/0/teams")
            .and_then(|t| t.as_array())
            .context("unexpected ESPN teams response shape")?;
        let mut names: Vec<String> = vec![];
        for t in team_list {
            let id = t
                .pointer("/team/id")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let team_name = t
                .pointer("/team/displayName")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            if id.is_empty() {
                continue;
            }
            if !team_name.is_empty() {
                names.push(team_name.to_string());
            }
            let roster: serde_json::Value = client
                .get(format!("{TEAMS}/{id}/roster"))
                .send()?
                .error_for_status()?
                .json()
                .with_context(|| format!("ESPN roster for team {id}"))?;
            for a in roster
                .get("athletes")
                .and_then(|a| a.as_array())
                .into_iter()
                .flatten()
            {
                if let Some(n) = a.get("fullName").and_then(|v| v.as_str()) {
                    names.push(n.to_string());
                }
            }
            for c in roster
                .get("coach")
                .and_then(|c| c.as_array())
                .into_iter()
                .flatten()
            {
                let n = format!(
                    "{} {}",
                    c.get("firstName").and_then(|v| v.as_str()).unwrap_or(""),
                    c.get("lastName").and_then(|v| v.as_str()).unwrap_or("")
                );
                if n.trim().len() > 1 {
                    names.push(n.trim().to_string());
                }
            }
            eprintln!("  {team_name}: ok");
        }
        names.sort();
        names.dedup();
        Ok(names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v() -> Vocab {
        Vocab::parse(
            "# names\nGiannis Antetokounmpo\nNikola Jokić\nShai Gilgeous-Alexander\nJaren Jackson Jr.\nLuka Dončić\nBucks\n\nnicole the yockage => Nikola Jokić\n",
        )
    }

    #[test]
    fn parses_terms_and_aliases() {
        let v = v();
        assert_eq!(v.terms.len(), 6);
        assert_eq!(v.terms[1].key, "nikolajokic");
        assert_eq!(v.aliases.len(), 1);
        assert_eq!(v.aliases[0].from_words, ["nicole", "the", "yockage"]);
    }

    #[test]
    fn corrects_near_misses_and_aliases() {
        let v = v();
        let (out, c) = v.correct(
            "Giannis Antetokumpo scored 40. Nicole the yockage had a triple double, and Shai Gilgeous Alexander was great. Luka Doncic too.",
            0.8,
        );
        assert_eq!(
            out,
            "Giannis Antetokounmpo scored 40. Nikola Jokić had a triple double, and Shai Gilgeous-Alexander was great. Luka Dončić too."
        );
        assert_eq!(c.len(), 4);
        assert!(c
            .iter()
            .any(|c| c.from == "Giannis Antetokumpo" && c.count == 1));
    }

    #[test]
    fn leaves_unrelated_text_alone() {
        let v = v();
        let text = "The bucket was full and the jockey rode; Jackson Pollock painted.";
        let (out, c) = v.correct(text, 0.8);
        assert_eq!(out, text, "{c:?}");
        assert!(c.is_empty());
    }

    #[test]
    fn handles_split_names_across_extra_words() {
        let v = v();
        let (out, _) = v.correct(
            "Shay Jill just Alexander was named player of the week.",
            0.75,
        );
        assert_eq!(out, "Shai Gilgeous-Alexander was named player of the week.");
    }

    #[test]
    fn no_false_positives_on_ordinary_prose() {
        let v = Vocab::parse("Tre Mann\nWill Hardy\nBoston Celtics\nStephen Curry\nGiannis Antetokounmpo\nNikola Jokic\n");
        let cases = [
            "Towns wants to remain in New York for now.",
            "The Jazz will have a new look this season.",
            "They usually lost to Boston. The Hawks were the only team to beat them.",
        ];
        for text in cases {
            let (out, c) = v.correct(text, 0.8);
            assert_eq!(out, text, "{c:?}");
        }
        let (out, _) = v.correct(
            "Giannis Antetokumpo and Nikola Yokic were great, as was Steph and Curry.",
            0.8,
        );
        assert_eq!(
            out,
            "Giannis Antetokounmpo and Nikola Jokic were great, as was Stephen Curry."
        );
    }

    #[test]
    fn exact_name_beats_near_miss_and_surname_must_hold() {
        let v = Vocab::parse("Anthony Carter\nAnthony Slater\nAnthony Davis\n");
        let (out, c) = v.correct("Anthony Slater of The Athletic reported it.", 0.8);
        assert_eq!(out, "Anthony Slater of The Athletic reported it.", "{c:?}");
        // Without the reporter in the vocab, the surname check alone must block it.
        let v = Vocab::parse("Anthony Carter\nAnthony Davis\n");
        let (out, c) = v.correct("Anthony Slater of The Athletic reported it.", 0.8);
        assert_eq!(out, "Anthony Slater of The Athletic reported it.", "{c:?}");
        // A genuinely mangled surname still gets fixed.
        let (out, _) = v.correct("Anthony Davies had 30.", 0.8);
        assert_eq!(out, "Anthony Davis had 30.");
    }

    #[test]
    fn phonetic_fold_and_scores() {
        assert_eq!(fold_phonetic("yokic"), fold_phonetic("jokic"));
        assert!(match_score("yokic", "jokic") >= 0.85);
        assert!(match_score("giannisantetokumpo", "giannisantetokounmpo") > 0.85);
        assert!(match_score("giannisantetokumposcored40", "giannisantetokounmpo") < 0.8);
        assert!(match_score("joking", "jokic") < 0.85);
        assert!(match_score("bucket", "bucks") < 0.85);
    }

    #[test]
    fn proper_noun_extraction() {
        let n = proper_nouns("Nate and Danny break down Bucks vs Nuggets. Guest Mike Richman joins; also LeBron James, and the trade.");
        assert!(n.iter().any(|x| x.contains("Mike Richman")), "{n:?}");
        assert!(n.contains(&"LeBron James".to_string()), "{n:?}");
        assert!(
            !n.iter().any(|x| x.contains("Nuggets. Guest")),
            "runs must stop at sentence ends: {n:?}"
        );
    }

    #[test]
    fn strip_html_and_prompt() {
        assert_eq!(
            strip_html("<p>Hi <b>there</b>&amp;more</p>"),
            "Hi there &more"
        );
        let v = v();
        let p = build_prompt(
            Some("NBA podcast."),
            None,
            "Bucks Recap",
            "<p>Giannis Antetokounmpo and Mike Richman discuss.</p>",
            &v,
        );
        assert!(p.starts_with("NBA podcast."));
        assert!(p.ends_with("Bucks Recap."));
        assert!(p.contains("Giannis Antetokounmpo"));
        assert!(p.contains("Mike Richman"));
    }
}
