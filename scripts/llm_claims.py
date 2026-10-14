"""LLM-proposed claims for the research pilot: one structured-output call per episode transcript.

The model proposes; a person still accepts or rejects every claim (see review_claims.py). Proposals
become ordinary pilot candidates whose review entries are pre-filled but stay `pending`, so the
pilot's finalize/select contract is unchanged. Entity IDs come only from an exact name match against
a catalogue; anything ambiguous or missing is left for the reviewer.

Providers: Anthropic (default), OpenAI, and OpenAI-compatible APIs (DeepSeek, Qwen/DashScope,
Moonshot/Kimi, Zhipu/GLM, or any `--base-url`). Every provider gets the same system prompt and schema
and passes the same validator, so their proposals are comparable; the provider and model are part of
the extractor fingerprint and of every candidate. Only Anthropic and OpenAI enforce the schema
server-side; the others promise JSON only, so `proposals()` is what rejects malformed claims.

Sending a transcript sends private source material to that provider, under its own retention and
jurisdiction. Only `pilot_dataset.py build --extractor llm` does this.
"""
import json
import os
import re
import unicodedata
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

EFFORT = "medium"  # Anthropic only
EXTRACTOR_VERSION = "llm-claims-v2"  # v2: transcripts are sectioned first; only NBA stretches are read
# Base URLs are the providers' documented OpenAI-compatible endpoints; override with base_url when a
# provider moves or for another region. Non-Anthropic models must be named explicitly: model lists
# change faster than this file, and a silent default would make bundles hard to compare.
PROVIDERS: dict[str, dict[str, Any]] = {
    "anthropic": {"key_env": "ANTHROPIC_API_KEY", "default_model": "claude-opus-5-5"},
    "openai": {"key_env": "OPENAI_API_KEY", "base_url": None, "format": "json_schema"},
    # DeepSeek's models reason before answering. Unbounded, a full transcript spent the whole 16k
    # output budget reasoning and returned nothing; with reasoning off, a 2026-10-08 trial invented
    # names (Batum, Porter Jr.) that low effort did not. Low effort plus 32k output is the setting.
    "deepseek": {"key_env": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com", "format": "json_object",
                 "options": {"reasoning_effort": "low"}, "max_tokens": 32000},
    "qwen": {"key_env": "DASHSCOPE_API_KEY", "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
             "format": "json_object"},
    "moonshot": {"key_env": "MOONSHOT_API_KEY", "base_url": "https://api.moonshot.ai/v1", "format": "json_object"},
    "zhipu": {"key_env": "ZHIPU_API_KEY", "base_url": "https://open.bigmodel.cn/api/paas/v4", "format": "json_object"},
    "compatible": {"key_env": "LLM_API_KEY", "base_url": None, "format": "json_object"},  # needs base_url
}
OUTPUT_TOKENS = 16000
MAX_SPAN = 15  # segments one claim may cite
# Default feature-freshness windows, fixed before any evaluation; the reviewer may shorten them.
EXPIRY_DAYS = {"injury": 7, "rotation": 7, "matchup": 3, "form": 14, "transaction": 30, "contract": 180}
CLAIM_TYPES = list(EXPIRY_DAYS)
CERTAINTIES = ["reported_fact", "opinion", "speculation", "conditional", "unknown"]
SCOPES = ["current", "upcoming", "historical", "unknown"]

SYSTEM = """You extract claims about NBA players and teams from a basketball podcast transcript for a \
team-strength forecasting model. The transcript is machine speech-to-text split into numbered segments; \
names can be misspelled and speakers are not labelled.

Extract only claims a forecaster could act on:
- injury: injuries, surgeries, health, expected absences or returns
- rotation: starting lineups, minutes, roles, who is in or out of the rotation
- transaction: trades, signings, waivers, roster moves (completed or reported as agreed)
- contract: contracts, extensions, options, qualifying offers
- matchup: a specific upcoming opponent or scheme note
- form: a sustained change in a player's performance

Skip predictions of team records, betting picks, general praise or criticism, jokes, and history that \
says nothing about the present. One claim per fact; merge repeated mentions; prefer fewer, high-value claims.

For each claim:
- claim_text: one self-contained sentence naming who and what, as the speakers state it. Add no \
outside knowledge. Keep hedges ("expected to", "reportedly").
- certainty: reported_fact only when presented as reported or announced news (a report, a team \
statement, something said at media day); opinion for the speakers' judgement; speculation for a guess; \
conditional for if/then scenarios; unknown otherwise.
- temporal_scope: current (true now), upcoming (expected future state, e.g. returns in November), \
historical (past event with no bearing on the present), unknown.
- entities: the players and teams the claim is about. Use the correct full name ("Jaden McDaniels", \
"Boston Celtics") when the misspelling is clearly that person or team; otherwise copy the name as heard \
and set asr_uncertain.
- segment_start, segment_end: inclusive segment numbers of the supporting passage, at most 15 segments.
- asr_uncertain: true when a name or key word may be a transcription error.
Never invent facts that are not in the transcript."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "claim_text": {"type": "string"},
            "claim_type": {"type": "string", "enum": CLAIM_TYPES},
            "certainty": {"type": "string", "enum": CERTAINTIES},
            "temporal_scope": {"type": "string", "enum": SCOPES},
            "entities": {"type": "array", "items": {
                "type": "object",
                "properties": {"type": {"type": "string", "enum": ["player", "team"]}, "name": {"type": "string"}},
                "required": ["type", "name"], "additionalProperties": False}},
            "segment_start": {"type": "integer"},
            "segment_end": {"type": "integer"},
            "asr_uncertain": {"type": "boolean"},
        },
        "required": ["claim_text", "claim_type", "certainty", "temporal_scope", "entities",
                     "segment_start", "segment_end", "asr_uncertain"],
        "additionalProperties": False}}},
    "required": ["claims"],
    "additionalProperties": False,
}

SECTION_KINDS = ["nba", "other_sport", "non_sport", "ad"]
SECTION_SYSTEM = """You label the parts of a podcast transcript so that only NBA basketball discussion is \
kept. The transcript is machine speech-to-text split into numbered segments.

Cover the whole transcript, in order, with contiguous sections; each is a range of segment numbers \
(start and end inclusive) with one kind:
- nba: discussion of NBA basketball (teams, players, coaches, trades, contracts, injuries, the league).
- other_sport: any other sport, including the NFL, MLB, college sports, soccer and the WNBA.
- non_sport: movies, TV, music, comedy, personal life, show business and anything else.
- ad: sponsor reads, promotions for other shows or products, betting promotions and responsible-gambling \
disclaimers, and listener-support appeals.
Use few, long sections: switch kind only when the topic really changes, and fold brief asides into the \
surrounding section. When NBA talk and something else are mixed, label it nba."""

SECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"sections": {"type": "array", "items": {
        "type": "object",
        "properties": {"start": {"type": "integer"}, "end": {"type": "integer"},
                       "kind": {"type": "string", "enum": SECTION_KINDS}},
        "required": ["start", "end", "kind"], "additionalProperties": False}}},
    "required": ["sections"],
    "additionalProperties": False,
}

# One request shape per task: (system prompt, schema, schema name).
TASKS: dict[str, tuple[str, dict[str, Any], str]] = {
    "claims": (SYSTEM, SCHEMA, "claims"), "sections": (SECTION_SYSTEM, SECTION_SCHEMA, "sections")}

# ESPN team ids, as used by the prediction model (`espn:team:<id>`).
TEAMS = {
    1: "Atlanta Hawks", 2: "Boston Celtics", 3: "New Orleans Pelicans", 4: "Chicago Bulls", 5: "Cleveland Cavaliers",
    6: "Dallas Mavericks", 7: "Denver Nuggets", 8: "Detroit Pistons", 9: "Golden State Warriors", 10: "Houston Rockets",
    11: "Indiana Pacers", 12: "LA Clippers", 13: "Los Angeles Lakers", 14: "Miami Heat", 15: "Milwaukee Bucks",
    16: "Minnesota Timberwolves", 17: "Brooklyn Nets", 18: "New York Knicks", 19: "Orlando Magic", 20: "Philadelphia 76ers",
    21: "Phoenix Suns", 22: "Portland Trail Blazers", 23: "Sacramento Kings", 24: "San Antonio Spurs",
    25: "Oklahoma City Thunder", 26: "Utah Jazz", 27: "Washington Wizards", 28: "Toronto Raptors", 29: "Memphis Grizzlies",
    30: "Charlotte Hornets",
}


class Extractor:
    """Which provider and model propose claims, and how they are asked."""

    def __init__(self, provider: str = "anthropic", model: str | None = None, base_url: str | None = None):
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}; choose from {', '.join(PROVIDERS)}")
        spec = PROVIDERS[provider]
        self.provider = provider
        self.model: str = model or spec.get("default_model") or ""
        self.base_url: str | None = base_url or spec.get("base_url")
        if not self.model:
            raise ValueError(f"--model is required for provider {provider}")
        if provider == "compatible" and not self.base_url:
            raise ValueError("provider compatible needs --base-url")
        self.format = spec.get("format", "anthropic")

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def fingerprint(self) -> dict[str, Any]:
        """Everything that determines what the extractor proposes, for the bundle manifest hash."""
        spec = PROVIDERS[self.provider]
        settings = {"effort": EFFORT} if self.provider == "anthropic" else {
            "format": self.format, "base_url": self.base_url, "options": spec.get("options", {}),
            "max_tokens": spec.get("max_tokens", OUTPUT_TOKENS)}
        return {"provider": self.provider, "model": self.model, **settings, "version": EXTRACTOR_VERSION, "system": SYSTEM,
                "schema": SCHEMA, "section_system": SECTION_SYSTEM, "section_schema": SECTION_SCHEMA,
                "expiry_days": EXPIRY_DAYS, "max_span": MAX_SPAN}

    def request(self, prompt: str, task: str) -> dict[str, Any]:
        if self.provider == "anthropic":
            return anthropic_call(prompt, self.model, task)
        return openai_call(prompt, self, task)

    def __call__(self, prompt: str) -> dict[str, Any]:
        return self.request(prompt, "claims")

    def sections(self, prompt: str) -> dict[str, Any]:
        return self.request(prompt, "sections")


def name_key(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]+", "", text)


SUFFIX = re.compile(r"\s+(jr|sr|ii|iii|iv)\.?$", re.I)


def base_key(name: str) -> str:
    """Key without a generational suffix: speakers say "Jimmy Butler", the catalogue has "Jimmy Butler III"."""
    return name_key(SUFFIX.sub("", name.strip()))


class Catalogue:
    """Exact (case, accent and punctuation-insensitive) name -> model entity id. No fuzzy matching.

    A name also matches with its generational suffix dropped, when that is still unambiguous, and
    through explicit aliases ("Herb Jones => Herbert Jones") for nicknames the catalogue lacks."""

    def __init__(self, players: dict[str, list[tuple[str, bool]]], aliases: dict[str, str] | None = None,
                 base: dict[str, list[tuple[str, bool]]] | None = None):
        self.players = players
        self.base = base or {}
        self.aliases = {name_key(k): v for k, v in (aliases or {}).items()}
        self.teams: dict[str, str] = {}
        nicknames: dict[str, list[int]] = {}
        for team_id, full in TEAMS.items():
            self.teams[name_key(full)] = f"espn:team:{team_id}"
            nicknames.setdefault(name_key(full.split()[-1]), []).append(team_id)
        self.teams[name_key("Los Angeles Clippers")] = "espn:team:12"
        self.teams[name_key("Sixers")] = "espn:team:20"
        self.teams[name_key("Blazers")] = "espn:team:22"
        for nick, ids in nicknames.items():
            if len(ids) == 1:
                self.teams.setdefault(nick, f"espn:team:{ids[0]}")

    @classmethod
    def load(cls, athletes: Path | list[Path] | None, aliases: Path | None = None) -> "Catalogue":
        """JSONL rows (athlete_id, name, active) from one or more files, later files adding players
        the earlier ones lack; aliases lines are `Spoken Name => Catalogue Name`. Missing files are skipped."""
        players: dict[str, list[tuple[str, bool]]] = {}
        base: dict[str, list[tuple[str, bool]]] = {}
        seen: set[str] = set()
        for path in [athletes] if isinstance(athletes, Path) else (athletes or []):
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row["athlete_id"]) in seen:
                    continue
                seen.add(str(row["athlete_id"]))
                entry = (str(row["athlete_id"]), bool(row.get("active")))
                players.setdefault(name_key(row["name"]), []).append(entry)
                base.setdefault(base_key(row["name"]), []).append(entry)
        names: dict[str, str] = {}
        if aliases is not None and aliases.exists():
            for line in aliases.read_text().splitlines():
                spoken, sep, canonical = line.split("#", 1)[0].partition("=>")
                if sep and spoken.strip() and canonical.strip():
                    names[spoken.strip()] = canonical.strip()
        return cls(players, names, base)

    @staticmethod
    def _pick(matches: list[tuple[str, bool]]) -> str | None:
        active = [m for m in matches if m[1]]
        chosen = active if len(active) == 1 else matches
        return f"espn:athlete:{chosen[0][0]}" if len(chosen) == 1 else None

    def resolve(self, kind: str, name: str) -> str | None:
        key = name_key(name)
        if kind == "team":
            return self.teams.get(key)
        if key in self.aliases:
            name, key = self.aliases[key], name_key(self.aliases[key])
        if SUFFIX.search(name.strip()):
            return self._pick(self.players.get(key, []))
        # No suffix said: every suffix variant is a candidate, so a spoken "Gary Trent" reaches the
        # active Gary Trent Jr. rather than his retired father, who matches the bare name exactly.
        return self._pick(self.base.get(base_key(name)) or self.players.get(key, []))


def transcript_prompt(title: str, published: str | None, segments: list[dict[str, Any]]) -> str:
    lines = [f"Episode: {title}", f"Published: {published or 'unknown'}", "", "Transcript segments:"]
    lines += [f"[{s['index']}] {s['corrected_text'].strip()}" for s in segments]
    return "\n".join(lines)


def anthropic_call(prompt: str, model: str, task: str = "claims") -> dict[str, Any]:
    """One streamed structured-output request. Imported lazily so the pilot stays stdlib-only.

    Server-side fallbacks are on: if the model declines (refusal), the API retries on a fallback model."""
    import anthropic

    system, schema, _ = TASKS[task]
    client = anthropic.Anthropic()
    output_config: Any = {"effort": EFFORT, "format": {"type": "json_schema", "schema": schema}}
    with client.beta.messages.stream(
        model=model,
        max_tokens=32000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_config=output_config,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    ) as stream:
        message = stream.get_final_message()
    if message.stop_reason == "refusal":
        raise RuntimeError(f"extraction refused (request {message._request_id})")
    if message.stop_reason == "max_tokens":
        raise RuntimeError(f"extraction truncated at max_tokens (request {message._request_id})")
    text = next(block.text for block in message.content if block.type == "text")
    return json.loads(text)


def parse_json(text: str) -> dict[str, Any]:
    """JSON from a json_object-mode reply; tolerates a Markdown fence, nothing else."""
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    data = json.loads(fenced.group(1) if fenced else text)
    if not isinstance(data, dict):
        raise ValueError("reply is not a JSON object")
    return data


def openai_messages(prompt: str, mode: str, task: str = "claims") -> list[dict[str, str]]:
    system, schema, _ = TASKS[task]
    if mode == "json_object":  # no server-side schema: state it, then validate the reply
        system += "\n\nReply with only a JSON object that matches this JSON Schema:\n" + json.dumps(schema)
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def openai_call(prompt: str, extractor: Extractor, task: str = "claims") -> dict[str, Any]:
    """Chat Completions request for OpenAI and OpenAI-compatible providers."""
    from openai import OpenAI

    key = os.environ.get(PROVIDERS[extractor.provider]["key_env"])
    if not key:
        raise RuntimeError(f"{PROVIDERS[extractor.provider]['key_env']} is not set")
    client = OpenAI(api_key=key, base_url=extractor.base_url)
    spec = PROVIDERS[extractor.provider]
    output_tokens = spec.get("max_tokens", OUTPUT_TOKENS)
    _, schema, schema_name = TASKS[task]
    if extractor.format == "json_schema":
        response_format: dict[str, Any] = {"type": "json_schema",
                                           "json_schema": {"name": schema_name, "schema": schema, "strict": True}}
        limits = {"max_completion_tokens": output_tokens}
    else:
        response_format, limits = {"type": "json_object"}, {"max_tokens": output_tokens}
    request: dict[str, Any] = {"model": extractor.model, "messages": openai_messages(prompt, extractor.format, task),
                               "response_format": response_format, **limits, **spec.get("options", {})}
    response = client.chat.completions.create(**request)
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise RuntimeError(f"{extractor.name} reply truncated at the output limit")
    if getattr(choice.message, "refusal", None):
        raise RuntimeError(f"{extractor.name} refused: {choice.message.refusal}")
    return parse_json(choice.message.content or "")


def proposals(response: dict[str, Any], n_segments: int) -> list[dict[str, Any]]:
    """Validate the model's output; drop (never repair) claims that cite impossible evidence."""
    out = []
    for claim in response.get("claims", []):
        start, end = claim.get("segment_start"), claim.get("segment_end")
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end < n_segments
                and end - start < MAX_SPAN):
            continue
        if (claim.get("claim_type") not in CLAIM_TYPES or claim.get("certainty") not in CERTAINTIES
                or claim.get("temporal_scope") not in SCOPES or not str(claim.get("claim_text", "")).strip()):
            continue
        entities = [e for e in claim.get("entities", []) if e.get("type") in ("player", "team") and str(e.get("name", "")).strip()]
        out.append({**claim, "claim_text": claim["claim_text"].strip(), "entities": entities})
    return out


def segment_kinds(response: dict[str, Any], n_segments: int) -> list[str]:
    """Kind of every segment. Ranges are clipped to the transcript; invalid ones are ignored; a segment
    no section covers stays `nba`, so a sloppy labelling can never silently drop basketball talk."""
    kinds = ["nba"] * n_segments
    for section in response.get("sections", []):
        start, end, kind = section.get("start"), section.get("end"), section.get("kind")
        if isinstance(start, int) and isinstance(end, int) and kind in SECTION_KINDS and start <= end:
            for i in range(max(0, start), min(n_segments - 1, end) + 1):
                kinds[i] = kind
    return kinds


def runs(kinds: list[str]) -> list[dict[str, Any]]:
    """Contiguous runs of one kind, as stored in the bundle's sections.jsonl."""
    out: list[dict[str, Any]] = []
    for i, kind in enumerate(kinds):
        if out and out[-1]["kind"] == kind:
            out[-1]["end_index"] = i
        else:
            out.append({"start_index": i, "end_index": i, "kind": kind})
    return out


def episode_candidates(
    episode_id: str, transcript_sha256: str, title: str, published: str | None, segments: list[dict[str, Any]],
    catalogue: Catalogue, digest: Callable[[Any], str], timestamp: Callable[[str], Any],
    call: Callable[[str], dict[str, Any]], extractor_name: str,
    sections_call: Callable[[str], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Pilot candidates, pre-filled (still pending) review entries and the section map for one episode.

    With `sections_call`, the transcript is first labelled (nba / other_sport / non_sport / ad) and only
    NBA segments are sent for extraction, keeping their original numbers; a claim must start and end
    inside NBA segments. An episode without NBA segments makes no extraction request."""
    kinds = segment_kinds(sections_call(transcript_prompt(title, published, segments)), len(segments)) \
        if sections_call is not None else ["nba"] * len(segments)
    kept = [s for s, kind in zip(segments, kinds, strict=True) if kind == "nba"]
    claims = [c for c in proposals(call(transcript_prompt(title, published, kept)), len(segments))
              if kinds[c["segment_start"]] == "nba" and kinds[c["segment_end"]] == "nba"] if kept else []
    candidates, reviews = [], []
    for claim in claims:
        span = segments[claim["segment_start"]:claim["segment_end"] + 1]
        candidate_id = digest([episode_id, transcript_sha256, claim["segment_start"], claim["segment_end"],
                               claim["claim_text"], EXTRACTOR_VERSION, extractor_name])
        entities = [{"type": e["type"], "name": e["name"], "model_entity_id": catalogue.resolve(e["type"], e["name"]) or ""}
                    for e in claim["entities"]]
        valid_until = None
        if published:
            valid_until = (timestamp(published) + timedelta(days=EXPIRY_DAYS[claim["claim_type"]])).isoformat()
        candidates.append({"candidate_id": candidate_id, "episode_id": episode_id, "transcript_sha256": transcript_sha256,
            "segment_ids": [s["segment_id"] for s in span], "start_secs": span[0]["start_secs"], "end_secs": span[-1]["end_secs"],
            "raw_evidence": " ".join(s["raw_text"].strip() for s in span),
            "corrected_evidence": " ".join(s["corrected_text"].strip() for s in span),
            "suggested_types": [claim["claim_type"]], "entity_suggestions": sorted({e["name"] for e in claim["entities"]}),
            "uncertainty_markers": [], "extractor_version": EXTRACTOR_VERSION, "extractor_model": extractor_name,
            "proposal": {k: claim[k] for k in ("claim_text", "claim_type", "certainty", "temporal_scope", "entities",
                                                 "asr_uncertain")}})
        reviews.append({"candidate_id": candidate_id, "status": "pending", "claim_type": claim["claim_type"],
            "claim_text": claim["claim_text"], "certainty": claim["certainty"], "temporal_scope": claim["temporal_scope"],
            "entities": entities, "valid_until": valid_until, "audio_checked": False, "notes": ""})
    return candidates, reviews, runs(kinds)

