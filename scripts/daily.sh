#!/bin/bash
# Scheduled collection: refresh, download and transcribe new episodes, then have an LLM propose
# claims for episodes not yet extracted, into a new pending bundle under data/pilots/auto-<UTC stamp>.
# Nothing reaches a model until a person reviews and finalizes the bundle (scripts/review_claims.py).
#
#   scripts/daily.sh FEED PUBLISHED_AFTER        e.g. scripts/daily.sh myshow 2026-10-01T00:00:00Z
#
# PROVIDER (default anthropic: openai, deepseek, qwen, moonshot, zhipu, compatible), MODEL (required
# except for anthropic) and BASE_URL choose the extractor. The provider's key comes from its usual
# environment variable or the login keychain, service "podcast-<provider>":
#   security add-generic-password -a "$USER" -s podcast-deepseek -w
# ATHLETES overrides the entity catalogue (athletes.jsonl with athlete_id, name, active).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
feed="${1:?feed name}"
after="${2:?published-after timestamp}"
athletes="${ATHLETES:-../PredictionMarkets/data/basketball/nba/players/athletes.jsonl}"
provider="${PROVIDER:-anthropic}"
key_env="$(.venv/bin/python scripts/llm_claims.py key-env "$provider")" || exit 1
llm_args=(--provider "$provider")
[ -n "${MODEL:-}" ] && llm_args+=(--model "$MODEL")
[ -n "${BASE_URL:-}" ] && llm_args+=(--base-url "$BASE_URL")

notify() { osascript -e "display notification \"$1\" with title \"Podcast claims\"" >/dev/null 2>&1 || true; }

echo "== $(date -u +%FT%TZ) collect"
# A failed refresh or download must not block extraction of episodes already transcribed.
./target/release/podcast run || echo "podcast run failed (exit $?); continuing with what is transcribed" >&2

if [ -z "${!key_env:-}" ]; then
  key="$(security find-generic-password -s "podcast-$provider" -w 2>/dev/null || true)"
  export "$key_env=$key"
fi
if [ -z "${!key_env:-}" ]; then
  echo "no $provider API key ($key_env or keychain podcast-$provider); skipping extraction" >&2
  notify "No $provider API key; claims not extracted"
  exit 1
fi

out="data/pilots/auto-$(date -u +%Y%m%dT%H%MZ)"
echo "== $(date -u +%FT%TZ) extract -> $out"
.venv/bin/python scripts/pilot_dataset.py build --feed "$feed" --unprocessed --published-after "$after" \
  --extractor llm "${llm_args[@]}" --limit 10 --athletes "$athletes" --out "$out" || { notify "Claim extraction failed; see log"; exit 1; }

if [ -f "$out/manifest.json" ]; then
  summary="$(.venv/bin/python - "$out/manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
failed = len(m["extraction_failures"])
print(f'{m["counts"]["candidates.jsonl"]} claims from {m["counts"]["episodes.jsonl"]} episodes'
      + (f", {failed} failed" if failed else ""))
PY
)"
  echo "$summary"
  notify "$summary to review: scripts/review_claims.py $out --finalize"
fi
