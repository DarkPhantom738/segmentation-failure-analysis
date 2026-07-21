#!/usr/bin/env bash
# Continue a seed downstream chain after TTA export (and optionally after confidence).
# Usage:
#   bash scripts/run_seed_downstream_from_confidence.sh <seed> [--skip-confidence] [--skip-consistency]
# Assumes configs already exist under outputs_converged/seed_*/analysis/configs/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

SEED="${1:?Usage: $0 <seed> [--skip-confidence] [--skip-consistency]}"
SKIP_CONF=0
SKIP_CONS=0
shift || true
for arg in "$@"; do
  case "$arg" in
    --skip-confidence) SKIP_CONF=1 ;;
    --skip-consistency) SKIP_CONS=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$SEED" -lt 1000 ]]; then
  TAG=$(printf "seed_%03d" "$SEED")
  SLUG=$(printf "seed%03d" "$SEED")
else
  TAG="seed_${SEED}"
  SLUG="seed${SEED}"
fi

ANALYSIS="outputs_converged/${TAG}/analysis"
CFG_DIR="${ANALYSIS}/configs"
LOG="${ANALYSIS}/logs/downstream_from_confidence.log"
LAYER_CFG="${CFG_DIR}/converged_${SLUG}_layer_aware.yaml"
CONS_CFG="${CFG_DIR}/converged_${SLUG}_consistency.yaml"
TRIAGE_CFG="${CFG_DIR}/converged_${SLUG}_triage.yaml"
VAL_CFG="${CFG_DIR}/converged_${SLUG}_method_validation.yaml"

mkdir -p "${ANALYSIS}/logs"
exec >>"$LOG" 2>&1

echo "=== seed=${SEED} tag=${TAG} from_confidence skip_conf=${SKIP_CONF} skip_cons=${SKIP_CONS} ==="
if [[ "$SKIP_CONF" -eq 0 ]]; then
  echo "=== confidence features ==="
  python scripts/run_layer_aware_latent_risk.py \
    --config "$LAYER_CFG" \
    --stage confidence
fi

if [[ "$SKIP_CONS" -eq 0 ]]; then
  echo "=== consistency failure detection ==="
  python scripts/run_consistency_failure_detection.py \
    --config "$CONS_CFG"
fi

echo "=== confidence consistency triage ==="
python scripts/run_confidence_consistency_triage.py \
  --config "$TRIAGE_CFG"

TRIAGE_DIR=$(python - <<PY
from pathlib import Path
import yaml
cfg = yaml.safe_load(open("${TRIAGE_CFG}"))
base = Path(cfg["paths"]["output_dir"])
cands = sorted(base.parent.glob(base.name + "*"), key=lambda p: p.stat().st_mtime)
for p in reversed(cands):
    if (p / "aggregate_metrics.csv").exists():
        print(p)
        break
else:
    print(base)
PY
)
echo "Using triage_dir=$TRIAGE_DIR"
python - <<PY
from pathlib import Path
import yaml
p = Path("${VAL_CFG}")
cfg = yaml.safe_load(p.read_text())
cfg["triage_dir"] = "${TRIAGE_DIR}"
cfg["validation_output_dir"] = "${ANALYSIS}/method_validation"
cfg.setdefault("paths", {})["output_dir"] = "${ANALYSIS}/method_validation"
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

echo "=== method validation ==="
python scripts/run_method_validation.py \
  --config "$VAL_CFG"

echo "=== write seed_analysis_summary.md ==="
python - <<PY
import json
from pathlib import Path
import pandas as pd
import yaml

seed = int("${SEED}")
tag = "${TAG}"
analysis = Path(f"outputs_converged/{tag}/analysis")
summary_ckpt = json.loads(Path(f"outputs_converged/{tag}/convergence_summary.json").read_text())
fail = pd.read_csv(analysis / "failure_tables/failure_metrics.csv")
cons = pd.read_csv(analysis / "consistency/case_level_features.csv")
mean_fg = float(cons["label_mean_fg_dice"].mean()) if "label_mean_fg_dice" in cons.columns else float("nan")

triage_cfg = yaml.safe_load(open("${VAL_CFG}"))
triage_dir = Path(triage_cfg["triage_dir"])
agg = pd.read_csv(triage_dir / "aggregate_metrics.csv")
primary = agg[(agg["failure_def"] == "lowest20_mean_fg")]
conf = primary[primary["method"] == "confidence"].iloc[0]
prop = primary[primary["method"] == "conf_consistency"].iloc[0]

auprc_c = float(conf["auprc"]); auprc_p = float(prop["auprc"])
cap_c = float(conf["capture_at_20"]); cap_p = float(prop["capture_at_20"])
da = auprc_p - auprc_c
dc = cap_p - cap_c
verdict = "POSITIVE REPLICATION" if (da > 0 and dc > 0) else "NO IMPROVEMENT"

lines = [
    f"# Seed {seed} downstream analysis summary",
    "",
    f"- **seed:** {seed}",
    f"- **best checkpoint epoch:** {summary_ckpt['best_epoch']}",
    f"- **development mean FG Dice:** {summary_ckpt['best_development_mean_foreground_dice']:.6f}",
    f"- **final 375-case mean foreground Dice:** {mean_fg:.6f}",
    f"- **failure table cases:** {len(fail)}",
    "",
    "## Primary endpoint: lowest20_mean_fg",
    "",
    f"- **confidence-only AUPRC:** {auprc_c:.6f}",
    f"- **confidence+consistency AUPRC:** {auprc_p:.6f}",
    f"- **AUPRC difference:** {da:+.6f}",
    f"- **confidence-only Capture@20%:** {cap_c:.6f}",
    f"- **confidence+consistency Capture@20%:** {cap_p:.6f}",
    f"- **Capture@20% difference:** {dc:+.6f}",
    f"- **n_pos:** {int(prop.get('n_pos', conf.get('n_pos', -1)))}",
    "",
    f"- analysis root: `{analysis}`",
    f"- triage dir: `{triage_dir}`",
    "",
    f"**{verdict}**",
    "",
]
(analysis / "seed_analysis_summary.md").write_text("\n".join(lines))
print("Wrote", analysis / "seed_analysis_summary.md")
print(verdict)
PY

echo "=== DONE ==="
