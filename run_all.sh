#!/bin/bash
set -e
cd "$(dirname "$0")"
for i in $(seq 1 200); do
  [ -f artifacts/corpus/manifest.json ] && break
  grep -q "fitness gate failed" /private/tmp/claude-501/-Users-aneesh-Documents-code/027a3cfc-aa64-4dd8-abce-73edb9f308c1/tasks/bp2ra83cc.output 2>/dev/null && { echo "GATE FAILED"; tail -6 /private/tmp/claude-501/-Users-aneesh-Documents-code/027a3cfc-aa64-4dd8-abce-73edb9f308c1/tasks/bp2ra83cc.output; exit 1; }
  sleep 15
done
echo "=== CORPUS ==="
.venv/bin/python -c "
import json;m=json.load(open('artifacts/corpus/manifest.json'))
for e in m['models']:
    a=f\" ASR={e['asr']:.3f}\" if e.get('asr') is not None else ''
    print(f\"{e['id']:20} acc={e['clean_acc']:.3f}{a}\")"
echo
echo "=== SIGNAL SEPARATION (clean vs backdoored, ground truth) ==="
.venv/bin/python -W ignore - <<'PY'
import sys, json; sys.path.insert(0,'.')
from pathlib import Path
import numpy as np
from cva.adapters.models import load_model
from cva.attacklab.arch import ARCH_REGISTRY
from cva.detectors.base import CheckContext
from cva.core.model import ModelBattery
from cva.detectors.model.universal_margin import UniversalMarginCheck
from cva.detectors.model.neural_cleanse import NeuralCleanseCheck
from cva.detectors.model.intrinsic import IntrinsicProbeCheck
C=Path('artifacts/corpus'); man=json.loads((C/'manifest.json').read_text())
x=np.load(C/'probe_x.npy')[:250]; y=np.load(C/'probe_y.npy')[:250]
ctx=CheckContext(x,y,None,ModelBattery(),{'nc_steps':300,'um_steps':50},'d',None,1)
print(f"{'model':18} {'truth':11} {'univ_margin':>12} {'neural_cl':>11} {'intrinsic':>10}  target/flagged")
for e in man['models']:
    if 'pt' not in e: continue
    h=load_model(C/e['pt'], ARCH_REGISTRY, e['id'])
    um=UniversalMarginCheck().check(h,ctx)[0]
    nc=NeuralCleanseCheck().check(h,ctx)[0]
    it=IntrinsicProbeCheck().check(h,ctx)[0]
    truth='backdoored' if e['backdoored'] else 'clean'
    tgt=e.get('target')
    print(f"{e['id']:18} {truth:11} {um.score_raw:12.2f} {nc.score_raw:11.2f} {it.score_raw:10.2f}  true={tgt}")
PY
