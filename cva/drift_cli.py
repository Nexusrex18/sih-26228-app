"""Offline Module D entrypoint: python -m cva.drift_cli --help."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cva.detectors.drift.registry  # noqa: F401 — entrypoint registration
from cva.core.drift import DriftConfig
from cva.core.drift_scan import scan_drift
from cva.loaders.drift import load_image_batch
from cva.report.render_html import render
from cva.report.report_json import build


def run(incoming, reference=None, out='artifacts/drift', config=None,
        reference_embeddings=None, incoming_embeddings=None):
    config = config or DriftConfig()
    if reference_embeddings and not reference:
        raise ValueError('--reference-embeddings requires --reference')
    inc = load_image_batch(incoming,incoming_embeddings)
    ref = load_image_batch(reference,reference_embeddings) if reference else None
    result = scan_drift(ref,inc,config)
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    report = build(result)
    report['dataset'] = report.pop('model')
    report['access_assumptions']['DATASET_IMAGES'] = {'available': True}
    report['reference'] = {'id':ref.batch_id,'n':len(ref.sample_ids)} if ref else None
    report['incoming_n'] = len(inc.sample_ids)
    report['config'] = asdict(config)
    report['coverage'] = {
        'assessed_checks':[r.check_id for r in result.plan if r.resolution.runnable],
        'not_assessed':{r.check_id:r.resolution.reason for r in result.plan if not r.resolution.runnable},
        'limitations':['MMD, semantic interpretation and intent classification remain planned.',
                       'Synthetic tests do not establish field accuracy; confidence is not yet calibrated.']}
    (out/'drift.report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    render(result,out/'drift.report.html','CV Assurance — Module D — Distribution Drift')
    (out/'drift.coverage.md').write_text('# Module D coverage\n\n'+ '\n'.join(
        f'- {r.check_id}: {r.resolution.state.value} — {r.resolution.reason}' for r in result.plan)
        +'\n\nMMD not implemented. No field accuracy or calibrated attack probability claimed.\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--incoming',required=True,help='Incoming image directory')
    parser.add_argument('--reference',help='Explicit reference image directory')
    parser.add_argument('--out',default='artifacts/drift')
    parser.add_argument('--reference-embeddings')
    parser.add_argument('--incoming-embeddings')
    parser.add_argument('--alpha',type=float,default=.05)
    parser.add_argument('--bins',type=int,default=10)
    parser.add_argument('--seed',type=int,default=0)
    args = parser.parse_args()
    try:
        result = run(args.incoming,args.reference,args.out,
                     DriftConfig(alpha=args.alpha,bins=args.bins,seed=args.seed),
                     args.reference_embeddings,args.incoming_embeddings)
    except ValueError as exc:
        parser.error(str(exc))
    print(f'{result.verdict}: {Path(args.out)/"drift.report.html"}')


if __name__ == '__main__':
    main()
