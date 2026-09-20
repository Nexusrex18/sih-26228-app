"""Offline dataset drift command; invoked through python -m cva.cli drift."""
import argparse
import datetime as dt
import json
import shlex
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

from cva.core.drift_scan import scan_drift
from cva.core.orchestrator import profile_hash_of
from cva.core.scanid import (
    SELFTEST_EPOCH,
    EvidenceStore,
    materialise_evidence,
    new_scan_id,
    selftest_scan_id,
)
from cva.loaders.drift import dataset_id, load_embeddings, load_image_dataset
from cva.loaders.safety import UnsafeArtifact
from cva.report.coverage import write as write_coverage
from cva.report.render_html import render
from cva.report.report_json import build

from .config import DriftConfig, load_profile
from .registry import build_checks


def reserve_output(out, selftest, seed):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    now = SELFTEST_EPOCH if selftest else dt.datetime.now(dt.UTC)
    candidates = [selftest_scan_id(seed)] if selftest else [new_scan_id(now,c) for c in range(10000)]
    for scan_id in candidates:
        path = out/scan_id
        try:
            path.mkdir()  # atomic allocation across concurrent processes; never overwrite
            return scan_id,path,now.isoformat()
        except FileExistsError:
            continue
    raise FileExistsError('Scan namespace exhausted or selftest output already exists; use a new output root')


def run(incoming, reference=None, out='artifacts/drift', config=None,
        reference_embeddings=None, incoming_embeddings=None, *, selftest=False, profile=None, overrides=None):
    effective = load_profile(profile, config=config, overrides=overrides, selftest=selftest)
    config = DriftConfig.from_profile(effective)
    if reference_embeddings and not reference:
        raise ValueError('--reference-embeddings requires --reference')
    inc = load_image_dataset(incoming)
    ref = load_image_dataset(reference) if reference else None
    inc_dist = load_embeddings(incoming_embeddings,inc)
    ref_dist = load_embeddings(reference_embeddings,ref) if ref else None
    scan_id,report_dir,created_at = reserve_output(out,selftest,config.seed)
    try:
        result = scan_drift(ref,inc,checks=build_checks(config),scan_id=scan_id,profile=effective,
                            min_samples=config.min_samples,reference_dist=ref_dist,incoming_dist=inc_dist)
        profile_hash = profile_hash_of(effective)
        try:
            commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[3],
                                            stderr=subprocess.DEVNULL,text=True).strip()
        except (OSError,subprocess.CalledProcessError):
            commit = 'unknown (source distribution)'
        result.created_at_utc = created_at
        result.profile_hash = profile_hash
        result.profile_name = effective['name']
        result.code_commit = commit
        result.seed = config.seed
        result.profile = effective
        result.target = {'dataset_path':str(Path(incoming).resolve()),'dataset_format':'image-folder',
                         'n_samples':len(inc.samples),'n_categories':len(inc.categories)}
        result.drift_summary = {'reference_id':dataset_id(ref) if ref else None,
                                'reference_n':len(ref.samples) if ref else 0,
                                'incoming_id':dataset_id(inc),'config':asdict(config)}
        result.calibration = {'method':'unavailable; confidence 0 is an uncalibrated placeholder','brier':None}
        for f in result.findings:
            f.produced_by = f'{commit}:{profile_hash}'
        if selftest:
            result.timings = dict.fromkeys(result.timings,0.)
        result.report_dir = str(report_dir)
        tokens = ['python','-m','cva.cli','drift','--incoming',str(Path(incoming).resolve())]
        # Store the exact validated profile so reproduction includes policy and axis overrides.
        profile_path = report_dir/'effective.profile.json'
        profile_path.write_text(json.dumps(effective,sort_keys=True,indent=2))
        tokens += ['--profile', 'effective.profile.json']
        if reference:
            tokens += ['--reference',str(Path(reference).resolve())]
        for flag,path in [('--reference-embeddings',reference_embeddings),('--incoming-embeddings',incoming_embeddings)]:
            if path:
                tokens += [flag,str(Path(path).resolve())]
        if selftest:
            tokens.append('--selftest')
        command = shlex.join(tokens)
        materialise_evidence(result.findings,EvidenceStore(Path(out)))
        (report_dir/'drift.report.json').write_text(json.dumps(build(result,command=command),indent=2,allow_nan=False))
        render(result,report_dir/'drift.report.html','CV Assurance — Module D — Distribution Drift',
               evidence_root=Path(out)/'evidence', command=command)
        write_coverage(result,report_dir/'drift.coverage.md')
        return result
    except BaseException:
        # Only this run's atomically reserved directory; shared evidence is immutable
        # and may already be referenced by another scan, so leave that store alone.
        shutil.rmtree(report_dir)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--incoming',required=True,help='Incoming image directory')
    parser.add_argument('--reference',help='Explicit reference image directory')
    parser.add_argument('--out',default='artifacts/drift')
    parser.add_argument('--profile', help='JSON profile; explicit flags override its psi settings')
    cache_help = ('NPZ cache: embeddings finite N x D, sample_ids ordered relative paths, '
                  'extractor_id and extractor_version scalar strings. Extraction is out of scope; '
                  'identity is self-declared, not verified provenance.')
    parser.add_argument('--reference-embeddings', help=cache_help)
    parser.add_argument('--incoming-embeddings', help=cache_help)
    parser.add_argument('--alpha',type=float,default=None)
    parser.add_argument('--bins',type=int,default=None)
    parser.add_argument('--seed',type=int,default=None)
    parser.add_argument('--min-samples',type=int,default=None)
    parser.add_argument('--min-ks-effect',type=float,default=None)
    parser.add_argument('--permutations',type=int,default=None)
    parser.add_argument('--selftest',action='store_true')
    args = parser.parse_args(argv)
    try:
        result = run(args.incoming,args.reference,args.out,
                     reference_embeddings=args.reference_embeddings,incoming_embeddings=args.incoming_embeddings,
                     selftest=args.selftest,profile=args.profile,
                     overrides={key:getattr(args,key) for key in ('alpha','bins','seed','min_samples','min_ks_effect','permutations')})
    except (ValueError,UnsafeArtifact,OSError) as exc:
        parser.error(str(exc))
    assessed = sum(row.resolution.runnable for row in result.plan)
    print(f'{result.verdict} — {assessed}/{len(result.plan)} checks assessed; '
          f'{len(result.plan)-assessed} coverage gaps/errors. {Path(result.report_dir)/"drift.report.html"}')
