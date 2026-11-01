"""Build original V6 initialization statistics in the isolated earlier prefix."""
import os,time,json,argparse
from pathlib import Path
BASE=Path(__file__).resolve().parent;OUT=BASE/'artifacts'/'v54_horizon'
os.environ['CTR_ARTIFACT_DIR']=str(OUT)
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
from threadpoolctl import threadpool_limits
from core_v6 import preference_stats,start,DAY,ROOT
from core_v2 import extra_statistics
import pickle

def main():
    ap=argparse.ArgumentParser();ap.add_argument('cut',type=int,choices=[1,2,3]);args=ap.parse_args();cut=start+args.cut*DAY;path=ROOT/f'preference_stats_{cut}.pkl'
    if path.exists():print(json.dumps(dict(status='cached',file=str(path))));return
    tic=time.monotonic()
    with threadpool_limits(limits=4):
        basepath=ROOT/f'stats_{cut}.pkl'
        if not basepath.exists():
            base=extra_statistics(cut)
            with basepath.open('xb') as f:pickle.dump(base,f,protocol=5)
            del base
        stats=preference_stats(cut)
    with path.open('xb') as f:pickle.dump(stats,f,protocol=5)
    elapsed=time.monotonic()-tic;print(json.dumps(dict(status='complete',cut=args.cut,seconds=elapsed)),flush=True)
    (OUT/f'prefix_stats_cut{args.cut}_timing.json').write_text(json.dumps(dict(cut=args.cut,seconds=elapsed,training_max_timestamp=cut-1,original_recipe=True),indent=2))
if __name__=='__main__':main()
