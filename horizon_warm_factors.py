"""Original warm16 factor recipe with a causal early-calendar initialization."""
import os,time,json,argparse
from pathlib import Path
BASE=Path(__file__).resolve().parent;OUT=BASE/'artifacts'/'v54_horizon'
os.environ['CTR_ARTIFACT_DIR']=str(OUT);os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import imp3_warmfeat as original
from imp2_warm_choice import init32 as original_init
from threadpoolctl import threadpool_limits

def main():
    p=argparse.ArgumentParser();p.add_argument('cut',type=int,choices=[1,2,3]);p.add_argument('--seed',type=int,default=151);args=p.parse_args();tic=time.monotonic()
    # Original max(cut-2,2) would look ahead for cut1. Clamp to fitted prefix.
    initcut=min(args.cut,2)
    original.init32=lambda ignored,dim:original_init(original.cutoff_of(initcut),dim)
    with threadpool_limits(limits=4):original.fit_cut(args.cut,lambda line:print(line,flush=True),args.seed)
    result=dict(cut=args.cut,seed=args.seed,initialization_cut=initcut,epochs=10,dimension=16,learning_rate=.045,regularization=.002,seconds=time.monotonic()-tic,recipe_adjustment='Initialization prefix clamped to fit cutoff; all remaining original warm16 settings unchanged')
    (OUT/f'warm16_cut{args.cut}_seed{args.seed}_timing.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
