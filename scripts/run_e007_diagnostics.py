"""E007 mechanical analysis or image rendering, isolated from prediction entry points."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tracking_demo.observation_diagnostics import analyze

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['analyze','render','audit'])
    parser.add_argument('--config',default='configs/e007_observation_diagnostics.json')
    args=parser.parse_args();config=json.loads((ROOT/args.config).read_text())
    if args.mode=='analyze':
        out=ROOT/config['output']
        if (out/'mechanical_and_sample_freeze.json').exists():raise FileExistsError('Frozen analysis exists; preserve it before any documented repair.')
        _,cases=analyze(ROOT,config);print(json.dumps({'windows_frozen':len(cases),'new_tracking_ticks':0}))
    elif args.mode=='render':
        from tracking_demo.observation_diagnostics import render_cases
        render_cases(ROOT,config)
    else:
        from tracking_demo.observation_diagnostics import audit_delivery
        audit_delivery(ROOT,config)

if __name__=='__main__':main()
