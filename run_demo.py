"""Single entry point: independent GT diagnosis, prediction, then post-run evaluation."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['diagnose', 'detect', 'run', 'render', 'evaluate', 'identity', 'trace', 'play', 'replay', 'images', 'all'])
    parser.add_argument('--config', default='configs/c1_baseline.json')
    parser.add_argument('--output', help='New output directory for demo replay/images; existing experiment evidence is protected')
    parser.add_argument('--inspect', action='store_true', help='Validate playback metadata without launching an interactive player')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = json.loads((root / args.config).read_text(encoding='utf-8'))
    if config.get('experiment')=='E006' and args.mode not in ('run','evaluate','identity'):
        parser.error('E006 seen screening is CPU cached run/evaluate/identity only; no automatic detect/all/heldout execution')

    if args.mode in ('play','replay','images'):
        import time
        entry_start=time.perf_counter()
        from tracking_demo.demo_delivery import execute
        execute(root,config,args.mode,args.output,args.inspect,entry_start)
        return
    if args.mode=='identity':
        from tracking_demo.identity_evaluation import evaluate_identities
        evaluate_identities(root,config)
        return
    if args.mode=='trace':
        from tracking_demo.association_diagnostics import trace_baselines
        trace_baselines(root,config)
        return
    if config.get('experiment') in ('E005','E006'):
        if args.mode=='run':
            from tracking_demo.temporal_comparison import replay
            replay(root,config)
        elif args.mode=='evaluate':
            if config['experiment']=='E006':
                from tracking_demo.soft_assignment_analysis import evaluate
            else:
                from tracking_demo.temporal_analysis import evaluate
            evaluate(root,config)
        else:parser.error('E005 supports cached run/evaluate and independent identity; no new GPU detection')
        return
    if config.get('experiment')=='E004':
        if args.mode=='detect':
            from tracking_demo.multicam import detect_views
            detect_views(root,config)
        elif args.mode=='run':
            from tracking_demo.improvement import replay
            replay(root,config)
        elif args.mode=='evaluate':
            from tracking_demo.analysis import compare_multicam
            compare_multicam(root,config)
        else:parser.error('E004 prediction config supports detect/run/evaluate; identity uses the segment evaluation config')
        return
    if config.get('experiment') == 'E003':
        from tracking_demo.improvement import execute
        execute(root,config,args.mode)
        return
    if config.get('experiment') == 'E002':
        from tracking_demo.multicam import diagnose_views, detect_views, fuse_and_track, render_cached
        if args.mode in ('diagnose','all'): diagnose_views(root,config)
        if args.mode in ('detect','all'): detect_views(root,config)
        if args.mode in ('run','all'): fuse_and_track(root,config)
        if args.mode == 'render': render_cached(root,config)
        if args.mode in ('evaluate','all'):
            from tracking_demo.analysis import compare_multicam
            compare_multicam(root,config)
        return
    if args.mode in ('detect','render'):
        parser.error('detect/render modes require the E002 multicamera configuration')
    if args.mode in ('diagnose', 'all'):
        from tracking_demo.analysis import diagnose
        diagnose(root, config)
    if args.mode in ('run', 'all'):
        from tracking_demo.pipeline import predict
        predict(root, config, root / args.config)
    if args.mode in ('evaluate', 'all'):
        from tracking_demo.analysis import evaluate
        evaluate(root, config)


if __name__ == '__main__':
    main()
