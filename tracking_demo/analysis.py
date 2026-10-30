"""GT-only calibration diagnostics and post-run localization evaluation."""
#空间评价，计算Precision，Recall，以及配对成功的位置相差的距离
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .geometry import GroundCamera, grid_to_world


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution(values):
    if not values:
        return {'count': 0, 'mean': None, 'median': None, 'p90': None, 'p95': None, 'max': None}
    a = np.asarray(values)
    return {'count': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p90': float(np.quantile(a, .9)), 'p95': float(np.quantile(a, .95)), 'max': float(a.max())}


def visible_gt(root, frame, camera, view_number=0):
    # This function is only called by diagnose/evaluate, never by prediction.
    records = json.loads((root / 'annotations_positions' / f'{frame:08d}.json').read_text(encoding='utf-8'))
    result = []
    for index, record in enumerate(records):
        box = next(v for v in record['views'] if v['viewNum'] == view_number)
        x1, y1, x2, y2 = [box[k] for k in ('xmin', 'ymin', 'xmax', 'ymax')]
        xy = grid_to_world(record['positionID'])
        if min(x1, y1, x2, y2) < 0 or x2 <= x1 or y2 <= y1 or x1 >= 1920 or y1 >= 1080:
            continue
        if not camera.in_roi(xy):
            continue
        result.append({'gt_index': index, 'x_m': float(xy[0]), 'y_m': float(xy[1]),
                       'box': [x1, y1, x2, y2],
                       'truncated': x1 <= 0 or y1 <= 0 or x2 >= 1919 or y2 >= 1079})
    return result


def diagnose(root, config):
    data = root / config['data_root']
    output = root / config['output'] / 'geometry_diagnostic'
    output.mkdir(parents=True, exist_ok=True)
    camera = GroundCamera(data, config['camera_calibration'], config['roi_m'])
    rows, per_frame = [], []
    camera_name = config.get('camera', 'C1')
    view_number = int(camera_name[1:])-1
    for frame in config['diagnostic_frames']:
        image = cv2.imread(str(data / f'Image_subsets/{camera_name}' / f'{frame:08d}.png'))
        if image is None:
            raise FileNotFoundError(frame)
        points = visible_gt(data, frame, camera, view_number)
        errors = []
        for gt in points:
            xy = np.array([gt['x_m'], gt['y_m']])
            uv = camera.to_image(xy)
            uv_cv = cv2.projectPoints(np.array([[*xy, 0.]]), camera.rvec, camera.t, camera.K,
                                      np.zeros(5))[0].reshape(2)
            wrong_units = cv2.projectPoints(np.array([[*xy, 0.]]), camera.rvec, camera.t_cm,
                                           camera.K, np.zeros(5))[0].reshape(2)
            x1, y1, x2, y2 = gt['box']
            bottom = np.array([(x1+x2)/2, y2])
            error = float(np.linalg.norm(uv - bottom))
            errors.append(error)
            rows.append({'frame': frame, 'gt_index': gt['gt_index'], 'x_m': xy[0], 'y_m': xy[1],
                         'projected_u': uv[0], 'projected_v': uv[1], 'bbox_bottom_u': bottom[0],
                         'bbox_bottom_v': bottom[1], 'difference_px': error,
                         'opencv_difference_px': float(np.linalg.norm(uv-uv_cv)),
                         'wrong_cm_as_m_difference_px': float(np.linalg.norm(wrong_units-bottom)),
                         'camera_depth_m': camera.depth(xy), 'truncated': gt['truncated']})
            cv2.rectangle(image, (x1, y1), (min(x2, 1919), min(y2, 1079)), (170, 170, 170), 1)
            a, b = tuple(np.rint(uv).astype(int)), tuple(np.rint(bottom).astype(int))
            cv2.line(image, a, b, (255, 160, 0), 1)
            cv2.circle(image, a, 5, (0, 255, 0), 2)
            cv2.drawMarker(image, b, (0, 0, 255), cv2.MARKER_CROSS, 8, 1)
        cv2.rectangle(image, (0, 0), (1920, 55), (20, 20, 20), -1)
        cv2.putText(image, f'GT CALIBRATION DIAGNOSTIC {camera_name} frame {frame:08d} - NOT MODEL PREDICTIONS',
                    (15, 22), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
        cv2.putText(image, 'green: projected GT ground; red cross: GT bbox bottom; gray: GT box',
                    (15, 45), cv2.FONT_HERSHEY_SIMPLEX, .6, (220, 220, 220), 1)
        cv2.imwrite(str(output / f'GT_calibration_{frame:08d}.jpg'), image)
        per_frame.append({'frame': frame, 'visible_gt': len(points), 'pixel_difference': distribution(errors)})
    write_csv(output / 'gt_projection.csv', rows)
    summary = {'calibration': camera.describe(), 'camera_center_m': (-camera.R.T @ camera.t).tolist(),
               'frames': per_frame,
               'max_opencv_difference_px': max(r['opencv_difference_px'] for r in rows),
               'correct_unit_pixel_difference': distribution([r['difference_px'] for r in rows]),
               'wrong_cm_as_m_pixel_difference': distribution([r['wrong_cm_as_m_difference_px'] for r in rows]),
               'scope': 'GT-to-image sanity check, not calibration accuracy or detector localization performance; no GT fit or parameter update.',
               'sources': ['data/Wildtrack_dataset/README.txt',
                           'https://github.com/hou-yz/MVDet/blob/master/multiview_detector/datasets/Wildtrack.py'],
               'conflict': 'Old toolkit (-300,-90) is inconsistent with README (-3,-9)m and MVDet (-300,-900)cm; use documented grid with parentheses and integer division.'}
    (output / 'diagnostic.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'geometry_diagnostic': per_frame, 'camera_center_m': summary['camera_center_m']}, indent=2))
    return summary


def compare_multicam(root, config):
    """E002 post-run evaluation: identical all-GT denominator and calibration-only mask."""
    from .multicam import cameras_for
    output=root/config['output']; cameras=cameras_for(root,config)
    frames=list(range(config['frame_start'],config['frame_stop'],config['frame_step']))
    threshold=config['evaluation']['distance_threshold_m']
    def common(xy):
        return all(camera.sees_ground(xy) for camera in cameras.values())
    conditions={}
    for variant in config['variants']:
        local_sources=config.get('local_comparison_sources',False)
        folder=root/config['e001_output'] if variant=='C1' and not local_sources else output/variant
        filename='observations.csv' if variant=='C1' and not local_sources else 'fused_observations.csv'
        with (folder/filename).open(encoding='utf-8') as handle:
            raw=[r for r in csv.DictReader(handle) if r.get('reason','accepted')=='accepted']
        with (folder/'tracks.csv').open(encoding='utf-8') as handle:
            tracks=list(csv.DictReader(handle))

        #读取结果后，将它们分成三类
        conditions[variant]={'raw_observations':raw,
                             'filtered_observed':[r for r in tracks if r['state']=='observed'],
                             'pure_predictions':[r for r in tracks if r['state']=='predicted']}
    grid=[(x,y) for x in np.arange(-3,9,.25) for y in np.arange(-9,27,.25)]
    overlap={'sample_step_m':.25,'full_roi_grid_points':len(grid),
             'C1_C2':sum(cameras['C1'].sees_ground(p) and cameras['C2'].sees_ground(p) for p in grid),
             'C1_C3':sum(cameras['C1'].sees_ground(p) and cameras['C3'].sees_ground(p) for p in grid),
             'C1_C2_C3':sum(common(p) for p in grid),
             'meaning':'Ground-footpoint projection only, not occlusion or true person visibility.'}
    (output/'geometric_overlap.json').write_text(json.dumps(overlap,indent=2),encoding='utf-8')
    frame_rows=[]; match_rows=[]; gt_rows=[]; distance_pools={}; all_gt={}
    for frame in frames:
        #读取标注
        records=json.loads((root/config['data_root']/'annotations_positions'/f'{frame:08d}.json').read_text(encoding='utf-8'))
        gt=[]
        for index,record in enumerate(records):
            xy=grid_to_world(record['positionID']) #将网格位置编号转换为米制XY
            if cameras['C1'].in_roi(xy): gt.append({'gt_index':index,'xy':xy})
        all_gt[frame]=gt
        for region in config['evaluation']['regions']:
            valid_gt=[g for g in gt if region=='full_roi' or common(g['xy'])]
            gt_rows.append({'frame':frame,'region':region,'gt_count':len(valid_gt)})
            for variant,pools in conditions.items():
                for pool,rows in pools.items():
                    current=[r for r in rows if int(r['frame'])==frame]
                    valid=[r for r in current if cameras['C1'].in_roi([float(r['x_m']),float(r['y_m'])])
                           and (region=='full_roi' or common([float(r['x_m']),float(r['y_m'])]))]

                    pairs=gated_pairs(
                        [[float(r['x_m']),float(r['y_m'])] for r in valid],
                        [g['xy'] for g in valid_gt],
                        threshold
                    )

                    key=(region,variant,pool); distance_pools.setdefault(key,[]).extend(d for _,_,d in pairs)
                    frame_rows.append({'frame':frame,'region':region,'variant':variant,'pool':pool,
                                       'gt_count':len(valid_gt),'prediction_count':len(valid),'excluded_by_region':len(current)-len(valid),
                                       'matched':len(pairs),'unmatched_predictions':len(valid)-len(pairs),'unmatched_gt':len(valid_gt)-len(pairs)})
                    for i,j,distance in pairs:
                        match_rows.append({'frame':frame,'region':region,'variant':variant,'pool':pool,
                                           'prediction_key':valid[i].get('fusion_id') or valid[i].get('obs_id') or valid[i].get('track_id'),
                                           'gt_index':valid_gt[j]['gt_index'],'distance_m':distance})
    totals={}; comparison=[]
    for (region,variant,pool),distances in distance_pools.items():
        rows=[r for r in frame_rows if (r['region'],r['variant'],r['pool'])==(region,variant,pool)]
        counts={key:sum(r[key] for r in rows) for key in ('gt_count','prediction_count','excluded_by_region','matched','unmatched_predictions','unmatched_gt')}

        stats=distribution(distances)

        counts['matched_distance_m']=stats
        counts['matched_gt_fraction']=counts['matched']/counts['gt_count'] if counts['gt_count'] else None
        totals.setdefault(region,{}).setdefault(variant,{})[pool]=counts
        comparison.append({'region':region,'variant':variant,'pool':pool,
                           **{k:v for k,v in counts.items() if k!='matched_distance_m'},
                           **{f'distance_{k}_m':stats[k] for k in ('mean','median','p90','p95','max')}})
    for region,variants in totals.items():
        assert len({pool['gt_count'] for pools in variants.values() for pool in pools.values()})==1

    #compare_multicam() 对每个时刻重复上述过程，保存以下内容
    write_csv(output/'comparison.csv',comparison) #整段汇总结果
    write_csv(output/'evaluation_per_frame.csv',frame_rows) #每时刻的数量统计
    write_csv(output/'evaluation_matches.csv',match_rows) #每个成功配对及距离
    write_csv(output/'gt_counts.csv',gt_rows) #各时刻、各区域的GT数量

    summary={'frames':len(frames),'frame_start':frames[0],'frame_last':frames[-1],
             'roi_m':config['roi_m'],'protocol':config['evaluation'],'totals':totals,
             'matching':'Same 1m in-assignment gate, maximum valid cardinality then minimum distance; every region/condition/pool independently evaluated.',
             'E001_reassessment':('Development uses E001/E002 image-observation caches, with empty trackers per condition.' if config.get('segment')=='development' else 'Validation C1/C2/C3 use fresh shared validation detections, with empty trackers per condition.') if config.get('local_comparison_sources') else 'C1 cached predictions re-evaluated here under all-GT/common-mask protocols. Original E001 C1-visible evaluation stays unchanged.',
             'identity_scope':'All counts are person-frame or prediction-state counts, not unique people or identity accuracy. No IDF1/HOTA.',
             'pure_prediction_scope':'Do not add independent pure-prediction matches to observed matches.'}
    if config.get('prediction_provenance'):
        summary['E001_reassessment']=config['prediction_provenance']
    (output/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # Fixed-time evaluation-only BEV panels; same GT and common mask in every panel.
    folder=output/'comparisons'; folder.mkdir(exist_ok=True)
    if config.get('experiment') in ('E003','E004','E005','E006'):
        print(json.dumps({'observed_comparison':[r for r in comparison if r['pool']=='filtered_observed']},indent=2))
        return summary
    for frame in (0,250):
        if frame not in frames: continue
        canvas=np.full((820,1200,3),24,np.uint8)
        cv2.putText(canvas,f'E002 EVAL ONLY frame {frame:08d}; gray=all GT; blue=common geometry; IDs independent per condition',
                    (10,25),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1)
        for col,(variant,pools) in enumerate(conditions.items()):
            def pixel(xy): return round(col*400+60+(xy[0]+3)*16.67),round(755-(xy[1]+9)*16.67)
            for xy in grid:
                if common(xy): cv2.circle(canvas,pixel(xy),1,(70,45,25),-1)
            for gt in all_gt[frame]: cv2.circle(canvas,pixel(gt['xy']),3,(160,160,160),1)
            for pool in ('filtered_observed','pure_predictions'):
                for row in pools[pool]:
                    xy=[float(row['x_m']),float(row['y_m'])]
                    if int(row['frame'])==frame and cameras['C1'].in_roi(xy):
                        color=(0,220,140) if pool=='filtered_observed' else (0,170,255)
                        cv2.circle(canvas,pixel(xy),3,color,-1 if pool=='filtered_observed' else 1)
            item=next(r for r in frame_rows if r['frame']==frame and r['region']=='full_roi' and r['variant']==variant and r['pool']=='filtered_observed')
            cv2.putText(canvas,variant,(col*400+30,60),cv2.FONT_HERSHEY_SIMPLEX,.7,(240,240,240),1)
            cv2.putText(canvas,f'GT {item["gt_count"]}, match {item["matched"]}, unGT {item["unmatched_gt"]}',
                        (col*400+15,87),cv2.FONT_HERSHEY_SIMPLEX,.43,(220,220,220),1)
        cv2.putText(canvas,'green filled: observed-filtered; orange hollow: forecast-only; world X right, Y up; no ground-truth identity score',
                    (10,803),cv2.FONT_HERSHEY_SIMPLEX,.48,(220,220,220),1)
        cv2.imwrite(str(folder/f'BEV_comparison_{frame:08d}.jpg'),canvas)
    print(json.dumps({'overlap':overlap,'observed_comparison':[r for r in comparison if r['pool']=='filtered_observed']},indent=2))
    return summary


def gated_pairs(predictions, ground_truth, threshold):
    """Maximum-cardinality one-to-one matching, then minimum distance within the gate."""
    if not predictions or not ground_truth:
        return []

    #计算每个预测点与每个GT点之间的距离
    distance = cdist(np.asarray(predictions), np.asarray(ground_truth))
    n, m = distance.shape

    unmatched = (n + m + 1) * (threshold + 1)
    #与融合分配结构很相似，门限内使用实际距离，门限外设置很大代价，并增加虚拟列，允许预测点不匹配
    cost = np.column_stack((np.where(distance <= threshold, distance, unmatched * 10),
                            np.full((n, n), unmatched)))

    #每个预测最多对应一个GT，每个GT最多对应一个预测；先争取更多门限内配对，再使总距离较小。
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(distance[i, j])) for i, j in zip(rows, cols) if j < m]


def evaluate(root, config):
    output = root / config['output']
    data = root / config['data_root']
    camera = GroundCamera(data, config['camera_calibration'], config['roi_m'])
    with (output / 'observations.csv').open(encoding='utf-8') as handle:
        observations = list(csv.DictReader(handle))
    with (output / 'tracks.csv').open(encoding='utf-8') as handle:
        tracks = list(csv.DictReader(handle))
    threshold = config['evaluation']['distance_threshold_m']
    pools = {'raw_observations': [], 'filtered_observed': [], 'pure_predictions': []}
    frame_rows, pair_rows, failures = [], [], []
    for frame in range(config['frame_start'], config['frame_stop'], config['frame_step']):
        gt = visible_gt(data, frame, camera)
        obs = [r for r in observations if int(r['frame']) == frame and r['reason'] == 'accepted']
        states = [r for r in tracks if int(r['frame']) == frame]
        candidates = {'raw_observations': obs,
                      'filtered_observed': [r for r in states if r['state'] == 'observed'],
                      'pure_predictions': [r for r in states if r['state'] == 'predicted']}
        for pool, rows in candidates.items():
            valid = [r for r in rows if camera.in_roi([float(r['x_m']), float(r['y_m'])])]
            pairs = gated_pairs([[float(r['x_m']), float(r['y_m'])] for r in valid],
                                [[r['x_m'], r['y_m']] for r in gt], threshold)
            pools[pool].extend(distance for _, _, distance in pairs)
            item = {'frame': frame, 'pool': pool, 'gt_count': len(gt), 'prediction_count': len(valid),
                    'excluded_outside_roi': len(rows)-len(valid), 'matched': len(pairs),
                    'unmatched_predictions': len(valid)-len(pairs), 'unmatched_gt': len(gt)-len(pairs),
                    'truncated_gt_count': sum(g['truncated'] for g in gt)}
            frame_rows.append(item)
            for i, j, distance in pairs:
                pair_rows.append({'frame': frame, 'pool': pool, 'prediction_key': valid[i].get('obs_id') or valid[i].get('track_id'),
                                  'gt_index': gt[j]['gt_index'], 'distance_m': distance})
            if pool == 'raw_observations':
                failures.append((item['unmatched_gt'] + item['unmatched_predictions'], frame, item,
                                 {j for _, j, _ in pairs}))
    write_csv(output / 'localization_per_frame.csv', frame_rows)
    write_csv(output / 'localization_matches.csv', pair_rows)
    totals = {}
    for pool, distances in pools.items():
        rows = [r for r in frame_rows if r['pool'] == pool]
        aggregate = {key: sum(r[key] for r in rows) for key in
                     ('gt_count', 'prediction_count', 'excluded_outside_roi', 'matched',
                      'unmatched_predictions', 'unmatched_gt', 'truncated_gt_count')}
        aggregate['matched_distance_m'] = distribution(distances)
        aggregate['matched_prediction_fraction'] = aggregate['matched'] / aggregate['prediction_count'] if aggregate['prediction_count'] else None
        aggregate['matched_gt_fraction'] = aggregate['matched'] / aggregate['gt_count'] if aggregate['gt_count'] else None
        totals[pool] = aggregate
    failure_dir = output / 'failure_cases'
    failure_dir.mkdir(exist_ok=True)
    cases = []
    for _, frame, item, matched_gt in sorted(failures, key=lambda r: (-r[0], r[1]))[:3]:
        image = cv2.imread(str(data / 'Image_subsets/C1' / f'{frame:08d}.png'))
        gt = visible_gt(data, frame, camera)
        for index, g in enumerate(gt):
            x1, y1, x2, y2 = g['box']
            color = (0, 220, 0) if index in matched_gt else (0, 180, 255)
            cv2.rectangle(image, (x1, y1), (min(x2, 1919), min(y2, 1079)), color, 2)
            uv = camera.to_image([g['x_m'], g['y_m']])
            cv2.circle(image, tuple(np.rint(uv).astype(int)), 4, color, -1)
        for row in observations:
            if int(row['frame']) == frame:
                cv2.drawMarker(image, (round(float(row['foot_u'])), round(float(row['foot_v']))),
                               (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
        cv2.rectangle(image, (0, 0), (1920, 55), (20, 20, 20), -1)
        cv2.putText(image, f'EVALUATION ONLY C1 {frame:08d}: GT matched green / unmatched orange; detection feet red',
                    (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
        cv2.putText(image, f'GT={item["gt_count"]} matched={item["matched"]} unmatched predictions={item["unmatched_predictions"]} unmatched GT={item["unmatched_gt"]}',
                    (12, 49), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        path = failure_dir / f'EVAL_{frame:08d}.jpg'
        cv2.imwrite(str(path), image)
        cases.append({**item, 'image': str(path.relative_to(root))})
    summary = {'protocol': config['evaluation'], 'roi_m': config['roi_m'],
               'frames': len(range(config['frame_start'], config['frame_stop'], config['frame_step'])),
               'totals': totals, 'failure_cases': cases,
               'scope': 'Development-subset ground-point localization diagnostic, not official benchmark or identity accuracy.',
               'matching': 'Distance gate is inside assignment; maximize valid cardinality then minimize distance. Each pool evaluated independently.',
               'pure_prediction_note': 'Independent diagnostic pool; do not add its matches to observed-pool matches.',
               'gt_identity': 'No IDF1/HOTA; GT identity semantics not established.'}
    (output / 'evaluation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'evaluation': totals, 'failure_frames': [c['frame'] for c in cases]}, indent=2))
    return summary
