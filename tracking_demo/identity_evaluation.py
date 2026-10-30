"""Offline ground-point identity evaluation. Never imported by prediction modules."""
#检查同一个真实人物，在一段时间里是否持续对应同一条预测轨迹。
import csv
from datetime import datetime
import json
import math
import os
import sys
import time

import motmetrics as mm
import numpy as np
from scipy.spatial.distance import cdist

from .geometry import grid_to_world


METRICS=['idf1','idp','idr','idtp','idfp','idfn','num_objects','num_predictions',
         'num_unique_objects','num_switches','num_fragmentations','num_frames']


def in_roi(xy,roi):
    return roi[0]<=xy[0]<roi[1] and roi[2]<=xy[1]<roi[3]

#确定哪些身份在当前时刻有空间对应的可能
def distance_matrix(gt,pred,gate=1.):
    """Full GT-by-prediction Euclidean distances in meters, <= gate inclusive."""
    a=np.array([[r['x_m'],r['y_m']] for r in gt],float).reshape(-1,2)
    b=np.array([[r['x_m'],r['y_m']] for r in pred],float).reshape(-1,2)
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite point')
    d=cdist(a,b)
    return np.where(d<=gate,d,np.nan)

#按时间积累关系，再计算整段指标
def accumulate(frames,gate=1.):
    mm.lap.default_solver='scipy'
    acc=mm.MOTAccumulator(auto_id=False,max_switch_time=np.inf)
    last=None
    for frame,gt,pred in frames:
        if last is not None and frame<=last:raise ValueError('Frames must strictly increase')
        for rows in (gt,pred):
            if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate per-frame identity')
        acc.update([r['id'] for r in gt],[r['id'] for r in pred],distance_matrix(gt,pred,gate),frameid=frame)
        last=frame
    return acc


def metrics_for(acc):
    #计算IDTP、IDFP、IDFN、IDF1等指标
    values=mm.metrics.create().compute(acc,metrics=METRICS,return_dataframe=False)
    return {k:float(v) if k in ('idf1','idp','idr') else int(v) for k,v in values.items()}


def load_gt(data,frame,roi):
    # Missing files raise: absence of a file is never an empty GT frame.
    records=json.loads((data/'annotations_positions'/f'{frame:08d}.json').read_text(encoding='utf-8'))
    result=[]
    for record in records:
        xy=grid_to_world(record['positionID'])
        # 每条GT包含真实人物编号＋当前地面位置
        if in_roi(xy,roi):result.append({'id':int(record['personID']),'x_m':float(xy[0]),'y_m':float(xy[1])})
    return result


def read_csv(path):
    with path.open(encoding='utf-8') as handle:return list(csv.DictReader(handle))


def save_csv(path,rows,fields=None):
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields or list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def identity_mapping(acc):
    host=mm.metrics.create();assignment=host.compute(acc,metrics=['id_global_assignment'],return_dataframe=False)['id_global_assignment']
    counts=mm.metrics.extract_counts_from_df_map(mm.metrics.events_to_df_map(acc.events))
    ocs,hcs,tps=counts;oids=sorted(ocs);hids=sorted(hcs);result=[]
    for i,j in zip(assignment['rids'],assignment['cids']):
        if i<len(oids) or j<len(hids):
            oid=oids[i] if i<len(oids) else None;hid=hids[j] if j<len(hids) else None
            result.append({'personID':int(oid) if oid is not None else '',
                           'track_id':int(hid) if hid is not None else '',
                           'GT_states':int(ocs.get(oid,0)),'prediction_states':int(hcs.get(hid,0)),
                           'valid_cooccurrences':int(tps.get((oid,hid),0)),
                           'IDFN':int(assignment['fnmatrix'][i,j]),'IDFP':int(assignment['fpmatrix'][i,j])})
    return result


def collect_spatial_tables(root,config):
    """Collect unchanged spatial-protocol tables; never derive them from ID events."""
    rows=[]
    for segment,sc in config['segments'].items():
        source=(root/sc['spatial_source']) if sc.get('spatial_source') else (root/next(iter(sc['track_sources'].values()))).parent.parent/'comparison.csv'
        for row in read_csv(source):
            if row['variant'] in sc['track_sources']:
                predictions=int(row['prediction_count']);gt=int(row['gt_count']);matched=int(row['matched'])
                rows.append({'segment':segment,**row,'spatial_precision':matched/predictions if predictions else '',
                             'spatial_recall':matched/gt if gt else '', 'source':source.relative_to(root).as_posix()})
    save_csv(root/config['output']/'spatial_summary.csv',rows)
    return rows


def evaluate_identities(root,config):
    output=root/config['output'];output.mkdir(parents=True,exist_ok=True)
    began=time.perf_counter();started=datetime.now().astimezone().isoformat();all_summary=[];pure_counts=[]
    for segment,sc in config['segments'].items():
        frame_ids=list(range(sc['frame_start'],sc['frame_stop'],sc['frame_step']))
        #读取GT
        gt={f:load_gt(root/config['data_root'],f,config['roi_m']) for f in frame_ids}

        for variant,source in sc['track_sources'].items():
            folder=output/segment/variant;folder.mkdir(parents=True,exist_ok=True)
            rows=read_csv(root/source);frames=[];frame_counts=[];excluded=0
            for f in frame_ids:
                current=[r for r in rows if int(r['frame'])==f]
                #读取tracks.csv，筛选预测
                observed=[r for r in current if r['state']=='observed']
                valid=[
                    r for r in observed
                    if in_roi(
                        [float(r['x_m']),float(r['y_m'])],
                        config['roi_m']
                    )
                ] #评价使用的是整个ROI内，本时刻获得观测更新的轨迹状态

                pred=[{'id':int(r['track_id']),'x_m':float(r['x_m']),'y_m':float(r['y_m'])} for r in valid]
                frames.append((f,gt[f],pred));excluded+=len(observed)-len(valid)
                frame_counts.append({'frame':f,'time_s':f/config['frame_number_hz'],'GT':len(gt[f]),'observed':len(pred),'observed_outside_roi':len(observed)-len(valid)})
                forecast=[r for r in current if r['state']=='predicted']
                pure_counts.append({'segment':segment,'variant':variant,'frame':f,'time_s':f/10,'pure_predictions':len(forecast),
                                    'pure_predictions_in_roi':sum(in_roi([float(r['x_m']),float(r['y_m'])],config['roi_m']) for r in forecast)})
            #accmulate
            acc=accumulate(frames,config['gate_m'])
            summary=metrics_for(acc)

            summary={'segment':segment,'variant':variant,**summary,'observed_excluded_outside_roi':excluded,'source_tracks':source}
            assert summary['num_objects']==sum(len(g) for _,g,_ in frames)
            assert summary['num_predictions']==sum(len(p) for _,_,p in frames)
            # Keep every pair, including invalid edges, as reviewable input evidence.
            with (folder/'distance_pairs.csv').open('w',newline='',encoding='utf-8') as handle: #每时刻所有GT×预测距离，包括门限外关系
                writer=csv.writer(handle);writer.writerow(['frame','personID','track_id','distance_m','valid_le_1m'])
                for f,g,p in frames:
                    if not g or not p:continue
                    distances=cdist([[r['x_m'],r['y_m']] for r in g],[[r['x_m'],r['y_m']] for r in p])
                    writer.writerows((f,a['id'],b['id'],float(distances[i,j]),bool(distances[i,j]<=config['gate_m'])) for i,a in enumerate(g) for j,b in enumerate(p))
            acc.events.to_csv(folder/'events.csv') #评价库记录的匹配、切换等事件
            save_csv(folder/'frame_counts.csv',frame_counts) #每时刻的GT数和有效观测更新数
            save_csv(folder/'identity_mapping.csv',identity_mapping(acc),['personID','track_id','GT_states','prediction_states','valid_cooccurrences','IDFN','IDFP'])#整段GT身份与预测轨迹的对应及贡献
            (folder/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8') #某片段、某条件的指标
            all_summary.append(summary);print(json.dumps(summary),flush=True)
    save_csv(output/'identity_summary.csv',all_summary) #各片段、各条件的汇总
    save_csv(output/'pure_prediction_counts.csv',pure_counts)
    collect_spatial_tables(root,config)
    summary={'status':'completed','started_at':started,'finished_at':datetime.now().astimezone().isoformat(),'pid':os.getpid(),
             'command_argv':[sys.executable,*sys.argv],'seconds':time.perf_counter()-began,'protocol':config,
             'metric':'Ground-point IDF1@1m; observed only, full ROI, project-specific splits; not official WILDTRACK score',
             'distance':'Full GT x observed Euclidean meters; <=1.0 valid, >1.0 NaN. No spatial winning-pair input.',
             'solver':'scipy','motmetrics_version':mm.__version__,'max_switch_time':'infinity','auto_id':False,
             'events_scope':'Library SWITCH/fragmentation are diagnostics; GT identity anomalies not edited.',
             'rows':all_summary}
    (output/'identity_run.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary
