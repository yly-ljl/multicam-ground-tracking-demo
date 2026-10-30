"""Camera/BEV prediction display: shared IDs, no GT, no reused stale boxes."""
from collections import defaultdict, deque

import cv2
import numpy as np


def color_for(track_id):
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[[(track_id * 37) % 180, 200, 245]]]),
                                            cv2.COLOR_HSV2BGR)[0, 0])


class Renderer:
    def __init__(self, camera, config):
        self.camera = camera
        self.config = config
        length = round(config['trail_seconds'] / .5)
        self.history = defaultdict(lambda: deque(maxlen=length))
        self.scale = 17.5

    def bev_pixel(self, xy):
        return (round(1340 + (xy[0] + 3) * self.scale),
                round(710 - (xy[1] + 9) * self.scale))

    def draw(self, image, frame, time_s, observations, states):
        camera_image = image.copy()
        for row in observations:
            box = [round(row[k]) for k in ('x1', 'y1', 'x2', 'y2')]
            track_id = row['track_id']
            color = color_for(track_id) if track_id != '' else (140, 140, 140)
            cv2.rectangle(camera_image, tuple(box[:2]), tuple(box[2:]), color, 2)
            label = f'ID {track_id} obs {row["confidence"]:.2f}' if track_id != '' else row['reason']
            cv2.putText(camera_image, label, (box[0], max(18, box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
            cv2.drawMarker(camera_image, (round(row['foot_u']), round(row['foot_v'])),
                           color, cv2.MARKER_CROSS, 10, 2)
        for row in states:
            if row['state'] == 'predicted' and self.camera.depth([row['x_m'], row['y_m']]) > 0:
                uv = self.camera.to_image([row['x_m'], row['y_m']])
                if 0 <= uv[0] < 1920 and 0 <= uv[1] < 1080:
                    point = tuple(np.rint(uv).astype(int))
                    color = color_for(row['track_id'])
                    cv2.circle(camera_image, point, 10, color, 2)
                    cv2.putText(camera_image, f'P ID {row["track_id"]} +{row["since_observation_s"]:.1f}s',
                                (point[0]+10, point[1]), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
        canvas = np.full((self.config['height'], self.config['width'], 3), 24, dtype=np.uint8)
        canvas[40:760, :1280] = cv2.resize(camera_image, (1280, 720))
        cv2.putText(canvas, f'C1  frame {frame:08d}  t={time_s:.1f}s | prediction | source 2 Hz / playback 1x',
                    (15, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, 'Ground XY (m)', (1300, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (240, 240, 240), 1)
        cv2.putText(canvas, 'Z=0 assumed; Y up', (1290, 48), cv2.FONT_HERSHEY_SIMPLEX, .5, (220, 220, 220), 1)
        for x in range(-3, 10, 3):
            a, b = self.bev_pixel((x, -9)), self.bev_pixel((x, 27))
            cv2.line(canvas, a, b, (65, 65, 65), 1)
            cv2.putText(canvas, str(x), (a[0]-7, a[1]+20), cv2.FONT_HERSHEY_SIMPLEX, .4, (200, 200, 200), 1)
        for y in range(-9, 28, 3):
            a, b = self.bev_pixel((-3, y)), self.bev_pixel((9, y))
            cv2.line(canvas, a, b, (65, 65, 65), 1)
            cv2.putText(canvas, str(y), (1308, a[1]+4), cv2.FONT_HERSHEY_SIMPLEX, .4, (200, 200, 200), 1)
        for row in states:
            track_id = row['track_id']
            color = color_for(track_id)
            self.history[track_id].append((row['x_m'], row['y_m']))
            points = [self.bev_pixel(xy) for xy in self.history[track_id] if self.camera.in_roi(xy)]
            if len(points) > 1:
                cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, color, 1)
            if row['in_roi']:
                p = self.bev_pixel((row['x_m'], row['y_m']))
                cv2.circle(canvas, p, 4, color, -1 if row['state'] == 'observed' else 1)
                label = str(track_id) if row['state'] == 'observed' else f'P{track_id}'
                cv2.putText(canvas, label, (p[0]+5, p[1]-4), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
        observed = sum(row['state'] == 'observed' for row in states)
        predicted = len(states) - observed
        rejected = sum(row['reason'] != 'accepted' for row in observations)
        cv2.putText(canvas, f'updated IDs {observed} | forecast-only {predicted} | excluded observations {rejected}',
                    (15, 786), cv2.FONT_HERSHEY_SIMPLEX, .6, (230, 230, 230), 1)
        cv2.putText(canvas, 'solid: updated', (1290, 759), cv2.FONT_HERSHEY_SIMPLEX, .4, (230, 230, 230), 1)
        cv2.putText(canvas, 'hollow P: forecast', (1290, 779), cv2.FONT_HERSHEY_SIMPLEX, .4, (230, 230, 230), 1)
        return canvas


class MultiRenderer:
    """Three synchronized camera tiles and one metric BEV, global IDs within this run."""
    def __init__(self, cameras, roi, trail_seconds=5, fusion_label='arithmetic mean'):
        self.cameras = cameras
        self.roi = roi
        self.trail_seconds = trail_seconds
        self.fusion_label = fusion_label
        self.history = defaultdict(deque)

    def update(self, time_s, states):
        # Called every physical tick, including when no image is rendered.
        for row in states:
            self.history[row['track_id']].append((time_s,row['x_m'],row['y_m']))
        for points in self.history.values():
            while points and points[0][0] < time_s-self.trail_seconds:
                points.popleft()

    @staticmethod
    def bev_pixel(xy):
        return round(1080+(xy[0]+3)*12.5), round(1030-(xy[1]+9)*12.5)

    def draw(self, images, frame, observations, states, group_counts):
        canvas = np.full((1080, 1920, 3), 24, np.uint8)
        positions = {'C1': (0, 0), 'C2': (960, 0), 'C3': (0, 540)}
        for name, image in images.items():
            view = image.copy()
            for row in observations:
                if row['camera'] != name:
                    continue
                track_id = row['track_id']
                color = color_for(track_id) if track_id != '' else (140, 140, 140)
                x1,y1,x2,y2 = [round(row[k]) for k in ('x1','y1','x2','y2')]
                cv2.rectangle(view,(x1,y1),(x2,y2),color,3)
                label = f'G{track_id}' if track_id != '' else row['reason']
                cv2.putText(view,label,(x1,max(80,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.8,color,2,cv2.LINE_AA)
                cv2.drawMarker(view,(round(row['foot_u']),round(row['foot_v'])),color,cv2.MARKER_CROSS,12,2)
            camera = self.cameras[name]
            for row in states:
                xy = [row['x_m'],row['y_m']]
                if row['state']=='predicted' and camera.sees_ground(xy):
                    p = tuple(np.rint(camera.to_image(xy)).astype(int))
                    color = color_for(row['track_id'])
                    cv2.circle(view,p,12,color,2)
                    cv2.putText(view,f'P{row["track_id"]}',(p[0]+12,p[1]),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2)
            cv2.rectangle(view,(0,0),(1920,50),(20,20,20),-1)
            cv2.putText(view,f'{name} GLOBAL IDs | frame {frame:08d} t={frame/10:.1f}s',
                        (12,33),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2)
            left,top = positions[name]
            canvas[top:top+540,left:left+960] = cv2.resize(view,(960,540))
        cv2.putText(canvas,'Ground XY (m), Z=0 assumed',(990,565),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),1)
        for x in range(-3,10,3):
            a,b = self.bev_pixel((x,-9)),self.bev_pixel((x,27))
            cv2.line(canvas,a,b,(65,65,65),1)
            cv2.putText(canvas,str(x),(a[0]-6,a[1]+18),cv2.FONT_HERSHEY_SIMPLEX,.4,(220,220,220),1)
        for y in range(-9,28,3):
            a,b=self.bev_pixel((-3,y)),self.bev_pixel((9,y))
            cv2.line(canvas,a,b,(65,65,65),1)
            cv2.putText(canvas,str(y),(1045,a[1]+4),cv2.FONT_HERSHEY_SIMPLEX,.4,(220,220,220),1)
        for row in states:
            key=row['track_id']; color=color_for(key); xy=(row['x_m'],row['y_m'])
            points=[self.bev_pixel((x,y)) for _,x,y in self.history[key] if self.cameras['C1'].in_roi((x,y))]
            if len(points)>1:
                cv2.polylines(canvas,[np.array(points,np.int32)],False,color,1)
            if row['in_roi']:
                p=self.bev_pixel(xy)
                cv2.circle(canvas,p,4,color,-1 if row['state']=='observed' else 1)
                cv2.putText(canvas,('G' if row['state']=='observed' else 'P')+str(key),
                            (p[0]+4,p[1]-3),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
        text=[f'Cameras: {" + ".join(images)}',f't={frame/10:.1f}s; source/playback 2 Hz, 1x',
              f'updated={sum(s["state"]=="observed" for s in states)}',
              f'forecast-only={sum(s["state"]=="predicted" for s in states)}',
              f'groups 1/2/3 members: {group_counts}',
              f'fusion gate 0.75m; {self.fusion_label}',
              'same global ID/color across views',
              'filled G: observation update', 'hollow P: forecast; no stale box',
              'GT is absent from prediction display']
        for index,line in enumerate(text):
            cv2.putText(canvas,line,(1280,610+30*index),cv2.FONT_HERSHEY_SIMPLEX,.55,(225,225,225),1,cv2.LINE_AA)
        return canvas
