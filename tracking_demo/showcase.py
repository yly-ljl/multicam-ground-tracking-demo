"""B3 presentation only: all-state equal-scale overview plus a labeled zoom."""
from collections import defaultdict,deque
import cv2
import numpy as np
from .render import color_for


class ShowcaseRenderer:
    def __init__(self,cameras,roi,bounds=(-6,13,-13,29),zoom=(0,9,9,24)):
        self.cameras=cameras;self.roi=roi;self.bounds=list(bounds);self.zoom=zoom
        self.history=defaultdict(deque);self.overview_clipped_states=0

    def update(self,time_s,states):
        #把当前状态的位置加入对应ID的绘图历史
        for s in states:
            self.history[s['track_id']].append((time_s,s['x_m'],s['y_m']))

        for points in self.history.values():
            while points and points[0][0]<time_s-5:points.popleft() #移除超过5秒的旧点

        for s in states:
            self.bounds[0]=min(self.bounds[0],int(np.floor(s['x_m']))-1)
            self.bounds[1]=max(self.bounds[1],int(np.ceil(s['x_m']))+1)
            self.bounds[2]=min(self.bounds[2],int(np.floor(s['y_m']))-1)
            self.bounds[3]=max(self.bounds[3],int(np.ceil(s['y_m']))+1)

    def _map(self,canvas,bounds,area,states,title):
        x0,y0,width,height=area;xmin,xmax,ymin,ymax=bounds
        scale=min(width/(xmax-xmin),height/(ymax-ymin))
        left=x0+(width-scale*(xmax-xmin))/2;top=y0+(height-scale*(ymax-ymin))/2
        def pixel(xy):return round(left+(xy[0]-xmin)*scale),round(top+(ymax-xy[1])*scale)
        def inside(xy):return xmin<=xy[0]<=xmax and ymin<=xy[1]<=ymax
        cv2.putText(canvas,title,(x0,y0-17),0,.57,(245,245,245),1,cv2.LINE_AA)
        cv2.rectangle(canvas,pixel((xmin,ymin)),pixel((xmax,ymax)),(110,110,110),1)
        for x in range(int(np.ceil(xmin/3))*3,int(xmax)+1,3):
            a,b=pixel((x,ymin)),pixel((x,ymax));cv2.line(canvas,a,b,(48,48,48),1)
            cv2.putText(canvas,str(x),(a[0]-8,a[1]+20),0,.4,(200,200,200),1)
        for y in range(int(np.ceil(ymin/3))*3,int(ymax)+1,3):
            a,b=pixel((xmin,y)),pixel((xmax,y));cv2.line(canvas,a,b,(48,48,48),1)
            cv2.putText(canvas,str(y),(a[0]-31,a[1]+4),0,.4,(200,200,200),1)
        rx0,rx1,ry0,ry1=self.roi
        if inside((rx0,ry0)) and inside((rx1,ry1)):
            cv2.rectangle(canvas,pixel((rx0,ry0)),pixel((rx1,ry1)),(160,130,70),1)
        for s in states:
            points=[pixel((x,y)) for _,x,y in self.history[s['track_id']] if inside((x,y))]
            color=color_for(s['track_id'])
            for a,b in zip(points,points[1:]):cv2.line(canvas,a,b,color,1,cv2.LINE_AA)
        shown=0
        for s in states:
            xy=(s['x_m'],s['y_m'])
            if not inside(xy):continue
            shown+=1;p=pixel(xy);color=color_for(s['track_id']);pred=s['state']=='predicted'
            cv2.circle(canvas,p,5 if pred else 4,color,1 if pred else -1,cv2.LINE_AA)
            label=('P' if pred else 'G')+str(s['track_id']);loc=(p[0]+5,p[1]-5+(s['track_id']%2)*12)
            cv2.putText(canvas,label,loc,0,.43,(15,15,15),3,cv2.LINE_AA)
            cv2.putText(canvas,label,loc,0,.43,color,1,cv2.LINE_AA)
        return shown,scale

    #将几种数据画在同一张画布上
    def draw(self,images,frame,observations,states):
        canvas=np.full((1080,1920,3),22,np.uint8)
        for row,name in enumerate(('C1','C2','C3')):
            tile=cv2.resize(images[name],(640,360)) #原图1920×1080缩成640×360
            for o in observations:
                if o['camera']!=name:continue
                tid=o['track_id'];color=color_for(tid) if tid!='' else (125,125,125)
                a,b=(round(o['x1']/3),round(o['y1']/3)),(round(o['x2']/3),round(o['y2']/3))
                cv2.rectangle(tile,a,b,color,1)
                label=f'G{tid}' if tid!='' else 'outside ROI'
                cv2.putText(tile,label,(a[0],max(36,a[1]-3)),0,.42,color,1,cv2.LINE_AA)
            for s in states:
                xy=[s['x_m'],s['y_m']]
                if s['state']=='predicted' and self.cameras[name].sees_ground(xy):
                    #对于纯预测状态，程序还会把其地面位置投回相机画面，画一个空心预测点。
                    uv=self.cameras[name].to_image(xy)/3;p=tuple(np.rint(uv).astype(int));color=color_for(s['track_id'])
                    cv2.circle(tile,p,5,color,1);cv2.putText(tile,f'P{s["track_id"]}',(p[0]+6,p[1]),0,.4,color,1)
            cv2.rectangle(tile,(0,0),(640,26),(20,20,20),-1)
            cv2.putText(tile,f'{name} | B3 | frame {frame:08d}  t={frame/10:.1f}s',(9,18),0,.49,(255,255,255),1,cv2.LINE_AA)
            canvas[row*360:(row+1)*360,:640]=tile
        cv2.putText(canvas,'B3: THREE CAMERAS / GROUND XY (m), Z=0 ASSUMED',(670,30),0,.66,(245,245,245),1,cv2.LINE_AA)
        cv2.putText(canvas,'Filled G: observed update   Hollow P: forecast only   Trail: last 5 seconds',(670,58),0,.5,(225,225,225),1,cv2.LINE_AA)
        cv2.putText(canvas,f'Source/playback 2 Hz, 1x | frame {frame:08d} | t={frame/10:.1f}s | GT absent',(670,83),0,.5,(210,210,210),1,cv2.LINE_AA)
        total,scale=self._map(canvas,self.bounds,(700,140,445,870),states,'OVERVIEW: every current state + context')
        zoomed,zscale=self._map(canvas,self.zoom,(1240,140,615,870),states,'ZOOM: X[0,9], Y[9,24] meters')
        self.overview_clipped_states+=len(states)-total
        assert total==len(states),'The overview must show all current states, including out-of-ROI forecasts'
        observed=sum(s['state']=='observed' for s in states)
        cv2.putText(canvas,f'All {total}: G {observed}, P {len(states)-observed} | zoom shows {zoomed}/{total}',(670,1051),0,.49,(235,235,235),1)
        cv2.putText(canvas,'Equal scale X/Y in each map; ROI outline; IDs independent of other experiment conditions',(670,1073),0,.43,(190,190,190),1)
        return canvas
