"""Build an 80-second narrated-by-captions edition and a one-page visual summary."""
import csv,json,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
root=Path(__file__).resolve().parents[1];out=root/'outputs/PHASE5_delivery'


def card(title,lines):
    frame=np.full((1200,1920,3),(24,22,18),np.uint8)
    cv2.putText(frame,title,(110,160),0,1.35,(245,245,245),2,cv2.LINE_AA)
    cv2.line(frame,(110,205),(1800,205),(200,160,70),3)
    for n,line in enumerate(lines):cv2.putText(frame,line,(115,300+92*n),0,.83,(225,225,225),1,cv2.LINE_AA)
    return frame


def explainer():
    writer=cv2.VideoWriter(str(out/'explainer_80s.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),2.,(1920,1200))
    if not writer.isOpened():raise RuntimeError('Explainer writer failed')
    frames=0
    def add(frame,count):
        nonlocal frames
        for _ in range(count):writer.write(frame);frames+=1
    add(card('B3: THREE-CAMERA GROUND TRACKING',[
        'WILDTRACK | fixed calibrated cameras | YOLO11n | offline 2 Hz',
        'Ground (X,Y,0) in meters; human height and full 3D boxes are not estimated.',
        'The next 60 seconds are continuous B3 predictions on previously seen frames 800-1395.',
        'All current states remain visible in the overview; the zoom is explicitly labeled.'
    ]),8)
    add(card('HOW TO READ THE PREDICTION',[
        'Filled G = current observation update; hollow P = forecast without a new observation.',
        'Camera boxes and ground points share the condition-local ID/color.',
        'Trails show the last 5 seconds. Both ground maps use equal X/Y scales.',
        'No GT appears in the continuous prediction section; imperfect tracks are retained.'
    ]),8)
    cap=cv2.VideoCapture(str(out/'reference_replay/prediction.mp4'));source=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        canvas=np.full((1200,1920,3),20,np.uint8);canvas[120:]=frame
        if source<40:
            title='B3 retained baseline: box feet -> ground fusion -> greedy temporal tracking'
            line='Three camera views + all-state overview + labeled zoom. Source time and IDs are unchanged.'
        elif source<80:
            title='E004: identity evaluation uses complete GT-by-observed distance relations (<=1m)'
            line='Seen-reference B3 IDF1 = 49.02%. Project-specific split/ROI; not an official benchmark score.'
        else:
            title='E005: more global matches did not improve the overall identity result'
            line='Reference B3_GLOBAL IDF1 = 48.26%; thresholds failed. B3 remains the default; heldout unused.'
        cv2.putText(canvas,title,(35,42),0,.77,(245,245,245),1,cv2.LINE_AA)
        cv2.putText(canvas,line,(35,87),0,.62,(210,210,210),1,cv2.LINE_AA)
        writer.write(canvas);frames+=1;source+=1
    cap.release();assert source==120
    add(card('FIXED COMPARISON: RETAIN B3',[
        'Ground-point IDF1@1m, observed only, full ROI:',
        'Old development:     B3 55.47%  -> GLOBAL 52.72%   (-2.75 pp)',
        'New development:    B3 60.17%  -> GLOBAL 54.28%   (-5.89 pp)',
        'Seen reference:        B3 49.02%  -> GLOBAL 48.26%   (-0.76 pp)',
        'All three identity criteria failed. No second candidate or heldout comparison was run.'
    ]),8)
    case=np.full((1200,1920,3),20,np.uint8)
    cv2.putText(case,'GT DIAGNOSTIC ONLY: MORE PAIRS CAN INHERIT THE WRONG PERSON',(60,65),0,.86,(100,200,255),2,cv2.LINE_AA)
    image=cv2.imread(str(root/'outputs/E005_temporal_assignment/cases/negative_04.jpg'));image=cv2.resize(image,(1800,900));case[140:1040,60:1860]=image
    cv2.putText(case,'Frame840: the actual global run joins prior person455 to current193; B3 keeps193 ->193.',(60,1105),0,.67,(235,235,235),1,cv2.LINE_AA)
    add(case,8)
    add(card('CAPABILITY BOUNDARIES / REVIEW STATUS',[
        'Occlusion, footpoint bias, duplicate groups and ID errors remain.',
        'GT identity anomalies are retained in evaluation and documented separately.',
        'Timing is a local measured workload, not production realtime certification.',
        'Frames1400-1995 remain unused. No training, new model, cloud publish or remote Git.',
        'Engineering complete; final package review and user acceptance are still required.'
    ]),8)
    writer.release();assert frames==160
    (out/'explainer_build.json').write_text(json.dumps({'frames':frames,'fps':2,'duration_seconds':80,'size':[1920,1200],
       'continuous_prediction_frames':120,'intro_frames':16,'outro_frames':24,'audio':'none','GT_only_page':'explicitly labeled final diagnostic page',
       'scope':'Source prediction frames copied in full below a separate 120px caption band. No model/tracker updates.'},indent=2),encoding='utf-8')


def one_page():
    perf=json.loads((out/'performance/performance.json').read_text(encoding='utf-8'))
    image=Image.new('RGB',(1240,1754),'#101820');draw=ImageDraw.Draw(image)
    fonts={n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n) for n in (22,25,28,32,44)}
    def text(x,y,s,size=25,color='#e8edf2'):draw.text((x,y),s,font=fonts[size],fill=color)
    def paragraph(y,s,size=25,width=1090):
        line=''
        for ch in s:
            if draw.textlength(line+ch,font=fonts[size])>width:
                text(70,y,line,size);y+=size+14;line=ch
            else:line+=ch
        if line:text(70,y,line,size);y+=size+14
        return y
    text(70,65,'三相机行人地面跟踪：阶段结果',44)
    text(70,135,'2026-10-16｜E004＋E005｜保留B3｜整包待最终审查',25,'#a9d5f2')
    first='WILDTRACK固定三相机，YOLO11n框底投到地面后等权融合。'
    second='真实时间步0.5秒；输出(X,Y,0)，未估计人体高度或完整三维框。'
    assert all(draw.textlength(s,font=fonts[25])<=1090 for s in (first,second))
    y=paragraph(205,first)
    y=paragraph(y,second)
    text(70,y+20,'身份对照：完整距离关系，observed，全ROI，≤1m',28,'#ffc76a');y+=85
    columns=[70,420,640,850,1040]
    for x,s in zip(columns,['片段','B3 IDF1','GLOBAL','差值pp','判断']):text(x,y,s,25,'#a9d5f2')
    y+=55
    for row in [('旧开发0–495','55.47%','52.72%','−2.75','失败'),('新开发500–795','60.17%','54.28%','−5.89','失败'),('已见参考800–1395','49.02%','48.26%','−0.76','失败')]:
        for x,s in zip(columns,row):text(x,y,s,22)
        draw.line((70,y+43,1165,y+43),fill='#344653',width=1);y+=64
    y=paragraph(y+25,'唯一候选只改时间分配。十二项门槛中九项空间门槛通过，三项身份门槛失败：拒绝采用，不改阈值、不加第二候选，1400–1995保留段不开启。')
    text(70,y+18,'真实正负例与仍存在的问题',28,'#ffc76a');y+=77
    y=paragraph(y,'frame140有正确延续的局部正例；frame840却在配对数更多时把人物455的轨迹接给193。几何配对数更多、总ID更少，都不等于身份更准确。遮挡、脚点偏差、同人分裂及GT身份异常仍保留。')
    text(70,y+18,'工程与复现证据',28,'#ffc76a');y+=77
    for sentence in ('31项测试通过，三路融合输入及B3回归一致。',
                     'E004新检测180张；性能另180次图像推理。',
                     '性能首遍及两次已加载重复复用同一模型，没有保留段推理。',
                     '没有新模型、训练或远端发布。'):
        y=paragraph(y,sentence)
    warm=perf['passes'][1:]
    y=paragraph(y+6,f'性能：首启合计{perf["first_startup_plus_first_pass_seconds"]:.2f}秒，{perf["first_startup_images_per_second"]:.2f}图/秒；两遍已加载为'+ '、'.join(f'{r["images_per_second_processing"]:.2f}' for r in warm)+'图/秒。包含读图、同步推理到编码；不是冷磁盘缓存测试或生产实时性认证。',22)
    y=paragraph(y+10,'交付：连续60秒预测视频＋80秒讲解版；时间/ID/XY与成员映射；完整指标与正负案例；播放、缓存复现、新目录图像推理三种入口。',22)
    text(70,1640,'入口：README.md / docs/DEMO_INDEX.md / docs/EVALUATION.md',22,'#a9d5f2')
    text(70,1680,'E001–E005原证据保留；Git仅本地；v0.2待用户最终验收。',22,'#a9d5f2')
    assert y<1620,(y,'One-page text overlaps footer')
    image.save(out/'results_one_page.png')


if __name__=='__main__':explainer();one_page();print('Built 80s explainer and one-page visual summary')
