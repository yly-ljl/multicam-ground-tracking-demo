"""Norfair association with a physical 0.5s FilterPy model and explicit missing ticks."""
#将当前时刻的融合组与已有轨迹关联，维护轨迹ID，并估计位置、速度和生命周期
import numpy as np
from filterpy.kalman import KalmanFilter
from norfair import Detection, Tracker
from norfair.filter import FilterFactory

#每次建立新轨迹时，为它创建独立的卡尔曼滤波器
class MetricFilterFactory(FilterFactory):
    def __init__(self, config):
        self.config = config

    def create_filter(self, initial_detection):
        c = self.config
        dt = c['tick_seconds']
        #内部状态有四个分量[X, Y, Vx, Vy]，当前测量只有两个分量[X, Y]
        kf = KalmanFilter(dim_x=4, dim_z=2)

        #运动矩阵 F
        '''
        短时间匀速预测
        新X  = 旧X + Vx×0.5
        新Y  = 旧Y + Vy×0.5
        新Vx = 旧Vx
        新Vy = 旧Vy
        '''
        kf.F = np.block([
            [np.eye(2), dt * np.eye(2)],
            [np.zeros((2, 2)), np.eye(2)]
        ])

        #观测矩阵 H
        kf.H = np.hstack((np.eye(2), np.zeros((2, 2)))) #测量对应状态中的X、Y，当前没有直接测量速度
        # Q描述运动预测的不确定性，例如人可能加速、减速或转向
        kf.Q = c['acceleration_std_mps2'] ** 2 * np.block([
            [dt**4 / 4 * np.eye(2), dt**3 / 2 * np.eye(2)],
            [dt**3 / 2 * np.eye(2), dt**2 * np.eye(2)]])

        kf.R = c['measurement_std_m'] ** 2 * np.eye(2) # R描述当前位置观测的不确定性

        # P描述跟踪器对自身状态估计的不确定性
        kf.P = np.diag([c['initial_position_std_m'] ** 2] * 2 +
                       [c['initial_velocity_std_mps'] ** 2] * 2)

        kf.x[:2, 0] = initial_detection.reshape(2)
        return kf

#管理多条轨迹，接收观测、调用跟踪库、整理输出
class GroundTracker:
    def __init__(self, config):
        self.config = config
        self.dt = config['tick_seconds']

        #当前B3使用Norfair原有的 Tracker，其时间关联方法是贪心配对
        method=config.get('association_method','greedy')
        if method=='greedy':tracker_class=Tracker

        elif method=='gated_global':
            from .association import GatedGlobalTracker
            tracker_class=GatedGlobalTracker
        elif method=='soft_global':
            from .association import SoftAssignmentTracker
            tracker_class=SoftAssignmentTracker
        else:raise ValueError(f'Unknown association method: {method}')
        self.tracker = tracker_class(distance_function='euclidean',
                               distance_threshold=config['distance_threshold_m'],
                               hit_counter_max=config['hit_counter_max'],
                               initialization_delay=config['initialization_delay'],
                               filter_factory=MetricFilterFactory(config),
                               past_detections_length=0)
        self.last_tick = None
        self.observation_counts = {}

    #把当前观测送入持续更新
    def update(self, time_s, observations):
        tick = int(round(time_s / self.dt)) #把时间换成内部步号
        if abs(time_s - tick * self.dt) > 1e-8 or (self.last_tick is not None and tick <= self.last_tick):
            raise ValueError('Time must increase on the configured physical tick grid.')
        first = tick if self.last_tick is None else self.last_tick + 1
        events = []
        states = []
        for current in range(first, tick + 1):
            actual_observations = observations if current == tick else []
            # 把融合位置包装成Norfair接受的观测
            detections = [
                Detection(
                    points=np.array([[o['x_m'], o['y_m']]]),
                    label='person',
                    data={'obs_id': o['obs_id'], 'tick': current}
                )
                for o in actual_observations
            ]
            # period=1 is lifecycle accounting, not dt. F/Q already encode 0.5s.
            active = self.tracker.update(detections=detections, period=1)
            states = []
            for obj in active:
                #检查轨迹最近一次收到的观测是不是来自当前时间步
                observed = obj.last_detection.data['tick'] == current
                if observed:
                    self.observation_counts[obj.id] = self.observation_counts.get(obj.id, 0) + 1
                x, y = obj.estimate[0]
                vx, vy = obj.estimate_velocity[0]
                count = self.observation_counts.get(obj.id, 0)
                states.append({'time_s': current * self.dt, 'track_id': obj.id,
                               'state': 'observed' if observed else 'predicted',
                               'obs_id': obj.last_detection.data['obs_id'] if observed else '',
                               'x_m': float(x), 'y_m': float(y), 'z_m': 0.,
                               'vx_mps': float(vx), 'vy_mps': float(vy),
                               'velocity_status': 'estimated' if count >= 2 else 'prior_zero', #表示至少收到两次观测后，速度被标为已估计。
                               'observations_so_far': count, 'age_s': obj.age * self.dt,
                               'since_observation_s': (current - obj.last_detection.data['tick']) * self.dt,
                               'hit_counter': obj.hit_counter})
            events.append({'time_s': current * self.dt, 'inserted_missing_tick': current != tick,
                           'observations': len(detections), 'active_tracks': len(states)})
        self.last_tick = tick
        return states, events
