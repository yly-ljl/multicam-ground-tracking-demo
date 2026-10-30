"""WILDTRACK C1 undistorted pixels <-> metric ground, world-to-camera R/t."""
#建立图像像素与统一地面坐标之间的转换关系，并检查转换结果是否可以使用
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def grid_to_world(position_id):
    return np.array([-3.0 + .025 * (position_id % 480),
                     -9.0 + .025 * (position_id // 480)], dtype=float)


class GroundCamera:
    def __init__(self, data_root, calibration='CVLab1', roi=(-3., 9., -9., 27.)):
        root = Path(data_root) / 'calibrations'
        #读取内参和畸变参数
        intrinsic = ET.parse(root / 'intrinsic_zero' / f'intr_{calibration}.xml').getroot()

        #K是三乘三内参矩阵，其内部参数描述一个点位于相机前方某个方向时，会落在什么像素位置
        self.K = np.fromstring(intrinsic.find('camera_matrix/data').text, sep=' ').reshape(3, 3)
        self.distortion = np.fromstring(intrinsic.find('distortion_coefficients/data').text, sep=' ')
        if np.any(self.distortion != 0):
            raise ValueError('This pipeline expects the zero-distortion calibration.')
        #读取外参，统一单位
        extrinsic = ET.parse(root / 'extrinsic' / f'extr_{calibration}.xml').getroot()
        self.rvec = np.fromstring(extrinsic.findtext('rvec'), sep=' ')
        self.t_cm = np.fromstring(extrinsic.findtext('tvec'), sep=' ')

        #得到旋转矩阵 R 和平移向量 t
        self.t = self.t_cm / 100.0 #以米为单位
        self.R = cv2.Rodrigues(self.rvec)[0]

        #构造地面专用的转换矩阵  H = K[r1 r2 t]
        self.H = self.K @ np.column_stack((self.R[:, :2], self.t))#因为地面点的 Z=0，旋转矩阵中与Z相乘的那一列不再贡献结果，所以这里只取前两列

        self.H_inv = np.linalg.inv(self.H)
        self.roi = tuple(roi)
        '''
        H      地面 → 图像
        H_inv  图像 → 地面
        roi    规定的地面研究区域
        '''

    def in_roi(self, xy):
        x, y = xy
        xmin, xmax, ymin, ymax = self.roi
        return bool(xmin <= x < xmax and ymin <= y < ymax)

    def depth(self, xy):
        #把地面位置补成 [X,Y,0]，转换到相机坐标，再取第三个分量 Zc
        return float((self.R @ np.array([*xy, 0.]) + self.t)[2])

    #把地面位置画回相机画面，用途是展示纯预测状态
    def to_image(self, xy):
        p = self.H @ np.array([*xy, 1.]) #与 to_ground() 方向相反,输入X，Y，输出单位像素
        return p[:2] / p[2]

    #检查地面点是否落入相机画面
    def sees_ground(self, xy):
        if self.depth(xy) <= 0:
            return False
        u, v = self.to_image(xy)
        #检查是否在1920×1080的画面范围内
        return bool(0 <= u < 1920 and 0 <= v < 1080)

    def to_ground(self, uv):

        p = self.H_inv @ np.array([*uv, 1.]) #s[u v 1]T = H[X Y 1]T

        #排除计算出现无效数值，或归一化分母过于接近零的情况
        if not np.isfinite(p).all() or abs(p[2]) < 1e-9:
            return np.array([np.nan, np.nan]), 'invalid_homogeneous_projection'

        xy = p[:2] / p[2] #p[:2]->[X/s Y/s 1/s],除以p[2]后，->[X Y 1]

        #排除算出的点不在相机前方的情况
        if self.depth(xy) <= 0:
            return xy, 'behind_camera'
        #排除点超出规定的地面研究区域的情况
        if not self.in_roi(xy):
            return xy, 'outside_roi'
        return xy, 'accepted'

    def describe(self):
        return {'K': self.K.tolist(), 'rvec_world_to_camera': self.rvec.tolist(),
                't_source_cm': self.t_cm.tolist(), 't_used_m': self.t.tolist(),
                'R_world_to_camera': self.R.tolist(), 'H_ground_m_to_undistorted_pixels': self.H.tolist(),
                'roi_m': self.roi, 'image_hw': [1080, 1920], 'ground_z_m': 0}
