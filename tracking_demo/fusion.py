"""Deterministic camera-order complete-link gated fusion before temporal tracking."""
#将同一时刻、不同相机得到的地面观测分组，再为每组计算一个融合位置
import numpy as np
from scipy.optimize import linear_sum_assignment


def fuse_observations(by_camera, cameras, gate, frame, variant='F'):
    groups = []
    #多台相机以此处理，当前顺序为C1→C2→C3
    for camera in cameras:

        observations = by_camera.get(camera, [])  #by_camera为按相机分类的有效观测，每条都已有米制XY
        matched = set() #matched 用来记录本轮哪些观测已经加入了已有组
        if groups and observations:

            # 每一行代表当前相机的一条观测，每一列代表一个已有组。
            costs = np.zeros((len(observations), len(groups)))
            allowed = np.zeros_like(costs, dtype=bool)
            for i, observation in enumerate(observations):
                # 取出当前观测的位置
                point = np.array([observation['x_m'], observation['y_m']])
                for j, group in enumerate(groups):
                    # 取出组成员的位置
                    members = np.array([[m['x_m'], m['y_m']] for m in group])
                    costs[i, j] = np.linalg.norm(point-members.mean(axis=0))
                    allowed[i, j] = (
                            camera not in {m['camera'] for m in group} #组内还没有当前相机的成员
                            and np.all(
                                np.linalg.norm(members-point, axis=1) <= gate #新观测到每个已有成员的距离，都不超过gate
                        )
                    )
            # penalty给“不匹配”设置一个足够大的代价。
            penalty = (len(observations)+len(groups)+1)*(gate+1)
            matrix = np.column_stack((
                np.where(allowed, costs, 10*penalty), #把禁止配对的位置替换成大代价。
                np.full(
                    (len(observations), len(observations)),
                    penalty) #增加允许“不匹配”的列
            ))

            #选择总成本最小的一对一方案
            '''
            具体来将，即：
            （1）每条观测选择一列
            （2）同一列不能被两条观测同时选择
            （3）所有选择的代价加起来尽量小
            '''
            rows, cols = linear_sum_assignment(matrix)

            for i, j in zip(rows, cols):
                if j < len(groups) and allowed[i, j]:
                    groups[j].append(observations[i])
                    matched.add(i)

        #第一次处理C1时，groups 为空，没有组可以加入，因此会为每条未匹配观测独立建立一个组
        #之后则会把剩余观测各自建立新组
        groups.extend(
            [observation]
            for i, observation in enumerate(observations)
            if i not in matched
        )
    result = []
    for index, members in enumerate(groups):
        points = np.array([[m['x_m'], m['y_m']] for m in members])
        mean = points.mean(axis=0)

        #计算成员之间的两两距离
        pairwise = np.linalg.norm(
            points[:, None, :]-points[None, :, :],
            axis=2
        )
        result.append(
            {'fusion_id': f'{variant}_{frame:08d}_{index:03d}',
                       'members': members,
                       'x_m': float(mean[0]),
                       'y_m': float(mean[1]),
                       'max_pair_distance_m': float(pairwise.max()) #记录组内最远两个成员相隔多远，用于描述组内位置分散程度
             }
        )
    return result
