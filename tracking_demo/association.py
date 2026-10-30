"""Prediction-only gated maximum-cardinality/minimum-distance association."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from norfair import Tracker


def gated_global_pairs(distance_matrix,distance_threshold):
    """Rows are detections, columns tracks; preserve the input and strict gate."""
    distances=np.asarray(distance_matrix,dtype=float)
    if distances.ndim!=2:raise ValueError('Expected a two-dimensional distance matrix')
    n,m=distances.shape
    if not n or not m:return [],[]
    allowed=np.isfinite(distances)&(distances<distance_threshold)
    if not allowed.any():return [],[]
    # Across any reassignment, <=min(n,m) distances can each change by <=2*bound.
    # Losing one real match therefore costs more than all possible distance gains.
    bound=float(np.max(np.abs(distances[allowed])))
    unmatched=(min(n,m)+1)*(2*bound+1)
    forbidden=(n+m+1)*unmatched
    costs=np.column_stack((np.where(allowed,distances,forbidden),np.full((n,n),unmatched)))
    rows,cols=linear_sum_assignment(costs)
    real=cols<m
    rows,cols=rows[real],cols[real]
    assert np.all(allowed[rows,cols])
    return rows.tolist(),cols.tolist()


class GatedGlobalTracker(Tracker):
    """Only replace matching; inherit prediction, lifecycle, stages, and hit updates."""
    def match_dets_and_objs(self,distance_matrix,distance_threshold):
        return gated_global_pairs(distance_matrix,distance_threshold)


def soft_global_pairs(distance_matrix,distance_threshold):
    """Minimize sum(d/tau) + unmatched observations; no track-side charge."""
    distances=np.asarray(distance_matrix,dtype=float)
    if distances.ndim!=2:raise ValueError('Expected a two-dimensional distance matrix')
    if not np.isfinite(distance_threshold) or distance_threshold<=0:raise ValueError('Expected a finite positive distance threshold')
    n,m=distances.shape
    if not n or not m:return [],[]
    allowed=np.isfinite(distances)&(distances>=0)&(distances<distance_threshold)
    # Every cost is nonnegative; all n rows unmatched cost n, so n+1 forbids an edge.
    real=np.full((n,m),float(n+1))
    real[allowed]=distances[allowed]/distance_threshold
    costs=np.column_stack((real,np.ones((n,n))))
    rows,cols=linear_sum_assignment(costs)
    selected=cols<m;rows,cols=rows[selected],cols[selected]
    assert np.all(allowed[rows,cols])
    return rows.tolist(),cols.tolist()


class SoftAssignmentTracker(Tracker):
    """Finite observation-side unmatched cost; inherit all lifecycle/stage behavior."""
    def match_dets_and_objs(self,distance_matrix,distance_threshold):
        return soft_global_pairs(distance_matrix,distance_threshold)
