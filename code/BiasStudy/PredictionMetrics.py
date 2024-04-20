import numpy as np
import pandas as pd
from keras.models import Model
from scipy import interpolate
from tabulate import tabulate

class PredictionMetrics(object):
    def __init__(self, variation: str, fpr: np.array, tpr: np.array, thresholds: np.array, best_threshold_method: str, best_threshold_idx: int, auc_score: float):
        self.fpr = fpr
        self.tpr = tpr
        self.thresholds = thresholds
        self.best_threshold_idx = best_threshold_idx
        self.auc_score = auc_score
        self.best_threshold_method = best_threshold_method
    
    def get_fpr(self) -> np.array:
        return self.fpr
    
    def get_tpr(self) -> np.array:
        return self.tpr
    
    def get_thresholds(self) -> np.array:
        return self.thresholds
    
    def get_best_threshold_idx(self) -> int:
        return self.best_threshold_idx
    
    def get_best_threshold(self):
        return self.thresholds[self.best_threshold_idx]
    
    def get_auc_score(self):
        return self.auc_score
    
    def get_best_threshold_method(self):
        return self.best_threshold_method
    
    #TODO: Fix Naming of function
    def get_tar_at_far(self, prefix='') -> str:
        fpr_levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        f_interp = interpolate.interp1d(self.fpr, self.tpr)      

        tpr_at_fpr = [f_interp(x) for x in fpr_levels]
        result = []
        for (far, tar) in zip(fpr_levels, tpr_at_fpr):
            result.append('{}TAR @ FAR = {} : {}'.format(prefix, far, tar))
        
        return "\n".join(result)

    def __str__(self):
        t = tabulate(
            [
                ['Best Threshold IDX', self.get_best_threshold_idx()],
                ['Best Threshold', self.get_best_threshold()],
                ['AUC Score', self.get_auc_score()]
            ],
            headers=['Metric Name', 'Value'],
            tablefmt='grid',
            floatfmt=".15f"
        )
        return t
        
