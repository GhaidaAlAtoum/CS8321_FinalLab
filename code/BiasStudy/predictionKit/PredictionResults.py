import numpy as np
from numpy import sqrt
from numpy import argmax
from tabulate import tabulate
from sklearn.metrics import roc_curve, roc_auc_score
from .PredictionMetrics import PredictionMetrics
from scipy import interpolate

class PredictionResults(object):
    def __init__(self, y_true: dict, y_pred: dict, embeddings: np.array, best_threshold_method: str, model_name: str):
        self.variations = np.array(list(y_true.keys()))
        self.model_name = model_name
        self.image_embeddings = embeddings
        self.y_true_dict = {}
        self.y_pred_dict = {}
        self.metrics_per_variations = {}
        
        self.y_true_dict[self.variations[0]] = np.array(y_true[self.variations[0]])
        self.y_true_dict[self.variations[1]] = np.array(y_true[self.variations[1]])

        self.y_pred_dict[self.variations[0]] = np.array(y_pred[self.variations[0]])
        self.y_pred_dict[self.variations[1]] = np.array(y_pred[self.variations[1]])
        
        self.__calculate_metrics(best_threshold_method = best_threshold_method)
        self.bias_score_dict = self.calculate_bias_between_variations()

    def get_bias_score_dict(self) -> dict:
        return self.bias_score_dict
    
    def get_bias_str(self) -> dict:
        result = []
        for key, value in self.bias_score_dict.items():
            result.append("{} Bias: {}".format(key, value[bias_score]))
        return "\n".join(result)

    def get_base_bias_score(self) -> float:
        return self.bias_score_dict["base_fpr"]["bias_score"]
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def get_y_true_variation(self, variation: str) -> np.array:
        self.is_valid_variation(variation)
        return self.y_true_dict[variation]
    
    def get_y_pred_variation(self, variation: str) -> np.array:
        self.is_valid_variation(variation)
        return self.y_pred_dict[variation]
    
    def get_y_true(self) -> np.array:
        y_true = None
        for key, value in self.y_true_dict.items():
            if y_true is None:
                y_true = np.array(value)
            else:
                y_true = np.append(y_true, value)
        return y_true
    
    def get_y_pred(self) -> np.array:
        y_pred = None
        for key, value in self.y_pred_dict.items():
            if y_pred is None:
                y_pred = np.array(value)
            else:
                y_pred = np.append(y_pred, value)
        return y_pred
    
    def get_variations(self) -> np.array:
        return self.variations
    
    def get_image_embeddings(self) -> np.array:
        return self.image_embeddings
    
    def get_overall_metrics(self) -> PredictionMetrics:
        return self.overall_metrics
    
    def get_metrics_variation(self, variation: str) -> PredictionMetrics:
        return self.metrics_per_variations[variation]
    
    def is_valid_variation(self, variation: str):
        if variation not in self.variations:
            raise Exception("Undefined variation {}".format(variation))
    
    def __calculate_metrics(self, best_threshold_method='gmeans'):
        for variation in self.variations:
            self.metrics_per_variations[variation] = PredictionResults.calculate_metrics(
                variation = variation,
                y_true = self.get_y_true_variation(variation),
                y_pred = self.get_y_pred_variation(variation),
                best_threshold_method = best_threshold_method
            )
        
        overall_y_true = self.get_y_true()
        overall_y_pred = self.get_y_pred()
        self.overall_metrics = PredictionResults.calculate_metrics(
            variation = 'all',
            y_true = overall_y_true,
            y_pred = overall_y_pred,
            best_threshold_method = best_threshold_method
        )
        

    @classmethod
    def calculate_metrics(cls, variation: str, y_true: np.array, y_pred: np.array, best_threshold_method: str):
        fpr, tpr, thresholds = roc_curve(
            y_true = y_true,
            y_score = y_pred
        )
        
        auc_score  = roc_auc_score(
            y_true = y_true,
            y_score = y_pred
        )
        
        best_threshold_idx = PredictionResults.get_best_threshod_idx(
            tpr = tpr, 
            fpr = fpr, 
            method = best_threshold_method
        )
        
        return PredictionMetrics(
            variation = variation,
            fpr = fpr, 
            tpr = tpr, 
            thresholds = thresholds,
            best_threshold_method = best_threshold_method, 
            best_threshold_idx = best_threshold_idx,
            auc_score = auc_score
        )

    @classmethod
    def get_best_threshod_idx(cls, tpr: np.array, fpr: np.array, method):
        if method == 'gmeans':
            gmeans = sqrt(tpr * (1-fpr))
            ix = argmax(gmeans)
            return ix
        elif method == 'Youden':
            J = tpr - fpr
            ix = argmax(J)
            return ix
        raise Exception("Unknown method {}".format(method))
    
    
    def calculate_bias_between_variations(self):
        bias_score_dict = {}
        variation_0_metrics = self.get_metrics_variation(self.variations[0])
        variation_1_metrics = self.get_metrics_variation(self.variations[1])
        overall_metrics = self.get_overall_metrics()
        
        # Finding TPR at FPR F. 
        # Since the best threshold per light-light and dark-dark can be different
        # Use the best threshold for overall to get FPR overall
        # Interploate TPR at that FPR
        
        variation_0_interp = interpolate.interp1d(
            variation_0_metrics.get_fpr(), 
            variation_0_metrics.get_tpr()
        )
        
        variation_1_interp = interpolate.interp1d(
            variation_1_metrics.get_fpr(),
            variation_1_metrics.get_tpr()
        )

        overall_best_fpr = overall_metrics.get_fpr()[overall_metrics.get_best_threshold_idx()]
        
        tpr_variation0_at_overall_fpr = variation_0_interp(overall_best_fpr)
        tpr_variation1_at_overall_fpr = variation_1_interp(overall_best_fpr)
        
        bias_score_dict["fpr_at_best_threshold"] = {
            "fpr": overall_best_fpr,
            "bias_score": abs(tpr_variation0_at_overall_fpr - tpr_variation1_at_overall_fpr)
        }
        
        bias_score_dict["base_fpr"] = {
            "fpr": 1e-1,
            "bias_score": abs(variation_0_interp(1e-1) - variation_1_interp(1e-1))
        }
        
        return bias_score_dict

    def __str__(self):
        variation_0_title = "{}-{}".format(self.variations[0], self.variations[0])
        variation_1_title = "{}-{}".format(self.variations[1], self.variations[1])
        variation_0_metrics = self.get_metrics_variation(self.variations[0])
        variation_1_metrics = self.get_metrics_variation(self.variations[1])
        overall_metrics = self.get_overall_metrics()
        bias_dict = self.get_bias_score_dict()
        t = tabulate(
            [
                ['OverAll', 'Best Threshold IDX', overall_metrics.get_best_threshold_idx()],
                ['OverAll', 'Best Threshold', overall_metrics.get_best_threshold()],
                ['OverAll', 'AUC Score', overall_metrics.get_auc_score()],
                ['OverAll', 'Bias At Best Threshold', bias_dict["fpr_at_best_threshold"]["bias_score"]],
                ['OverAll', 'Base Bias at {} FPR'.format(bias_dict["base_fpr"]["fpr"]), bias_dict["base_fpr"]["bias_score"]],
                [variation_0_title, 'Best Threshold IDX', variation_0_metrics.get_best_threshold_idx()],
                [variation_0_title, 'Best Threshold', variation_0_metrics.get_best_threshold()],
                [variation_0_title, 'AUC Score', variation_0_metrics.get_auc_score()],
                [variation_1_title, 'Best Threshold IDX', variation_1_metrics.get_best_threshold_idx()],
                [variation_1_title, 'Best Threshold', variation_1_metrics.get_best_threshold()],
                [variation_1_title, 'AUC Score', variation_1_metrics.get_auc_score()]
            ],
            headers=['Variation', 'Metric Name', 'Value'],
            tablefmt='grid',
            floatfmt=".15f"
        )
        return t
    
    