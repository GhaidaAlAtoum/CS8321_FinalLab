from matplotlib import pyplot as plt
from .PredictionMetrics import PredictionMetrics
from .PredictionResults import PredictionResults
from typing import List
from collections.abc import Sequence
import pathlib

def plot_multiple_roc(
    prediction_results: List[PredictionResults],
    save: bool = True,
    save_dir: str = "."
):
    number_of_predictions = len(prediction_results)
    f, axs = plt.subplots(1, number_of_predictions, sharey=True, figsize=(number_of_predictions * 5,  5))
    file_names = []
    for idx, prediction_result in enumerate(prediction_results):
        plot_roc(
            prediction_result = prediction_result,
            ax = axs[idx],
            save = False
        )
        file_names.append(prediction_result.get_model_name())
    
    if save:
        output_dir = "{}/output_images".format(save_dir)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True) 
        image_path = "{}/{}.png".format(output_dir, "_".join(file_names))
        plt.savefig(image_path, bbox_inches='tight')


def plot_roc(prediction_result: PredictionResults, fig_size: (int,int) = (5,5), ax: plt.axis = None, save: bool = True, save_dir: str = "."):
    variations = prediction_result.get_variations()
    variation_0 = variations[0]
    variation_1 = variations[1]
    
    variation_0_metrics = prediction_result.get_metrics_variation(variation_0)
    variation_1_metrics = prediction_result.get_metrics_variation(variation_1)

    overall_metrics = prediction_result.get_overall_metrics()
    
    if ax is None:
        plt.figure(figsize=fig_size)
        ax = plt.gca()

    ax.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    
    ax.plot(
        variation_0_metrics.get_fpr(),
        variation_0_metrics.get_tpr(), 
        marker='.',
        markersize=10,
        label="{}-{}-AucScore({:.5f})".format(
            variation_0,
            variation_0,
            variation_0_metrics.get_auc_score()
        ),
        markevery=[variation_0_metrics.get_best_threshold_idx()]
    )

    ax.plot(
        variation_1_metrics.get_fpr(),
        variation_1_metrics.get_tpr(), 
        marker='.',
        markersize=10,
        label="{}-{}-AucScore({:.5f})".format(
            variation_1,
            variation_1,
            variation_1_metrics.get_auc_score()
        ),
        markevery=[variation_1_metrics.get_best_threshold_idx()]
    )

    ax.plot(
        overall_metrics.get_fpr(),
        overall_metrics.get_tpr(), 
        marker='.',
        markersize=10,
        label="Overall-AucScore({:.5f})".format(
            overall_metrics.get_auc_score()
        ),
        markevery=[overall_metrics.get_best_threshold_idx()]
    )
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title("ROC - {} - BaseBiasScore {:.8f}".format(prediction_result.get_model_name(), prediction_result.get_base_bias_score()))
    ax.legend()
    
    if save:
        output_dir = "{}/output_images".format(save_dir)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True) 
        image_path = "{}/{}.png".format(output_dir, prediction_result.get_model_name())
        ax.figure.savefig(image_path)