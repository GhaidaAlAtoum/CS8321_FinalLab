import os
import sys
import platform
from os import environ

if environ.get('BIAS_STUDY_TF_SETUP') is None:
    environ['KMP_DUPLICATE_LIB_OK']='True'
    environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    import tensorflow as tf
    from tensorflow import keras

    print(f"Python Platform: {platform.platform()}")
    print(f"Tensor Flow Version: {tf.__version__}")
    print(f"Keras Version: {keras.__version__}")
    print()
    print(f"Python {sys.version}")
    environ["BIAS_STUDY_TF_SETUP"] = '1'

from .PredictionMetrics import PredictionMetrics
from .PredictionResults import PredictionResults
from . import PredictionToolKit
from . import PredictionPlotKit
from . import GradCam