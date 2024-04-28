import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_config_path', type=str, required=True)
parser.add_argument('-o', '--logging_output_dir', type=str, required=True)

# sys.path.append('/notebooks/code/')
from BiasStudy import datasets, predictionKit
from BiasStudy.datasets import FairFaceDataset
from BiasStudy.config.TrainingConfig import BiasStudyConfig
from BiasStudy.trainingKit import TrainingLogger
from BiasStudy.trainingKit.TrainingModel import BiasModel
import tensorflow as tf
import tensorflow.keras
from keras.preprocessing.image import ImageDataGenerator

global args
args = parser.parse_args()

main_logger = TrainingLogger.setup_logger(
    "main", 
    args.logging_output_dir
)

main_logger.info("Loading Model Config from {}".format(args.model_config_path))

bias_config = BiasStudyConfig(args.model_config_path)
train_dataset_config = bias_config.get_train_dataset_config()


main_logger.info("Reading Dataset from {}".format(train_dataset_config.get_fair_face_path()))

fair_face_dataset = FairFaceDataset(
    data_dir = train_dataset_config.get_fair_face_path(),
    train_labels_csv_name = "fairface_label_train.csv",
    validation_labels_csv_name = "fairface_label_val.csv",
    under_sample = True,
    image_shape = (224,224,3),
    feature_column = "file",
    output_col = "binary_race",
    overwrite_sample_number = train_dataset_config.get_sample_number()
)

train_df = fair_face_dataset.get_train_pd()

train_datagen = ImageDataGenerator(
    validation_split = 0.2
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    target_size = (fair_face_dataset.get_image_width(),fair_face_dataset.get_image_height()),
    x_col = fair_face_dataset.get_feature_col_name(),
    y_col = fair_face_dataset.get_output_col_name(), 
    batch_size = bias_config.get_general_config().get_batch_size(),
    class_mode = "categorical",
    subset = "training"
)


validation_generator = train_datagen.flow_from_dataframe(
    train_df,
    target_size = (fair_face_dataset.get_image_width(),fair_face_dataset.get_image_height()),
    x_col = fair_face_dataset.get_feature_col_name(),
    y_col = fair_face_dataset.get_output_col_name(),
    batch_size = bias_config.get_general_config().get_batch_size(),
    class_mode = "categorical",
    subset = "validation"
)

main_logger.info("Create Model and Compile")
bias_model = BiasModel(
    num_classes = 2,
    image_shape = fair_face_dataset.get_image_shape(),
    bias_config = bias_config
)

bias_model.get_model().summary()

bias_model.compile_model()

main_logger.info("Train Model")

bias_model.train_model(
    train_generator = train_generator,
    validation_generator = validation_generator,
    fit_verbose = 1, 
    callback_verbose = 1
)

main_logger.info("Completed Model Training")

main_logger.info("Evaluate Model")

train_loss, train_acc, validation_loss, test_acc = bias_model.evaluate_model(
    train_generator = train_generator,
    validation_generator = validation_generator
)

main_logger.info("Train: %.3f, Test: %.3f" % (train_acc, test_acc))


main_logger.info("Save Model")

bias_model.save()