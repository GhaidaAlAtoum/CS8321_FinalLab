from ..config.TrainingConfig import GeneralRunConfig, TrainingDataSetConfig, TrainingConfig, BiasStudyConfig, ModelConfig, FlatLayerConfig, ConvBlockConfig
from . import TrainingLogger
from typing import Tuple
from pathlib import Path

import tensorflow as tf
import tensorflow.keras
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import datasets, layers, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, EarlyStopping, LearningRateScheduler
from keras.preprocessing.image import DataFrameIterator
from tensorflow.keras.layers import Input, Conv2D 
from tensorflow.keras.layers import MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

# https://stackoverflow.com/a/72746245
class EpochModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):

    def __init__(self,
                 filepath,
                 logging_output_dir,
                 frequency=1,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 options=None,
                 **kwargs):
        super(EpochModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,
                                                   mode, "epoch", options)
        self.epochs_since_last_save = 0
        self.frequency = frequency
        self.logger = TrainingLogger.setup_logger(self.__class__.__name__, logging_output_dir)
        
    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        # pylint: disable=protected-access
        if self.epochs_since_last_save % self.frequency == 0:
            self._save_model(epoch=epoch, batch=None, logs=logs)
            self.logger.info("Saving Model Checkpoint at {}".format(epoch))

    def on_train_batch_end(self, batch, logs=None):
        pass
    

class BiasModel():
    def __init__(
        self,
        num_classes: int,
        bias_config: BiasStudyConfig,
        image_shape: Tuple[int, int, int] = (224,224,3)
    ):
        self.num_classes = num_classes
        self.image_shape = image_shape
        self.bias_config = bias_config
        self.logger = TrainingLogger.setup_logger(
            self.__class__.__name__, 
            self.bias_config.get_general_config().get_output_dir()
        )
        self.__create_model()

    def __create_model(self):
        model_config = self.bias_config.get_model_config()
        self.model_name = model_config.get_model_name()
        self.model = keras.Sequential(name= self.model_name)
        self.model.add(Input(shape=self.image_shape))
        
        for block_num, conv_block in sorted(model_config.get_conv_layers().items()):
            for conv_layer_num in range(0, conv_block.get_num_conv_layers()):
                self.model.add(
                    Conv2D(
                        filters = conv_block.get_num_filters(),
                        kernel_size = conv_block.get_kernel_size(),
                        padding = 'same',
                        activation = 'relu',
                        name = "block{}_conv{}".format(block_num, conv_layer_num)
                    )
                )
                
            self.model.add(
                MaxPool2D(pool_size =2, strides =2, padding ='same', name="block{}_pool".format(block_num))
            )
            if conv_block.is_dropout_enabled():
                self.model.add(Dropout(rate = 0.2))
        
        self.model.add(Flatten(name = "flatten"))
        
        if model_config.is_flat_layers_defined():
            for flat_layer_num, flat_config in sorted(model_config.get_flat_layers().items()):
                self.model.add(Dense(units = flat_config.get_num_units(), activation ='relu', name = flat_config.get_layer_name()))

        self.model.add(
            Dense(units = self.num_classes, activation ='softmax', name = "prediction")
        )
    
    def compile_model(self):
        self.logger.info("Compiling the model")
        loss = None
        if self.num_classes > 1:
            loss = "categorical_crossentropy"
        else:
            loss = "mean_squared_error"
        train_config = self.bias_config.get_train_config()
        self.model.compile(
            optimizer = Adam(learning_rate=train_config.get_learning_rate()),
            loss = loss,
            metrics = ['accuracy']
        )
    
    def train_model(
        self,
        train_generator,
        validation_generator,
        use_multiprocessing = False,
        fit_verbose: int = 1, 
        callback_verbose: int = 0
    ):
        train_config = self.bias_config.get_train_config()
        checkpoint_dir, logging_dir = self.__create_train_dirs()
        
        checkpoint_file_path = checkpoint_dir + "cp-{epoch:02d}.ckpt"
        checkpoint_callback = EpochModelCheckpoint(
            filepath = checkpoint_file_path,
            logging_output_dir = self.bias_config.get_general_config().get_output_dir(),
            monitor = 'val_accuracy',
            frequencey = train_config.get_checkpoint_frequencey(),
            verbose = callback_verbose
        )

        log_csv_file_path = logging_dir + "logs{}.csv".format(self.model_name)
        log_csv_callback = CSVLogger(
            filename = log_csv_file_path,
            append = True
        )

        self.logger.info("Early Stopping Defined with patience {}".format(train_config.get_early_stopping_patience()))
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=train_config.get_early_stopping_patience(), 
            mode='min',
            min_delta=0.0001,
            verbose = callback_verbose
        )
        
        callbacks_list = [checkpoint_callback, log_csv_callback]
        if train_config.is_early_stopping_enabled():
            callbacks_list.append(early_stopping)
            
        self.logger.info("Start Fitting the model")
        self.model.fit(
            train_generator,
            epochs = train_config.get_epoch_num(),
            validation_data = validation_generator,
            callbacks = callbacks_list,
            verbose = fit_verbose,
            use_multiprocessing = use_multiprocessing
        )
        self.logger.info("Completed Fitting the model")
    
    def evaluate_model(self, train_generator, validation_generator):
        train_loss, train_acc = self.model.evaluate(train_generator)
        validation_loss, test_acc = self.model.evaluate(validation_generator)
        return train_loss, train_acc, validation_loss, test_acc
    
    def save(self):
        output_dir_name = self.bias_config.get_general_config().get_output_dir()
        model_dir = "{}/{}/model/".format(output_dir_name, self.model_name)
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        weight_dir = "{}/{}/weights/".format(output_dir_name, self.model_name)
        Path(weight_dir).mkdir(parents=True, exist_ok=True)
        
        self.model.save(model_dir + "model.h5")
        self.model.save_weights(weight_dir + "weights.h5")
        
    def __create_train_dirs(self):
        output_dir_name = self.bias_config.get_general_config().get_output_dir()
        checkpoint_dir = "{}/{}/checkpoints/".format(output_dir_name, self.model_name)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        logging_dir = "{}/{}/csv_logging/".format(output_dir_name, self.model_name)
        Path(logging_dir).mkdir(parents=True, exist_ok=True)

        return checkpoint_dir, logging_dir
    
    def get_model(self):
        return self.model