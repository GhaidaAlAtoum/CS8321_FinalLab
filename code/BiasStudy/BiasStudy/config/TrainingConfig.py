import yaml
from pathlib import Path

class GeneralRunConfig():
    GENERAL_CONFIG_KEY = "general"
    GENERAL_BATCH_SIZE_KEY = "batch_size"
    OUTPUT_DIR_KEY = "output_dir"
    
    def __init__(self, config_dict: dict):
        self.__validate_general_config(config_dict)
        self.__load_general_config(config_dict)
    
    @classmethod
    def get_base_key(cls) -> str:
        return GeneralRunConfig.GENERAL_CONFIG_KEY
    
    def __validate_general_config(self, config_dict: dict):
        required_keys = [
            self.OUTPUT_DIR_KEY,
            self.GENERAL_BATCH_SIZE_KEY
        ]
        
        general_config = config_dict[self.GENERAL_CONFIG_KEY]
        
        if not all(k in general_config for k in required_keys):
            raise Exception(
                "{} is missing one/or more of the following :{}".format(
                    self.GENERAL_CONFIG_KEY, required_keys
                )
            )
            
        if type(general_config[self.GENERAL_BATCH_SIZE_KEY]) is not int:
            raise Exception(
                "{} is expected to be int - {} is given".format(self.GENERAL_BATCH_SIZE_KEY, type(general_config[self.GENERAL_BATCH_SIZE_KEY]))
            )
    
    def __load_general_config(self, config_dict: dict):
        general_config = config_dict[self.GENERAL_CONFIG_KEY]
        self.batch_size = general_config[self.GENERAL_BATCH_SIZE_KEY]
        self.output_dir = general_config[self.OUTPUT_DIR_KEY]
    
    def get_batch_size(self) -> int:
        return self.batch_size
    
    def get_output_dir(self) -> str:
        return self.output_dir

class TrainingDataSetConfig():
    TRAIN_DATASET_CONFIG_KEY = "train_dataset"
    TRAIN_DATASET_FILE_PATH_KEY = "fair_face_path"
    TRAIN_DATASET_NUM_SAMPLES = "number_samples"
    TRAIN_DATASET_OVERWRITE_SAMPLE_NUM = "overwrite_sample_number"
    
    @classmethod
    def get_base_key(cls) -> str:
        return TrainingDataSetConfig.TRAIN_DATASET_CONFIG_KEY
    
    def __init__(self, config_dict: dict):
        self.__validate_train_dataset_config(config_dict)
        self.__load_train_dataset_config(config_dict)
    
    def __validate_train_dataset_config(self, config_dict: dict):
        train_dataset_config = config_dict[self.TRAIN_DATASET_CONFIG_KEY]
        required_keys = [
            self.TRAIN_DATASET_FILE_PATH_KEY,
            self.TRAIN_DATASET_OVERWRITE_SAMPLE_NUM
        ]
        if not all(k in train_dataset_config for k in required_keys):
            raise Exception(
                "{} is missing one/or more of the following :{}".format(
                    self.TRAIN_DATASET_CONFIG_KEY, required_keys
                )
            )
        
        if type(train_dataset_config[self.TRAIN_DATASET_OVERWRITE_SAMPLE_NUM]) is not bool:
            raise Exception(
                "{} is expected to be bool - {} is given".format(Tself.RAIN_DATASET_OVERWRITE_SAMPLE_NUM, type(train_dataset_config[self.TRAIN_DATASET_OVERWRITE_SAMPLE_NUM]))
            )
            
        if train_dataset_config[self.TRAIN_DATASET_OVERWRITE_SAMPLE_NUM] == True:
            if self.TRAIN_DATASET_NUM_SAMPLES not in train_dataset_config:
                raise Exception("{} is missing {}".format(self.TRAIN_DATASET_CONFIG_KEY, self.TRAIN_DATASET_NUM_SAMPLES))
        
    def __load_train_dataset_config(self, config_dict: dict):
        train_dataset_config = config_dict[self.TRAIN_DATASET_CONFIG_KEY]
        self.fair_face_path = train_dataset_config[self.TRAIN_DATASET_FILE_PATH_KEY]
        self.overwrite_sample_number = train_dataset_config[self.TRAIN_DATASET_OVERWRITE_SAMPLE_NUM]
        if self.overwrite_sample_number :
            self.sample_number = train_dataset_config[self.TRAIN_DATASET_NUM_SAMPLES]
        else :
            self.sample_number = None
    
    def get_fair_face_path(self) -> str:
        return self.fair_face_path
    
    def get_sample_number(self): 
        return self.sample_number
    
    def is_overwrite_sample_number_enabled(self) -> bool:
        return self.overwrite_sample_number

class TrainingConfig():
    TRAIN_CONFIG_KEY = "train_config"
    
    NUM_EPOCHS_KEY = "num_epochs"
    EARLY_STOPPING_PATIENCE_KEY = "early_stopping_patience"
    CHECK_POINT_FREQ_KEY = "checkpoint_frequencey"
    LEARNING_RATE = "learning_rate"
    ENABLE_EARLY_STOPPING = "enable_early_stopping"
    
    def __init__(self, config_dict: dict):
        self.__validate_training_config(config_dict)
        self.__load_training_config(config_dict)
        
    def __validate_training_config(self, config_dict: dict):
        required_keys = [
            self.NUM_EPOCHS_KEY,
            self.EARLY_STOPPING_PATIENCE_KEY,
            self.CHECK_POINT_FREQ_KEY
        ]
        
        training_config = config_dict[self.TRAIN_CONFIG_KEY]
        if not all(k in training_config for k in (required_keys)):
            raise Exception(
                "{} Missing one/or more of the following : {}".format(self.TRAIN_CONFIG_KEY, required_keys)
            )

    def __load_training_config(self, config_dict: dict):
        training_config = config_dict[self.TRAIN_CONFIG_KEY]
        self.epoch_num = training_config[self.NUM_EPOCHS_KEY]
        self.early_stopping_patience = training_config[self.EARLY_STOPPING_PATIENCE_KEY]
        self.checkpoint_frequencey = training_config[self.CHECK_POINT_FREQ_KEY]
        
        self.learning_rate = training_config.get(self.LEARNING_RATE, 0.001)
        self.enable_early_stopping = training_config.get(self.ENABLE_EARLY_STOPPING, True)
    
    def get_epoch_num(self) -> int:
        return self.epoch_num
    
    def get_early_stopping_patience(self) -> int:
        return self.early_stopping_patience
    
    def get_checkpoint_frequencey(self) -> int:
        return self.checkpoint_frequencey
    
    def get_learning_rate(self) -> float:
        return self.learning_rate
    
    def is_early_stopping_enabled(self) -> bool:
        return self.enable_early_stopping
            
    @classmethod
    def get_base_key(cls) -> str:
        return TrainingConfig.TRAIN_CONFIG_KEY

class FlatLayerConfig():
    NUM_UNITS = "num_units"
    
    def __init__(self, layer_name: str, layer_config: dict):
        self.layer_name = layer_name
        self.__validate(layer_config)
        self.__load(layer_config)
    
    def __validate(self, layer_config: dict):
        if self.NUM_UNITS not in layer_config:
            raise Exception("{} is missing {}".format(self.layer_name, self.NUM_UNITS))
    
    def __load(self, layer_config: dict):
        self.num_units = layer_config[self.NUM_UNITS]
        
    def get_layer_name(self) -> str:
        return self.layer_name
    
    def get_num_units(self) -> int:
        return self.num_units

class ConvBlockConfig():
    NUM_CONV_LAYERS = "num_conv_layers"
    NUM_FILTERS_PER_CONV = "num_filters"
    KERNEL_SIZE = "kernel_size"
    DROPOUT_ENABLED = "dropout_enaled"
    
    def __init__(self, block_name: str, conv_block_config: dict):
        self.block_name = block_name
        self.conv_block_config = conv_block_config
        self.__validate_conv_block_config()
        self.__load_conv_block_config()
    
    def __validate_conv_block_config(self):
        required_keys = [
            self.NUM_CONV_LAYERS,
            self.NUM_FILTERS_PER_CONV,
            self.KERNEL_SIZE
        ]
         
        if not all(k in self.conv_block_config for k in (required_keys)):
            raise Exception(
                "{} Missing one/or more of the following : {}".format(self.block_name, required_keys)
            )

    def __load_conv_block_config(self):
        self.num_conv_layers = self.conv_block_config[self.NUM_CONV_LAYERS]
        self.num_filters = self.conv_block_config[self.NUM_FILTERS_PER_CONV]
        self.kernel_size = self.conv_block_config[self.KERNEL_SIZE]
        self.dropout_enaled = self.conv_block_config.get(self.DROPOUT_ENABLED, False)
        
    def get_block_name(self) -> str:
        return self.block_name
    
    def get_num_conv_layers(self) -> int:
        return self.num_conv_layers
    
    def get_num_filters(self) -> int:
        return self.num_filters
        
    def get_kernel_size(self) -> int:
        return self.kernel_size
    
    def is_dropout_enabled(self) -> bool:
        return self.dropout_enaled
    

class ModelConfig():
    MODEL_NAME_KEY = "model_name"
    CONV_BLOCKS_KEY = "conv_blocks"
    FLATT_LAYERS_KEY = "flatt_layers"
    
    def __init__(self, model_config_dict: dict):
        self.__validate(model_config_dict)
        self.__load(model_config_dict)
        
    def __validate(self, model_config_dict: dict):
        required_keys = [ 
            self.CONV_BLOCKS_KEY,
            self.MODEL_NAME_KEY
        ]
        
        if not all(k in model_config_dict for k in (required_keys)):
            raise Exception(
                "Missing one/or more of the following : {}".format(required_keys)
            )
        
        self.flat_layers_exist = False
        if self.FLATT_LAYERS_KEY in model_config_dict:
            self.flat_layers_exist = True
        
    def __load(self, model_config_dict: dict):
        self.modle_name = model_config_dict[self.MODEL_NAME_KEY]
        self.conv_layers_configs = {}
        for block_num, (block_name, block_config) in enumerate(model_config_dict[self.CONV_BLOCKS_KEY].items()):
            self.conv_layers_configs[block_num] = ConvBlockConfig(block_name, block_config)
            
        if not self.conv_layers_configs:
            raise Exception("No Conv layers are configured")
        
        self.flat_layers_configs = {}
        if self.flat_layers_exist:
            for flat_layer_num, (flat_layer_name, flat_layer_config) in enumerate(model_config_dict[self.FLATT_LAYERS_KEY].items()):
                self.flat_layers_configs[flat_layer_num] = FlatLayerConfig(flat_layer_name, flat_layer_config)
                
    def get_conv_layers(self) -> dict:
        return self.conv_layers_configs
    
    def get_flat_layers(self) -> dict:
        return self.flat_layers_configs
    
    def is_flat_layers_defined(self) -> bool:
        return self.flat_layers_exist
    
    def get_model_name(self) -> str:
        return self.modle_name 
    
class BiasStudyConfig():
    RUN_CONFIG_KEY = "run_config"
    MODEL_CONFIG_NAME = "model_config"
    
    def __init__(self, config_path: str):
        self.config_dict = BiasStudyConfig.read_config_yaml(config_path)
        self.__validate_config()
        self.general_config = GeneralRunConfig(self.run_config_dict)
        self.train_dataset_config = TrainingDataSetConfig(self.run_config_dict)
        self.train_config = TrainingConfig(self.run_config_dict)
        self.modle_config = ModelConfig(self.model_config_dict)
        self.save_config_yaml()
        
    def __validate_config(self):
        if self.RUN_CONFIG_KEY not in self.config_dict:
            raise Exception("Missing {}".format(self.RUN_CONFIG_KEY))
            
        if self.MODEL_CONFIG_NAME not in self.config_dict:
            raise Exception("Missing {}".format(self.MODEL_CONFIG_NAME))
        
        self.run_config_dict = self.config_dict[self.RUN_CONFIG_KEY]
        self.model_config_dict = self.config_dict[self.MODEL_CONFIG_NAME]
                
        required_run_keys = [
            GeneralRunConfig.get_base_key(),
            TrainingDataSetConfig.get_base_key(),
            TrainingConfig.get_base_key()
        ]
        
        if not all(k in self.run_config_dict for k in (required_run_keys)):
            raise Exception(
                "Missing one/or more of the following : {}".format(required_run_keys)
            )
        
    def get_general_config(self) -> GeneralRunConfig:
        return self.general_config
    
    def get_train_dataset_config(self) -> TrainingDataSetConfig:
        return self.train_dataset_config
        
    
    def get_train_config(self) -> TrainingConfig:
        return self.train_config
    
    def get_model_config(self) -> ModelConfig:
        return self.modle_config
    
    @classmethod
    def read_config_yaml(cls, file_path: str) -> dict:
        with open(file_path, 'r') as stream:
            config = yaml.safe_load(stream)
        
        return config
    
    def save_config_yaml(self):
        output_dir_name = self.general_config.get_output_dir()
        model_name = self.modle_config.get_model_name()
        config_dir = "{}/{}/config/".format(output_dir_name, model_name)
        Path(config_dir).mkdir(parents=True, exist_ok=True)
        
        config_path = "{}/config.yaml".format(config_dir)
        with open(config_path, "w", encoding = "utf-8") as yaml_file:
            dump = yaml.dump(
                self.config_dict, 
                default_flow_style = False,
                allow_unicode = True, 
                encoding = None
            )
            yaml_file.write(dump)
