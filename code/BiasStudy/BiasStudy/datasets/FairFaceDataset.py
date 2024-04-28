import pandas as pd
from typing import Tuple

class FairFaceDataset():
    
    def __init__(self,
                 data_dir: str,
                 train_labels_csv_name: str,
                 validation_labels_csv_name: str,
                 under_sample : bool = True,
                 image_shape :  Tuple[int, int, int] = (224,224,3),
                 feature_column : str = "file",
                 output_col : str = "binary_race",
                 overwrite_sample_number : int = None):
        """
        This class expects the data_dir for Fair Face dataset to look like:
        data_dir
        ├── train
        │   └── image.jpg
        ├── val
        │   └── image.jpg
        ├── train_labels_csv_name.csv
        └── validation_labels_csv_name.csv
        
        @inproceedings{karkkainenfairface,
          title={FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation},
          author={Karkkainen, Kimmo and Joo, Jungseock},
          booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
          year={2021},
          pages={1548--1558}
        }
        
        Github: https://github.com/joojs/fairface
        """
        
        self.__data_dir = data_dir
        self.__train_labels_csv_name = train_labels_csv_name
        self.__validation_labels_csv_name = validation_labels_csv_name
        self.__under_sample = under_sample
        self.__image_shape = image_shape
        self.__feature_column = feature_column
        self.__output_col = output_col
        self.__overwrite_sample_number = overwrite_sample_number
        
        self.__setup_train__()
        self.__setup_validation__()

        if under_sample:
            self.__under_sample__(overwrite_sample_number)
        
    def __setup_train__(self):
        base_train_df = pd.read_csv("{}/{}".format(self.__data_dir, self.__train_labels_csv_name))
        
        base_train_df[self.__feature_column] = base_train_df[self.__feature_column].apply(lambda x: "{}/{}".format(self.__data_dir, x))
        
        # Remove entries belonging to one of the following 'Latino_Hispanic','Southeast Asian', 'Middle Eastern'
        self.train_df = base_train_df[
            ~base_train_df['race'].isin(['Latino_Hispanic','Southeast Asian', 'Middle Eastern'])
        ].copy()
        
        # Merge White and East Asian to light . Black and Indian to dark
        self.train_df[self.__output_col] = self.train_df.apply(
            lambda row: 'light' if row.race in ('White','East Asian') else 'dark', axis=1
        )
        
        self.train_df.drop(
            columns=['age', 'gender', 'race', 'service_test'], 
            axis=1, 
            inplace=True
        )
        
        self.train_df.reset_index(inplace=True, drop=True)

    def __setup_validation__(self):
        base_validation_df = pd.read_csv("{}/{}".format(self.__data_dir, self.__validation_labels_csv_name))
        base_validation_df[self.__feature_column] = base_validation_df[self.__feature_column].apply(lambda x: "{}/{}".format(self.__data_dir, x))
        
        # Remove entries belonging to one of the following 'Latino_Hispanic','Southeast Asian', 'Middle Eastern'
        self.validation_df = base_validation_df[~base_validation_df['race'].isin(['Latino_Hispanic','Southeast Asian', 'Middle Eastern'])].copy()
        
        # Merge White and East Asian to light . Black and Indian to dark
        self.validation_df[self.__output_col] = self.validation_df.apply(
            lambda row: 'light' if row.race in ('White','East Asian') else 'dark', axis=1
        )
        
        self.validation_df.drop(
            columns=['age', 'gender', 'race', 'service_test'], 
            axis=1, 
            inplace=True
        )
        
        self.validation_df.reset_index(inplace=True, drop=True)
    
    def __under_sample__(self, overwrite_sample_number=None):
        print("Before Under Sampling: " , self.train_df.binary_race.value_counts().to_dict())
        if overwrite_sample_number == None:
            self.imbalance_class_min = self.train_df.binary_race.value_counts().min()
        else:
            self.imbalance_class_min = overwrite_sample_number
        self.train_df = self.train_df.groupby(self.__output_col).sample(self.imbalance_class_min)
        print("After Under sampleing: " , self.train_df.binary_race.value_counts().to_dict())
        self.train_df.reset_index(inplace=True, drop=True)
    
    def get_train_pd(self) -> pd.DataFrame:
        return self.train_df.copy()
    
    def get_validation_pd(self) -> pd.DataFrame:
        return self.validation_df.copy()
    
    def get_data_dir(self) -> str:
        return self.__data_dir
    
    def is_under_sample(self) -> bool:
        return self.__under_sample
    
    def get_feature_col_name(self) -> str:
        return self.__feature_column
    
    def get_output_col_name(self) -> str:
        return self.__output_col
    
    def get_image_shape(self) -> Tuple[int, int, int]:
        return self.__image_shape
    
    def get_image_height(self) -> int:
        return self.__image_shape[1]
    
    def get_image_width(self) -> int:
        return self.__image_shape[0]
    
    def get_image_num_channels(self) -> int:
        return self.__image_shape[2]