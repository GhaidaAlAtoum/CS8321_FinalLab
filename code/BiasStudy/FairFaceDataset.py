import pandas as pd

class FairFace():
    DATA_DIR = "/notebooks/data/fairface"
    TRAIN_LABELS = "fairface_label_train.csv"
    VAL_LABELS = "fairface_label_val.csv"
    IMAGE_HEIGHT = 224
    IMAGE_WIDTH = 224
    X_COL = "file"
    Y_COL = "binary_race"
    NUM_CHANNELS = 3
    
    def __init__(self,
                 under_sample=True, 
                 overwrite_sample_number=None):
        self.base_train_df = pd.read_csv("{}/{}".format(self.DATA_DIR, self.TRAIN_LABELS))
        self.base_train_df[self.X_COL] = self.base_train_df[self.X_COL].apply(lambda x: "{}/{}".format(self.DATA_DIR, x))
        
        self.base_test_df = pd.read_csv("{}/{}".format(self.DATA_DIR, self.VAL_LABELS))
        self.base_test_df[self.X_COL] = self.base_test_df[self.X_COL].apply(lambda x: "{}/{}".format(self.DATA_DIR, x))
        self.__setup_train__()
        self.__setup_test__()

        if under_sample:
            self.__under_sample__(overwrite_sample_number)
        
    def __setup_train__(self):
        # Remove entries belonging to one of the following 'Latino_Hispanic','Southeast Asian', 'Middle Eastern'
        self.train_df = self.base_train_df[
            ~self.base_train_df['race'].isin(['Latino_Hispanic','Southeast Asian', 'Middle Eastern'])
        ].copy()
        
        # Merge White and East Asian to light . Black and Indian to dark
        self.train_df[self.Y_COL] = self.train_df.apply(
            lambda row: 'light' if row.race in ('White','East Asian') else 'dark', axis=1
        )
        
        self.train_df.drop(
            columns=['age', 'gender', 'race', 'service_test'], 
            axis=1, 
            inplace=True
        )

    def __setup_test__(self):
        # Remove entries belonging to one of the following 'Latino_Hispanic','Southeast Asian', 'Middle Eastern'
        self.test_df = self.base_test_df[
            ~self.base_test_df['race'].isin(['Latino_Hispanic','Southeast Asian', 'Middle Eastern'])
        ].copy()
        
        # Merge White and East Asian to light . Black and Indian to dark
        self.test_df[self.Y_COL] = self.test_df.apply(
            lambda row: 'light' if row.race in ('White','East Asian') else 'dark', axis=1
        )
        
        self.test_df.drop(
            columns=['age', 'gender', 'race', 'service_test'], 
            axis=1, 
            inplace=True
        )
        
        fair_face.test_df.reset_index(inplace=True)
    
    def __under_sample__(self, overwrite_sample_number=None):
        print("Before: " , self.train_df.binary_race.value_counts().to_dict())
        if overwrite_sample_number == None:
            self.imbalance_class_min = self.train_df.binary_race.value_counts().min()
        else:
            self.imbalance_class_min = overwrite_sample_number
        self.train_df = self.train_df.groupby(self.Y_COL).sample(self.imbalance_class_min)
        print("After: " , self.train_df.binary_race.value_counts().to_dict())
        