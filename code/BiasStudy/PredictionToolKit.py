from PIL import Image
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from keras.models import Model
from .PredictionResults import PredictionResults

def load_image_as_array(path, size=(224, 224)):
    img = Image.open(path)
    img = img.resize(size)
    return np.array(img).astype(float)

def calculate_similarity(embeddings_array: np.array, embedding_1_idx: int, embedding_2_idx: int):
    # Others would calculate similarity just cosine not 1-cosine
    return 1 - cosine(embeddings_array[embedding_1_idx], embeddings_array[embedding_2_idx])

def predict(
    feature_name: str,
    unique_image_id_col_name: str,
    file_path_col_name: str,
    pair_id_col_name: str,
    y_col: str,
    image_dir: str,
    dataset: pd.DataFrame,
    model: Model,
    verbose: int,
    best_threshold_method: str
) -> PredictionResults:
    y_true_dict = {}
    y_pred_dict = {}
    
    variations = []
    for variation in dataset[feature_name].unique():
        variations.append(variation)
        y_true_dict[variation] = []
        y_pred_dict[variation] = []
    
    if len(variations) > 2:
        raise Exception("Current implementation doesn't support more than 2 group variations")

    y_true = {variations[0]: [], variations[1]: []}
    y_pred = {variations[0]: [], variations[1]: []}
    
    # Prepare Image Embeddings Array
    number_of_unique_imgs = dataset[unique_image_id_col_name].unique().shape[0]
    model_output_shape = model.output.shape[1]
    image_embeddings = np.zeros([number_of_unique_imgs, model_output_shape])
    
    for gid, group in dataset.groupby(pair_id_col_name):
        if len(group) > 2:
            raise Exception("Current implementation doesn't support more than 2 images per group")
        sample_from_group = group.sample(1)
        is_matching_group = sample_from_group[y_col].item()
        group_variation = sample_from_group[feature_name].item()
        
        y_true_dict[group_variation].append(is_matching_group)
        image_ids_in_group = group[unique_image_id_col_name].tolist()
        for rowIdx, row in group.iterrows():
            image_id = row[unique_image_id_col_name]
            if not np.any(image_embeddings[image_id]):
                print(".", end='')
                image_path = "{}/{}".format(image_dir, row[file_path_col_name])
                image_array = load_image_as_array(image_path)
                image_array = np.expand_dims(image_array, axis=0)
                image_embeddings[image_id] = model.predict(image_array, verbose=verbose)
        
        similarity_score = calculate_similarity(
            embeddings_array = image_embeddings, 
            embedding_1_idx = image_ids_in_group[0], 
            embedding_2_idx = image_ids_in_group[1]
        )
        
        y_pred_dict[group_variation].append(similarity_score)
    
    return PredictionResults(
        y_true =  y_true_dict, 
        y_pred = y_pred_dict,
        embeddings = image_embeddings,
        best_threshold_method = best_threshold_method,
        model_name = model.name
    )