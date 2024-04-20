from keras.models import Model
from .PredictionToolKit import *
from matplotlib import pyplot as plt
import matplotlib as mpl
from tensorflow import keras
import tensorflow as tf

# https://keras.io/examples/vision/grad_cam/

from typing import Callable
import pathlib

def make_gradcam_heatmap(
    pre_processor_function: Callable,
    img_path: str, 
    model: Model, 
    last_conv_layer_name: str, 
    pred_index=None
):
    img_array = load_image_as_array(img_path)
    img_array = pre_processor_function(img_array)
    img_array = img_array.reshape(1,224,224,3)
    
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(
    title: str,
    image_id: int,
    img_path: str, 
    heatmap: np.array, 
    save: bool = True,
    save_dir: str = "./", 
    alpha=0.4, 
    ax: plt.axis = None, 
    fig_size: (int,int) = (5,5)
):
    if ax is None:
        plt.figure(figsize=fig_size)
        ax = plt.gca()
        
    # Load the original image
    img = load_image_as_array(img_path)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    if save:
        output_dir = "{}/output_images".format(save_dir)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True) 
        image_path = "{}/{}_HeatMap.png".format(output_dir, image_id)
        superimposed_img.save(image_path)

    # Display Grad CAM
    ax.imshow(superimposed_img)
    ax.set_title(title)