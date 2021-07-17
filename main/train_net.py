# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from IPython.display import Image, display
import matplotlib.cm as cm


# %%
# Define some constants
image_size=(300, 300)
image_input=(300, 300, 3)
BATCH_SIZE=32


# %%
get_ipython().system('cp "/content/drive/MyDrive/Colab Notebooks/Deep Learning - Final Project/main_dataset.zip" .')
get_ipython().system('unzip main_dataset.zip')


# %%
# Create data generator that reads image from directory and make some augmentation on data
image_generator = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255.,
    rotation_range=45,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.3,    
)


# %%
# Divide dataset into trainin set and validation
training_set = image_generator.flow_from_directory(
    "/content/dataset",
    target_size=image_size,
    batch_size=BATCH_SIZE,
    subset='training',
    shuffle=True
)

validation_set = image_generator.flow_from_directory(
    "/content/dataset",
    target_size=image_size,
    batch_size=BATCH_SIZE,
    subset='validation',
    shuffle=True
)


# %%
# Explore dataset
c = 0
for batch, outputs in training_set:
    print(outputs[0])
    plt.imshow(batch[0])
    plt.show()

    c += 1
    if c== 10:
        break


# %%
# Load pretrained model
conv_base = Xception(
    include_top=False,
    weights='imagenet',
    input_shape=image_input,
    pooling='avg'
)

conv_base.summary()


# %%
# Create model
model = keras.Sequential()

model.add(keras.layers.InputLayer(input_shape=image_input))
model.add(conv_base)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(units=3, activation="sigmoid"))

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

conv_base.trainable = False

model.summary()


# %%
# Fine tune classifier
history = model.fit(
    training_set,
    validation_data=validation_set,
    epochs=5,
)


# %%
# Make two last convolutional block trainable
conv_base.trainable = True


# %%
for layer in conv_base.layers:
    if layer.name[0:7] == 'block14' or layer.name[0:7] == 'block13': 
        layer.trainable = True
    else:
        layer.trainable = False


# %%
# Compile model to apply changes
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()


# %%
# Train model 
history = model.fit(
    training_set,
    validation_data=validation_set,
    epochs=5,
)


# %%
# Display gradcam 

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


# %%

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
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


# %%
def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))


# %%
def display_gradcam(image_path, last_conv_layer_name):
  # Prepare image
  img_array = get_img_array(img_path, size=image_size) / 255.

  # Make model  # Remove last layer's softmax
  # model.layers[-1].activation = None

  # Print what the top predicted class is
  preds = model.predict(img_array)
  print("Predicted:", preds)

  # Generate class activation heatmap
  heatmap = make_gradcam_heatmap(img_array, conv_base, last_conv_layer_name)

  # Display heatmap
  plt.matshow(heatmap)
  plt.show()

  save_and_display_gradcam(img_path, heatmap)


# %%
selected_conv = [
  "block14_sepconv2_act",
  "block13_sepconv2_act",
  "block12_sepconv2_act",
  "block11_sepconv2_act",
  "block10_sepconv2_act",
  "block9_sepconv2_act",
  "block8_sepconv2_act",
  "block7_sepconv2_act",
  "block6_sepconv2_act",
  "block5_sepconv2_act",
  "block4_sepconv2_act",
  "block3_sepconv2_act"
]
last_conv_layer_name = 'block14_sepconv2_act'


# %%
def display_gradcam_multiple(img_path, selected_conv):
  for i in selected_conv:
      
    img_array = get_img_array(img_path, size=image_size) / 255.

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, conv_base, i)

    # Display heatmap
    plt.matshow(heatmap)
    plt.show()

    save_and_display_gradcam(img_path, heatmap)


# %%
# predict_samples(img_path)
display_gradcam_multiple(img_path, selected_conv)


# %%


img_path = keras.utils.get_file(
    "8754test.jpg", "https://gdb.rferl.org/EDFF2936-65DF-4E60-98D4-613A6DED7917_w1023_r1_s.jpg"
)


img_array = get_img_array(img_path, size=image_size) / 255.

# Print what the top predicted class is
preds = model.predict(img_array)
print("Predicted:", preds)

display_gradcam_multiple(img_path, selected_conv)


# %%
img_path = keras.utils.get_file(
    "test0351.jpg", "https://i.ibb.co/zrZQ5gq/944103.jpg"
)


img_array = get_img_array(img_path, size=image_size) / 255.


# Print what the top predicted class is
preds = model.predict(img_array)
print("Predicted:", preds)

display_gradcam_multiple(img_path, selected_conv)


# %%

img_path = keras.utils.get_file(
    "0554df.jpg", "https://i.ibb.co/P9FWHrP/43388-626.jpg"
)


img_array = get_img_array(img_path, size=image_size) / 255.

# Make model  # Remove last layer's softmax
# model.layers[-1].activation = None

# Print what the top predicted class is
preds = model.predict(img_array)
print("Predicted:", preds)

display_gradcam_multiple(img_path, selected_conv)


# %%


img_path = keras.utils.get_file(
    "0546tyghddf.jpg", "https://i.ibb.co/jMRvnFF/317407-475.jpg"
)


img_array = get_img_array(img_path, size=image_size) / 255.

# Print what the top predicted class is
preds = model.predict(img_array)
print("Predicted:", preds)

display_gradcam_multiple(img_path, selected_conv)


# %%
model.save("/content/drive/MyDrive")

# %%
conv_base.save("/content/drive/MyDrive")


