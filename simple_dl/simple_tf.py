import os
import glob
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
from PIL import ImageChops

IMAGE_HOME_DIR = "/root/stage1_train"
TRAIN_SPLIT = 0.8
IMG_SIZE = (256, 256)
NUM_CLASSES = 2
BATCH_SIZE = 6
EPOCHS = 50

def generate_combined_mask(path_list):
    for img_path in tqdm(path_list):
        searchpath = os.path.join(img_path, "masks", "*.png")
        masklist = glob.glob(searchpath)
        firstmask = Image.open(masklist[0], 'r')
        img_w, img_h = firstmask.size
        background_image = Image.new('L', (img_w, img_h), 0)
        for m in masklist:
            background_image = ImageChops.lighter(background_image, Image.open(m))
        new_mask_dir = os.path.join(img_path, "masks2")
        os.makedirs(new_mask_dir, exist_ok=True)
        new_mask_path = os.path.join(new_mask_dir, "newmask.png")
        background_image.save(new_mask_path)

image_dirs = [d.path for d in os.scandir(IMAGE_HOME_DIR) if d.is_dir()]
#generate_combined_mask(image_dirs)

image_list = [f.path for i in image_dirs for f in os.scandir(os.path.join(i, "images")) if f.is_file()]
masks_list = [f.path for i in image_dirs for f in os.scandir(os.path.join(i, "masks2")) if f.is_file()]

class Nucleii(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            tgt_array = np.array(img) / 255
            y[j] = np.expand_dims(tgt_array, 2)
        return x, y

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    previous_block_activation = x  # Set aside residual

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

# Split our img paths into a training and a validation set
val_samples = int(len(image_list) * (1 - TRAIN_SPLIT))
random.Random(1337).shuffle(image_list)
random.Random(1337).shuffle(masks_list)
train_image_list = image_list[:-val_samples]
train_masks_list = masks_list[:-val_samples]
val_image_list = image_list[-val_samples:]
val_masks_list = masks_list[-val_samples:]

# Instantiate data Sequences for each split
train_gen = Nucleii(BATCH_SIZE, IMG_SIZE, train_image_list, train_masks_list)
val_gen = Nucleii(BATCH_SIZE, IMG_SIZE, val_image_list, val_masks_list)

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()
# Build model
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    model = get_model(IMG_SIZE, NUM_CLASSES)
    model.summary()
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

callbacks = [
    keras.callbacks.ModelCheckpoint("nucleii_segmentation.h5", save_best_only=True)
]

#first_batch = train_gen.__getitem__(0)

# Train the model, doing validation at the end of each epoch.
model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=callbacks)