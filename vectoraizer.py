#!/usr/bin/env python
# coding: utf-8

import os
import sys
import matplotlib.pyplot as plt
from matplotlib import pyplot
import skimage.io
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config
from vectoraizer.result_to_svg import result_to_svg
import tensorflow as tf

print('Tensorflow version: {}'.format(tf.__version__))

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/colab_weights.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(Config):
    NAME = "shapes"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 3  # background + 3 shapes
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    USE_MINI_MASK = False
    TRAIN_ROIS_PER_IMAGE = 32
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5
    RUN_EAGERLY = True


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'rect', 'circle', 'polygon']

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image_path = os.path.join(IMAGE_DIR, file_names[0])
print(image_path)
image = skimage.io.imread(image_path)
# image = skimage.color.rgba2rgb(image)
# pyplot.imshow(image, interpolation='nearest')
# plt.show()

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]

print(r)

visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
# result_to_svg(image, r['rois'], r['masks'], r['class_ids'])
