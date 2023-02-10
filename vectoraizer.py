#!/usr/bin/env python
# coding: utf-8

import os
import sys

import skimage.io

import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config
from vectoraizer.result_to_svg import result_to_svg
from vectoraizer.shapes import shapes_types

# CLI args
weights_path = sys.argv[1]
assert os.path.isfile(weights_path), 'Missing weights file'
input_img_path = sys.argv[2]
assert os.path.isfile(input_img_path), 'Missing input image file'
output_img_path = sys.argv[3] if 3 < len(sys.argv) else None

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

class InferenceConfig(Config):
    NAME = "shapes"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + len(shapes_types)
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    USE_MINI_MASK = False
    TRAIN_ROIS_PER_IMAGE = 32
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights
model.load_weights(weights_path, by_name=True)

class_names = ['BG'] + list(map(lambda s: s.name, shapes_types))

image = skimage.io.imread(input_img_path)
# image = skimage.color.rgba2rgb(image)
# pyplot.imshow(image, interpolation='nearest')
# plt.show()

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
svg = result_to_svg(image, r['rois'], r['masks'], r['class_ids'])

svg.save(output_img_path or 'vectoraized.svg')
