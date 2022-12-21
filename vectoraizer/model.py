import tensorflow as tf
import tensorflow.keras.layers as KL
import numpy as np
from mrcnn.utils import unmold_mask, unmold_detections_graph

classes = {
    1: 'rect',
    2: 'ellipse',
    3: 'path',
}

classIndexes = {
    'rect': 1,
    'ellipse': 2,
    'path': 3
}

class VectorsLayer(KL.Layer):

    def __init__(self, config=None, **kwargs):
        super(VectorsLayer, self).__init__(**kwargs)
        self.config = config
        self.rectangles_layer = RectanglesLayer(config)
        self.ellipses_layer = EllipsesLayer(config)
        # self.paths_layer = PathsLayer()

    def unmold_inputs(self, inputs):
        return tf.map_fn(
            lambda x: unmold_detections_graph(x[0], x[1], x[2][1:4], x[2][4:7], x[2][7:11]),
            (inputs[0], inputs[1], inputs[3], tf.constant([1]))
        )

    def call(self, inputs):
        """
        inputs[0]: detections [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)]]
        inputs[1]: mrcnn_mask [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
        inputs[2]: input_image [None, None, config.IMAGE_SHAPE[2]]
        inputs[3]: input_image_meta Dict
        """
        # inputs = self.unmold_inputs(inputs) # final_rois, final_class_ids, final_scores, final_masks
        # print('Inputs: {}'.format(inputs))
        rectangles = self.rectangles_layer(inputs)
        ellipses = self.ellipses_layer(inputs)

        return rectangles, ellipses


def filter_detections_by_class(detections, class_name):
    return tf.where(detections[:,:,4] == classIndexes[class_name])


class RectanglesLayer(KL.Layer):
    """
    Returns: [(batch_index, detection_index), (x1, y1, width, height)]
    """

    def __init__(self, config=None, **kwargs):
        super(RectanglesLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        """
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # boxes = inputs[0]
        # class_ids = inputs[1]
        detections = inputs[0]
        mrcnn_mask = inputs[1]
        input_image = inputs[2]
        input_image_meta = inputs[3]

        # Gets the coordinates of rectangle detections in the original detections tensor
        detection_coords = filter_detections_by_class(detections, 'rect')
        rects = tf.gather_nd(detections, detection_coords)

        def detection_to_rect(detection):
            print('Rect')
            return tf.reshape([detection[1], detection[0], detection[3] - detection[1], detection[2] - detection[0]], [4])

        # Map each rectangle detection coordinates set to its result
        rects = tf.map_fn(
            fn=detection_to_rect,
            elems=rects
        )

        return detection_coords, rects


class EllipsesLayer(KL.Layer):
    """
    Returns: [(batch_index, detection_index), (cx, cy, rx, ry)]
    """

    def __init__(self, config=None, **kwargs):
        super(EllipsesLayer, self).__init__(**kwargs)
        self.config = config
        self.conv2d_layer = KL.Conv2D(32, kernel_size=(3, 3), activation='relu')
        self.max_pooling_2d_layer = KL.MaxPooling2D(pool_size=(2, 2))
        self.flatten_layer = KL.Flatten()
        self.dense1_layer = KL.Dense(128, activation='relu')
        self.dense2_layer = KL.Dense(4, activation='linear')

    def call(self, inputs):
        # class_ids = inputs[1]
        # masks = inputs[3]
        detections = inputs[0]
        mrcnn_mask = inputs[1]
        input_image = inputs[2]
        input_image_meta = inputs[3]

        # Gets the coordinates of ellipse detections in the original detections tensor
        detection_coords = filter_detections_by_class(detections, 'ellipse')
        masks = tf.gather_nd(mrcnn_mask, detection_coords)

        x = self.conv2d_layer(masks)
        x = self.max_pooling_2d_layer(x)
        x = self.flatten_layer(x)
        x = self.dense1_layer(x)
        return detection_coords, self.dense2_layer(x)
