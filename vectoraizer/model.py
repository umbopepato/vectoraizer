import tensorflow as tf
import tensorflow.keras.layers as KL
import numpy as np

classes = {
    1: 'rect',
    2: 'circle',
    3: 'polygon',
}

classIndexes = {
    'rect': 1,
    'circle': 2,
    'polygon': 3
}

class VectorsLayer(KL.Layer):

    def __init__(self, config=None, **kwargs):
        super(VectorsLayer, self).__init__(**kwargs)
        self.config = config
        self.ellipses_layer = EllipsesLayer()
        self.rectangles_layer = RectanglesLayer(config)
        # self.paths_layer = PathsLayer()

    def call(self, inputs):
        """
        inputs[0]: detections [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)]]
        inputs[1]: mrcnn_mask [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
        inputs[2]: input_image_meta Dict
        """
        detections = inputs[0]
        mrcnn_mask = inputs[1]
        rectangles = self.rectangles_layer(inputs)

        return [rectangles]


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
        inputs[0]: detections [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)]]
        inputs[1]: mrcnn_mask [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
        inputs[2]: input_image_meta Dict
        """
        detections = inputs[0]

        # Gets the coordinates of rectangle detections in the original detections tensor
        detection_coords = filter_detections_by_class(detections, 'rect')
        rects = tf.gather_nd(detections, detection_coords)

        def detection_to_rect(detection):
            return tf.reshape([detection[1], detection[0], detection[3] - detection[1], detection[2] - detection[0]], [4])

        # Map each rectangle detection coordinates set to its result
        rects = tf.map_fn(
            fn=detection_to_rect,
            elems=rects
        )

        return detection_coords, rects


class EllipsesLayer(KL.Layer):
    """
    Returns: [(batch_index, detection_index), (x1, y1, width, height)]
    """

    def __init__(self, config=None, **kwargs):
        super(EllipsesLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        """
        inputs[0]: detections [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)]]
        inputs[1]: mrcnn_mask [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
        inputs[2]: input_image_meta Dict
        """
        detections = inputs[0]

        # Gets the coordinates of rectangle detections in the original detections tensor
        detection_coords = filter_detections_by_class(detections, 'ellipse')
        ellipses = tf.gather_nd(detections, detection_coords)

        def detection_to_ellipse(detection):
            return tf.reshape([detection[1], detection[0], detection[3] - detection[1], detection[2] - detection[0]], [4])

        # Map each rectangle detection coordinates set to its result
        ellipses = tf.map_fn(
            fn=detection_to_ellipse,
            elems=ellipses
        )

        return detection_coords, ellipses

