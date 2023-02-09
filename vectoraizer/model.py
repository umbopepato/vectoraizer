import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.backend as K

from mrcnn.utils import smooth_l1_loss, unmold_detections_graph, compute_iou, compute_iou_graph, ragged_slice, \
    batch_slice
from vectoraizer.shapes import Rectangle, Ellipse

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
            lambda x: unmold_detections_graph(x[0], x[1], self.config.IMAGE_SHAPE, x[2][4:7], x[2][7:11]),
            (inputs[0], inputs[1], inputs[3], tf.zeros(tf.shape(inputs[0])[0])),
            fn_output_signature=(tf.float32, tf.int32, tf.float32, tf.float32, tf.int32)
        )

    def call(self, inputs):
        """
        inputs[0]: detections [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)]]
        inputs[1]: mrcnn_mask [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
        inputs[2]: input_image [None, None, config.IMAGE_SHAPE[2]]
        inputs[3]: input_image_meta Dict
        """
        inputs = self.unmold_inputs(inputs)
        rectangles = self.rectangles_layer(inputs)
        ellipses = self.ellipses_layer(inputs)

        return rectangles, ellipses


def prepare_class_detections(class_ids, class_id):
    return tf.where(class_ids == class_id)


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
        # detections = inputs[0]
        boxes = inputs[0]
        class_ids = inputs[1]
        masks = inputs[3]

        # Gets the coordinates of rectangle detections in the original detections tensor
        detection_coords = prepare_class_detections(class_ids, Rectangle.class_id)
        rects = tf.gather_nd(boxes, detection_coords)

        # def detection_to_rect(detection):
        #     return tf.reshape([detection[1], detection[0], detection[3] - detection[1], detection[2] - detection[0]], [4])
        #
        # Map each rectangle detection coordinates set to its result
        # rects = tf.map_fn(
        #     fn=detection_to_rect,
        #     elems=rects
        # )

        return Rectangle.class_id, detection_coords, rects[:, :4]


class EllipsesLayer(KL.Layer):
    """
    Returns: [(batch_index, detection_index), (cx, cy, rx, ry)]
    """

    def __init__(self, config=None, **kwargs):
        super(EllipsesLayer, self).__init__(**kwargs)
        self.config = config
        self.conv2d_layer = KL.Conv2D(32, kernel_size=(3, 3), activation='relu', name='ellipse_conv2d')
        self.max_pooling_2d_layer = KL.MaxPooling2D(pool_size=(2, 2), name='ellipse_maxpooling')
        self.flatten_layer = KL.Flatten(name='ellipse_flatten')
        self.dense1_layer = KL.Dense(128, activation='relu', name='ellipse_dense1')
        self.dense2_layer = KL.Dense(4, activation='linear', name='ellipse_dense2')

    def call(self, inputs):
        # detections = inputs[0]
        # masks = inputs[1]
        class_ids = inputs[1]
        masks = inputs[3]

        # Gets the coordinates of ellipse detections in the original detections tensor
        detection_coords = prepare_class_detections(class_ids, Ellipse.class_id)
        masks = tf.expand_dims(tf.gather_nd(masks, detection_coords), axis=-1)

        # print('Masks for ellipses: {}'.format(masks))

        x = self.conv2d_layer(masks)
        x = self.max_pooling_2d_layer(x)
        x = self.flatten_layer(x)
        x = self.dense1_layer(x)
        return Ellipse.class_id, detection_coords, self.dense2_layer(x)

def vectors_loss_graph(detections, masks, predicted_shapes):
    """
    Parameters
    ----------
    original_shapes: (shapes_classes, shapes_bboxes, shapes_params) Original shapes parameters
    predicted_shapes: [num_classes, (class_id, coordinates, prediction)] Actual predicted shapes
    """
    print('detections: {}'.format(detections))
    print('predicted_shapes: {}'.format(predicted_shapes))

    # target_class_ids = K.reshape(target_class_ids, (-1,))
    # target_bbox = K.reshape(target_bbox, (-1, 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    # positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    # positive_roi_class_ids = tf.cast(
    #     tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    # indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1).numpy().tolist()

    losses = []

    for class_shape_predictions in predicted_shapes:
        class_id, coordinates, predictions = class_shape_predictions
        if tf.shape(coordinates)[0] == 0:
            continue
        bboxes = tf.gather_nd(detections, coordinates)
        gt_masks = tf.gather_nd(masks, coordinates)

        if class_id == Rectangle.class_id:
            print('Class: {}\nPreds: {}\nBboxes: {}'.format(class_id, predictions, bboxes))

        # gt_classes = ragged_slice(original_shapes, 0)
        # gt_boxes = ragged_slice(original_shapes, 1)

        # if class_id == Rectangle.class_id:
            # gt_rect_indices = tf.unstack(tf.where(gt_classes == Rectangle.class_id), axis=1)
            # gt_rect_indices[1] = tf.ones(tf.shape(gt_rect_indices[1])[0], dtype=tf.int64)
            # gt_rect_indices = tf.stack(gt_rect_indices, axis=1)
            # print('gt_rect_indices: {}'.format(gt_rect_indices))
            #
            # gt_boxes = tf.gather_nd(original_shapes, gt_rect_indices)
            #
            # print('gt_boxes: {}'.format(gt_boxes))

            # for pi, coordinate in enumerate(coordinates):
            #     batch = coordinate[0]
            #     # print('\nIteration: {}\nBatch: {} -> {} ({})\nGT Boxes: {}\nGT Classes: {}\n'.format(pi, coordinate, batch, predictions[pi], gt_boxes[batch], gt_classes[batch]))
            #     batch_gt_rect_indices = tf.where(gt_classes[batch] == Rectangle.class_id)
            #     batch_gt_bboxes = tf.gather(gt_boxes[batch], batch_gt_rect_indices)
            #     if tf.shape(batch_gt_bboxes)[0] == 0:
            #         # No corresponding rect shape specs, skip batch
            #         continue

            # rpn_roi_area = (predictions[:, 2] - predictions[:, 0]) * \
            #                (predictions[:, 3] - predictions[:, 1])
            # gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
            #               (gt_boxes[:, 3] - gt_boxes[:, 1])

            # Compute overlaps [predictions, target_shapes]
            # overlaps = tf.zeros((tf.shape(predictions)[0], tf.shape(gt_boxes)[0]))
            # for i in range(tf.shape(overlaps)[1]):
            #     gt = gt_boxes[i]
            #     overlaps[:, i] = compute_iou_graph(gt, predictions, gt_box_area[i], rpn_roi_area)
            #
            # print('>>> Rects\nPredictions: {}\nGT Shapes: {}\bOverlaps: {}'.format(predictions, gt_shapes, overlaps))
            # loss = K.switch(tf.size(input=coord_indices) > 0,
            #                 smooth_l1_loss(y_true=gt_shapes, y_pred=predictions),
            #                 tf.constant(0.0))
            # losses.append(K.mean(loss))
            # tf.print('Loss', loss)

        # if classes[class_id] == 'ellipse':
        #     loss = K.switch(tf.size(input=coord_indices) > 0,
        #                     smooth_l1_loss(y_true=gt_shapes, y_pred=predictions),
        #                     tf.constant(0.0))
        #     losses.append(K.mean(loss))
        #     tf.print('Loss', loss)
            # print('Class: {}, coords: {}, predictions: {}'.format(classes[class_id], coordinates, predictions))

    return K.mean(tf.constant(losses))
