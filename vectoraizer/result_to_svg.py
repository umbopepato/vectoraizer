import numpy as np
from matplotlib.colors import rgb2hex
from pysvg import shape as svg_shape
from pysvg import structure

from vectoraizer.shapes import shapes_types, Ellipse


def extract_color(image, x1, x2, y1, y2):
    x = int(x1 + (x2 - x1) / 2)
    y = int(y1 + (y2 - y1) / 2)
    return rgb2hex(image[y, x] / 255)

def boxes_should_be_switched(ba, bb, ma, mb):
    x1 = max(ba[0], bb[0])
    y1 = max(ba[1], bb[1])
    x2 = min(ba[2], bb[2])
    y2 = min(ba[3], bb[3])
    if x2 < x1 or y2 < y1:
        return False
    return np.count_nonzero(ma[x1:x2, y1:y2] == True) > np.count_nonzero(mb[x1:x2, y1:y2] == True)

def sort_results_3d(boxes, masks, class_ids):
    n = len(boxes)
    swapped = True
    indexes = list(range(len(boxes)))
    while swapped and n > 0:
        swapped = False
        for i in range(1, n):
            if boxes_should_be_switched(boxes[indexes[i - 1]], boxes[indexes[i]], masks[:, :, indexes[i - 1]], masks[:, :, indexes[i]]):
                tmp = indexes[i - 1]
                indexes[i - 1] = indexes[i]
                indexes[i] = tmp
                swapped = True
        n = n - 1
    return (
        np.array([boxes[x] for x in indexes]),
        np.array([class_ids[x] for x in indexes]),
    )


def result_to_svg(image, boxes, masks, class_ids):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    """
    boxes, class_ids = sort_results_3d(boxes, masks, class_ids)

    image_height, image_width = image.shape[:2]
    svg = structure.Svg(0, 0, image_width, image_height)
    bg = svg_shape.Rect(0, 0, image_width, image_height)
    bg.set_fill(extract_color(image, 0, 0, 0, 0))
    svg.addElement(bg)

    objects_count = boxes.shape[0]
    for i in range(objects_count):
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue

        shape_class = shapes_types[class_ids[i] - 1]
        y1, x1, y2, x2 = boxes[i]
        color = extract_color(image, x1, x2, y1, y2)

        if shape_class == Ellipse:
            width = x2 - x1
            height = y2 - y1
            cx = x1 + width / 2
            cy = y1 + height / 2
            rx = width / 2
            ry = height / 2
            shape = shape_class(cx, cy, rx, ry)
        else:
            shape = shape_class(x1, y1, x2 - x1, y2 - y1)

        element = shape.to_svg()

        if element is not None:
            element.set_fill(color)
            svg.addElement(element)

    return svg
