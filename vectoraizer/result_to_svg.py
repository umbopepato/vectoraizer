import numpy as np
from matplotlib.colors import rgb2hex
from pysvg import structure
from pysvg import shape as svg_shape
from vectoraizer.shapes import Rectangle, shapes_types, Ellipse

def extract_color(image, x1, x2, y1, y2):
    x = int(x1 + (x2 - x1) / 2)
    y = int(y1 + (y2 - y1) / 2)
    return rgb2hex(image[y, x] / 255)

def result_to_svg(image, boxes, masks, class_ids):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
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
