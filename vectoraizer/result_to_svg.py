import numpy as np
from matplotlib.colors import rgb2hex
from pysvg import structure
from pysvg import shape
from pysvg.builders import StyleBuilder

classes = {
    1: 'rect',
    2: 'circle',
    3: 'polygon',
}

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
    bg = shape.Rect(0, 0, image_width, image_height)
    bg.set_fill(extract_color(image, 0, 0, 0, 0))
    svg.addElement(bg)

    objects_count = boxes.shape[0]
    for i in range(objects_count):
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue

        element = None
        object_class = classes[class_ids[i]]
        y1, x1, y2, x2 = boxes[i]
        color = extract_color(image, x1, x2, y1, y2)
        # style = StyleBuilder()
        # style.setFilling(color)

        if object_class == 'circle':

            radius = (x2 - x1) / 2
            element = shape.Circle(x1 + radius, y1 + radius, radius)

        elif object_class == 'rect':

            element = shape.Rect(x1, y1, x2 - x1, y2 - y1)

        elif object_class == 'polygon':

            element = shape.Polygon()

        if element is not None:
            element.set_fill(color)
            svg.addElement(element)

    svg.save('test.svg')
