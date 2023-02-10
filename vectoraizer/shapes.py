import random
from abc import abstractmethod
from typing import Tuple
import cv2
import numpy as np


def random_color():
    return tuple([random.randint(0, 255) for _ in range(3)])


class Shape:
    fill_color = None
    bounding_box: Tuple[int, int, int, int] = None

    @property
    def name(self):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def generate(image_width, image_height):
        """
        Generate a random shape inside the image sizes

        Parameters
        ----------
        image_width : int
        image_height : int

        Returns
        -------
        Shape
        """
        raise NotImplementedError

    @abstractmethod
    def draw(self, canvas, color, with_bbox):
        """

        Parameters
        ----------
        canvas : ndarray
        color : ndarray
        with_bbox : bool
        """
        raise NotImplementedError

    def draw_bbox(self, canvas):
        x, y, width, height = self.bounding_box
        cv2.rectangle(
            canvas,
            (x, y),
            (x + width, y + height),
            (255, 0, 0),
        )


class Ellipse(Shape):
    """
    <circle>, <ellipse>
    """

    name = 'ellipse'

    def __init__(self, cx, cy, rx, ry):
        """
        Parameters
        ----------
        cx: int
        cy: int
        rx: int
        ry: int
        """
        self.cx = cx
        self.cy = cy
        self.rx = rx
        self.ry = ry
        # self.rot = rot TODO
        self.bounding_box = (cx - rx, cy - ry, rx * 2, ry * 2)
        self.fill_color = random_color()

    def is_circle(self):
        return self.rx == self.ry

    @staticmethod
    def generate(image_width, image_height):
        rx = random.randint(int(image_width / 6), int(image_width / 5))
        ry = random.randint(int(image_height / 6), int(image_height / 5))
        return Ellipse(
            random.randint(rx, image_width - rx),
            random.randint(ry, image_height - ry),
            rx,
            ry,
        )

    def draw(self, canvas, color=None, with_bbox=False):
        cv2.ellipse(
            canvas,
            (self.cx, self.cy),
            (self.rx, self.ry),
            color=color if color is not None else self.fill_color,
            angle=0,
            startAngle=0,
            endAngle=360,
            thickness=-1,
        )
        if with_bbox:
            self.draw_bbox(canvas)
        return canvas


class Triangle(Shape):
    name = 'triangle'

    def __init__(self, x, y, width, height):
        """
        Parameters
        ----------
        x: int
        y: int
        width: int
        height: int
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        # self.rot = rot TODO
        self.bounding_box = (x, y, width, height)
        self.fill_color = random_color()

    @staticmethod
    def generate(image_width, image_height):
        w = random.randint(int(image_width / 5), int(image_width / 3))
        h = random.randint(int(image_height / 5), int(image_height / 3))
        return Triangle(
            random.randint(0, image_width - w),
            random.randint(0, image_height - h),
            w,
            h,
        )

    def draw(self, canvas, color=None, with_bbox=False):
        cv2.fillPoly(
            canvas,
            np.int32([np.array([[self.x, self.y + self.height], [self.x + self.width / 2, self.y], [self.x + self.width, self.y + self.height]])]),
            color if color is not None else self.fill_color
        )
        if with_bbox:
            self.draw_bbox(canvas)
        return canvas


class Rectangle(Shape):
    """
    <rect>
    """

    name = 'rect'

    def __init__(self, x, y, width, height):
        """
        Parameters
        ----------
        x: int
        y: int
        width: int
        height: int
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        # self.rot = rot TODO
        self.bounding_box = (x, y, width, height)
        self.fill_color = random_color()

    def is_square(self):
        return self.width == self.height

    @staticmethod
    def generate(image_width, image_height):
        w = random.randint(int(image_width / 5), int(image_width / 3))
        h = random.randint(int(image_height / 5), int(image_height / 3))
        return Rectangle(
            random.randint(0, image_width - w),
            random.randint(0, image_height - h),
            w,
            h,
        )

    def draw(self, canvas, color=None, with_bbox=False):
        cv2.rectangle(
            canvas,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            color if color is not None else self.fill_color,
            -1,
        )
        if with_bbox:
            self.draw_bbox(canvas)
        return canvas


class Pentagon(Shape):
    name = 'pentagon'

    def __init__(self, x, y, width, height):
        """
        Parameters
        ----------
        x: int
        y: int
        width: int
        height: int
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        # self.rot = rot TODO
        self.bounding_box = (x, y, width, height)
        self.fill_color = random_color()

    @staticmethod
    def generate(image_width, image_height):
        w = random.randint(int(image_width / 5), int(image_width / 3))
        h = random.randint(int(image_height / 5), int(image_height / 3))
        return Pentagon(
            random.randint(0, image_width - w),
            random.randint(0, image_height - h),
            w,
            h,
        )

    def draw(self, canvas, color=None, with_bbox=False):
        vertices = np.array([[60, 13], [110, 48], [92, 110], [30, 110], [13, 48]])
        x_min, y_min = np.min(vertices[:, 0]), np.min(vertices[:, 1])
        x_max, y_max = np.max(vertices[:, 0]), np.max(vertices[:, 1])
        x_mapper = lambda item: self.x + ((item - x_min) * self.width) / (x_max - x_min)
        y_mapper = lambda item: self.y + ((item - y_min) * self.height) / (y_max - y_min)
        vertices[:, 0] = np.vectorize(x_mapper)(vertices[:, 0])
        vertices[:, 1] = np.vectorize(y_mapper)(vertices[:, 1])
        cv2.fillPoly(
            canvas,
            np.int32([vertices]),
            color if color is not None else self.fill_color
        )
        # cv2.rectangle(
        #     canvas,
        #     (self.x, self.y),
        #     (self.x + self.width, self.y + self.height),
        #     color if color is not None else self.fill_color,
        #     -1,
        # )
        if with_bbox:
            self.draw_bbox(canvas)
        return canvas

shapes_types = [Ellipse, Triangle, Rectangle, Pentagon]
shapes_classes = dict(map(lambda s: (s.name, s), shapes_types))
