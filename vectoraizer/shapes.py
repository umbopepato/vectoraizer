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
        rx = random.randint(int(image_width / 5), int(image_width / 3))
        ry = random.randint(int(image_height / 5), int(image_height / 3))
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
        w = random.randint(int(image_width / 4), int(image_width * 0.8))
        h = random.randint(int(image_height / 4), int(image_height * 0.8))
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


class Path(Shape):
    """
    <polygon>, TODO <path>, <line>, <polyline>
    """

    name = 'path'

    def __init__(self, vertices):
        """
        Parameters
        ----------
        vertices: ndarray
        """
        self.vertices = vertices
        minx = np.min(vertices[:, 0])
        miny = np.min(vertices[:, 1])
        maxx = np.max(vertices[:, 0])
        maxy = np.max(vertices[:, 1])
        self.bounding_box = (minx, miny, maxx - minx, maxy - miny)
        self.fill_color = random_color()

    def is_closed(self):
        # TODO
        # return np.array_equal(self.vertices[0], self.vertices[-1])
        return True

    def is_line(self):
        return self.vertices.size == 2

    @staticmethod
    def generate(image_width, image_height):
        num_vertices = random.randint(3, 8)

        # TODO random lines, polylines and polygons
        # closed = bool(random.getrandbits(1)) if num_vertices > 2 else False
        # vertices = np.array([np.random.randint(0, min(image_width, image_height), 2) for _ in range(num_vertices)])
        # if closed:
        #     vertices = np.append(vertices, [vertices[0]], axis=0)

        X_rand, Y_rand = np.sort(np.random.random(num_vertices)), np.sort(np.random.random(num_vertices))
        X_new, Y_new = np.zeros(num_vertices), np.zeros(num_vertices)

        # divide the interior points into two chains
        last_true = last_false = 0
        for i in range(1, num_vertices):
            if i != num_vertices - 1:
                if random.getrandbits(1):
                    X_new[i] = X_rand[i] - X_rand[last_true]
                    Y_new[i] = Y_rand[i] - Y_rand[last_true]
                    last_true = i
                else:
                    X_new[i] = X_rand[last_false] - X_rand[i]
                    Y_new[i] = Y_rand[last_false] - Y_rand[i]
                    last_false = i
            else:
                X_new[0] = X_rand[i] - X_rand[last_true]
                Y_new[0] = Y_rand[i] - Y_rand[last_true]
                X_new[i] = X_rand[last_false] - X_rand[i]
                Y_new[i] = Y_rand[last_false] - Y_rand[i]

        # randomly combine x and y and sort by polar angle
        np.random.shuffle(Y_new)
        vertices = np.stack((X_new, Y_new), axis=-1)
        vertices = vertices[np.argsort(np.arctan2(vertices[:, 1], vertices[:, 0]))]

        # arrange points end to end to form a polygon
        vertices = np.cumsum(vertices, axis=0)

        # Remap to fit the desired bounding box
        x_min, y_min = np.min(vertices[:, 0]), np.min(vertices[:, 1])
        x_max, y_max = np.max(vertices[:, 0]), np.max(vertices[:, 1])

        if x_min < 0:
            vertices[:, 0] += 0 - x_min

        if y_min < 0:
            vertices[:, 1] += 0 - y_min

        x_min, y_min = np.min(vertices[:, 0]), np.min(vertices[:, 1])
        x_max, y_max = np.max(vertices[:, 0]), np.max(vertices[:, 1])
        new_x = random.randint(0, int(image_width * 0.8))
        new_y = random.randint(0, int(image_height * 0.8))
        x_mapper = lambda item: new_x + ((item - x_min) * (image_width - new_x) * random.uniform(0.7, 1)) / (x_max - x_min)
        y_mapper = lambda item: new_y + ((item - y_min) * (image_height - new_y) * random.uniform(0.7, 1)) / (y_max - y_min)
        vertices[:, 0] = np.vectorize(x_mapper)(vertices[:, 0])
        vertices[:, 1] = np.vectorize(y_mapper)(vertices[:, 1])

        return Path(np.int32(vertices))

    def draw(self, canvas, color=None, with_bbox=False):
        if self.is_closed():
            cv2.fillPoly(
                canvas,
                [self.vertices],
                color if color is not None else self.fill_color
            )
        else:
            cv2.polylines(
                canvas,
                [self.vertices],
                self.is_closed(),
                color if color is not None else self.fill_color
            )
        if with_bbox:
            self.draw_bbox(canvas)
        return canvas
