import mediapipe as mp
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import dlib

mp_iris = mp.solutions.iris


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Iris:
    def __init__(self, points, align='left'):
        allow = []
        if align == 'left':
            allow = [468, 470, 472, 471, 469]
        if align == 'right':
            allow = [473, 475, 477, 474, 476]
        self.center = points[allow[0]]
        self.top = points[allow[1]]
        self.bottom = points[allow[2]]
        self.left = points[allow[3]]
        self.right = points[allow[4]]

    def get_list(self):
        return [self.center, self.top, self.bottom,  self.left, self.right]


class ColorUtils:
    def __init__(self):
        pass

    def get_points(self, image: np.ndarray):
        with mp_iris.Iris() as iris:
            results = iris.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        landmarks = np.array(results.face_landmarks_with_iris.ListFields()[0][1])
        w, h = image.shape[1], image.shape[0]
        points = []
        for landmark in landmarks:
            points.append(Point(x=int(landmark.x * w), y=int(landmark.y * h)))

        return points

    def evaluate_iris_and_pupil_rad(self, iris: Iris) -> (int, int):
        r1 = math.sqrt((iris.center.x - iris.top.x)**2 + (iris.center.y - iris.top.y)**2)
        r2 = math.sqrt((iris.center.x - iris.bottom.x)**2 + (iris.center.y - iris.bottom.y)**2)
        r3 = math.sqrt((iris.center.x - iris.left.x)**2 + (iris.center.y - iris.left.y)**2)
        r4 = math.sqrt((iris.center.x - iris.right.x)**2 + (iris.center.y - iris.right.y)**2)
        r = np.array([r1, r2, r3, r4]).mean()

        iris_radius = int(r)
        pupil_radius = int(r/2)

        return iris_radius, pupil_radius

    def make_mask(self, image: np.ndarray, iris: Iris) -> np.ndarray:
        iris_radius, pupil_radius = self.evaluate_iris_and_pupil_rad(iris)

        color_mask = image.copy()
        cv2.circle(color_mask, (iris.center.x, iris.center.y), iris_radius, (0, 255, 0), cv2.FILLED)
        cv2.circle(color_mask, (iris.center.x, iris.center.y), pupil_radius, (255, 0, 0), cv2.FILLED)

        mask = np.zeros((image.shape[0], image.shape[1]))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if color_mask[i, j, 0] == 0 and color_mask[i, j, 1] == 255 and color_mask[i, j, 2] == 0:
                    mask[i, j] = 1

        return mask


    def process_image(self, image: np.ndarray):
        """
        notes: image will be const
        """
        points = self.get_points(image)
        left = Iris(points, align='left')
        right = Iris(points, align='right')

        mask = np.array(self.make_mask(image, left) + self.make_mask(image, right), dtype=bool)
        b, g, r = cv2.split(image)

        return np.median(r[mask]), np.median(g[mask]), np.median(b[mask]), left, right

    def plot_image_and_detected_color(self, image: np.ndarray, savepath: str):
        r, g, b, left, right = self.process_image(image)

        fig, axs = plt.subplots(1, 3, figsize=(9, 4))
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[2].set_xticks([])
        axs[2].set_yticks([])

        axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        iris_radius, pupil_radius = self.evaluate_iris_and_pupil_rad(left)
        cv2.circle(image, (left.center.x, left.center.y), iris_radius, (0, 255, 0))
        cv2.circle(image, (left.center.x, left.center.y), pupil_radius, (0, 255, 0))

        iris_radius, pupil_radius = self.evaluate_iris_and_pupil_rad(right)
        cv2.circle(image, (right.center.x, right.center.y), iris_radius, (0, 255, 0))
        cv2.circle(image, (right.center.x, right.center.y), pupil_radius, (0, 255, 0))


        axs[1].imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        image[:, :, :] = [r, g, b]
        axs[2].imshow(image)

        plt.rcParams['savefig.facecolor'] = 'white'
        plt.savefig(savepath, dpi=700)


def process_dir():
    cu = ColorUtils()
    images_path = './data_to_inference'
    result_dir = './plots'

    for path_from_top, subdirs, files in os.walk(images_path):
        for f in files:
            if not f.endswith('jpg'):
                continue
            image = cv2.imread(path_from_top + '/' + f)
            print("Working with " + f)
            cu.plot_image_and_detected_color(image, result_dir + '/' + f)


def detect():
    process_dir()
    # image = cv2.imread('1.png')
    # cu = ColorUtils()
    # cu.plot_image_and_detected_color(image, 'out.jpg')
    # print(strange_cycle())


detect()


"""Some unused code"""
def draw_circle(image: np.ndarray, point, rad=10):
    cv2.circle(image, (point.x, point.y), rad, (0, 255, 0), int(rad))


def draw_all_landmarks(image: np.ndarray, points: np.ndarray):
    for point in points:
        draw_circle(image, point, rad=2)


def draw_only_iris(image: np.ndarray, points: np.ndarray):
    left = Iris(points, align='left').get_list()
    right = Iris(points, align='right').get_list()

    for point in left:
        draw_circle(image, point, rad=2)

    for point in right:
        draw_circle(image, point, rad=2)
