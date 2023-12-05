import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from tensorflow.keras.models import load_model
from yad2k.models.keras_yolo import yolo_head
from yad2k.utils.utils import draw_boxes, get_colors_for_classes, scale_boxes, read_classes, read_anchors, \
    preprocess_image


def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
    box_scores = tf.multiply(box_confidence, box_class_probs)

    box_classes = tf.math.argmax(box_scores, axis=-1)
    box_class_scores = tf.math.reduce_max(box_scores, axis=-1)

    filtering_mask = box_class_scores >= threshold

    scores = box_class_scores[filtering_mask]
    boxes = boxes[filtering_mask]
    classes = box_classes[filtering_mask]

    return scores, boxes, classes


tf.random.set_seed(10)
box_confidence = tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed=1)
boxes = tf.random.normal([19, 19, 5, 4], mean=1, stddev=4, seed=1)
box_class_probs = tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed=1)
scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=0.5)
print("scores[2] = " + str(scores[2].numpy()))
print("boxes[2] = " + str(boxes[2].numpy()))
print("classes[2] = " + str(classes[2].numpy()))
print("scores.shape = " + str(scores.shape))
print("boxes.shape = " + str(boxes.shape))
print("classes.shape = " + str(classes.shape))

assert type(scores) == EagerTensor, "Use tensorflow functions"
assert type(boxes) == EagerTensor, "Use tensorflow functions"
assert type(classes) == EagerTensor, "Use tensorflow functions"

assert scores.shape == (1789,), "Wrong shape in scores"
assert boxes.shape == (1789, 4), "Wrong shape in boxes"
assert classes.shape == (1789,), "Wrong shape in classes"

assert np.isclose(scores[2].numpy(), 9.270486), "Values are wrong on scores"
assert np.allclose(boxes[2].numpy(), [4.6399336, 3.2303846, 4.431282, -2.202031]), "Values are wrong on boxes"
assert classes[2].numpy() == 8, "Values are wrong on classes"

print("\033[92m All tests passed!")


def iou(box1, box2):
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    xi1 = None
    yi1 = None
    xi2 = None
    yi2 = None
    inter_width = None
    inter_height = None
    inter_area = None

    box1_area = None
    box2_area = None
    union_area = None

    iou = None

    return iou


box1 = (2, 1, 4, 3)
box2 = (1, 2, 3, 4)

print("iou for intersecting boxes = " + str(iou(box1, box2)))
assert iou(box1, box2) < 1, "The intersection area must be always smaller or equal than the union area."
assert np.isclose(iou(box1, box2),
                  0.14285714), "Wrong value. Check your implementation. Problem with intersecting boxes"

box1 = (1, 2, 3, 4)
box2 = (5, 6, 7, 8)
print("iou for non-intersecting boxes = " + str(iou(box1, box2)))
assert iou(box1, box2) == 0, "Intersection must be 0"

box1 = (1, 1, 2, 2)
box2 = (2, 2, 3, 3)
print("iou for boxes that only touch at vertices = " + str(iou(box1, box2)))
assert iou(box1, box2) == 0, "Intersection at vertices must be 0"

box1 = (1, 1, 3, 3)
box2 = (2, 3, 3, 4)
print("iou for boxes that only touch at edges = " + str(iou(box1, box2)))
assert iou(box1, box2) == 0, "Intersection at edges must be 0"

print("\033[92m All tests passed!")
