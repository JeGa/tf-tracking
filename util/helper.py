import util.global_config as global_config

import tensorflow as tf
import time
import datetime
import os
import numpy as np
from PIL import Image
from PIL import ImageDraw


def tfprint(node, out):
    """
    For debugging.
    """
    return tf.Print(node, out, summarize=10)


class timeit:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.duration = time.time() - self.start_time

    def time(self):
        return self.duration


def draw_bb_and_save(images, gt_bbs, frcnn_bbs):
    """
    :param images: Shape [batch_size/sequence_size, height, width, 3].
    :param gt_bbs: Shape [batch_size/sequence_size, 10, 4]. bbs in format x, y, w, h.
    :param frcnn_bbs: Shape [batch_size/sequence_size, 10, 4]. bbs in format ymin, xmin, ymax, xmax.
    """
    height = images.shape[1]
    width = images.shape[2]

    for i in range(images.shape[0]):
        img = Image.fromarray(images[i].astype(np.uint8))
        draw = ImageDraw.Draw(img)

        for j in range(gt_bbs.shape[1]):
            gt = gt_bbs[i][j]
            pred = frcnn_bbs[i][j]

            # Wants [x0, y0, x1, y1]
            draw.rectangle([pred[1] * width, pred[0] * height, pred[3] * width, pred[2] * height], outline='red')

            x = gt[0]
            y = gt[1]
            w = gt[2]
            h = gt[3]

            xmin = x - w / 2.0
            ymin = y - h / 2.0
            xmax = x + w / 2.0
            ymax = y + h / 2.0

            draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline='blue')

        file = os.path.normpath(os.path.join(global_config.cfg['results'], 'input_image'))

        img.save(file + '_' + str(i) + '.jpg')


def summary_file(parameters):
    current_datetime = datetime.datetime.now().strftime('%d-%m-%y.%H-%M-%S')

    if len(parameters) == 0:
        return os.path.join(global_config.cfg['train_summaries'], current_datetime)

    format_string = '-{}' * (len(parameters) - 1)
    format_string = '.{}' + format_string

    parameters_string = format_string.format(*parameters)

    summary_name = current_datetime + parameters_string

    return os.path.join(global_config.cfg['train_summaries'], summary_name)


def debug():
    import IPython
    IPython.embed()
    import sys
    sys.exit()
