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


def draw_bb_and_save(images, bbs):
    """
    :param images: Shape [batch_size, height, width, 3].
    :param bbs: Shape [10, 4]. bbs in format ymin, xmin, ymax, xmax.
    """
    counter = 0
    for i in images:
        img = Image.fromarray(i.astype(np.uint8))
        draw = ImageDraw.Draw(img)

        height = images.shape[1]
        width = images.shape[2]

        for bb in bbs:
            # Wants [x0, y0, x1, y1]
            draw.rectangle([bb[1] * width, bb[0] * height, bb[3] * width, bb[2] * height], outline='red')

        file = os.path.normpath(os.path.join(global_config.cfg['results'], 'input_image'))

        img.save(file + '_' + str(counter) + '.jpg')

        counter += 1


def summary_file(parameters):
    current_datetime = datetime.datetime.now().strftime('%d-%m-%y.%H-%M-%S')

    if len(parameters) == 0:
        return os.path.join(global_config.cfg['train_summaries'], current_datetime)

    format_string = '-{}' * (len(parameters) - 1)
    format_string = '.{}' + format_string

    parameters_string = format_string.format(*parameters)

    summary_name = current_datetime + parameters_string

    return os.path.join(global_config.cfg['train_summaries'], summary_name)
