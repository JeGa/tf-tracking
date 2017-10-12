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


def draw_bb_and_save(images, gt_bbs, frcnn_bbs, lstm_bbs):
    """
    :param images: Shape [batch_size/sequence_size, height, width, 3].
    :param gt_bbs: Shape [batch_size/sequence_size, 10, 4]. bbs in format x, y, w, h.
    :param frcnn_bbs: Shape [batch_size/sequence_size, 10, 4]. bbs in format ymin, xmin, ymax, xmax.
    :param lstm_bbs: Shape [batch_size/sequence_size, 10, 4]. bbs in format x, y, w, h.
    """
    height = images.shape[1]
    width = images.shape[2]

    for i in range(images.shape[0]):
        img = Image.fromarray(images[i].astype(np.uint8))
        draw = ImageDraw.Draw(img)

        for j in range(gt_bbs.shape[1]):
            gt = gt_bbs[i][j]
            pred = frcnn_bbs[i][j]
            lstm = lstm_bbs[i][j]

            # Wants [x0, y0, x1, y1].
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

            x = lstm[0]
            y = lstm[1]
            w = lstm[2]
            h = lstm[3]

            xmin = x - w / 2.0
            ymin = y - h / 2.0
            xmax = x + w / 2.0
            ymax = y + h / 2.0

            draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline='green')

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


def draw_allbbs_and_cls_labels_and_save(
        image,
        reg_targets,
        reg_predictions,
        region_proposals,
        cls_targets,
        cls_predictions,
        filename):
    """
    :param image: Shape (height, width, 3).
    :param reg_targets: Shape (10, 4). Format [x, y, w, h].
    :param reg_predictions: Shape (10, 4). Format [x, y, w, h].
    :param region_proposals: Shape (10, 4). Format [ymin, xmin, ymax, xmax].
    :param cls_targets: Shape (10).
    :param cls_predictions: Shape (10, 11).
    :param filename: String.
    """
    height = image.shape[0]
    width = image.shape[1]

    img = Image.fromarray((image * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    players = reg_targets.shape[0]

    # draw.rectangle wants [x0, y0, x1, y1].

    # gt bbs.
    for i in range(players):
        gt_bb = reg_targets[i]

        xmin, ymin, xmax, ymax = xywh_to_xmin_ymin_xmax_ymax(gt_bb)
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline='red')
        draw.text([xmin * width, ymin * height - 10], str(i + 1), fill='red')

    # rps.
    for i in range(players):
        rp_bb = region_proposals[i]
        gt_label = cls_targets[i]

        draw.rectangle([rp_bb[1] * width, rp_bb[0] * height, rp_bb[3] * width, rp_bb[2] * height], outline='green')
        draw.text([rp_bb[1] * width + 20, rp_bb[0] * height - 10], str(int(gt_label)), fill='green')

    # lstm
    for i in range(players):
        pred_bb = reg_predictions[i]

        xmin, ymin, xmax, ymax = xywh_to_xmin_ymin_xmax_ymax(pred_bb)
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline='blue')

    # The predicted bb.
    for i in range(players):
        pred_label = np.argmax(cls_predictions[i])

        # Take lstm prediction from player i.
        if pred_label == 0:
            xmin, ymin, xmax, ymax = xywh_to_xmin_ymin_xmax_ymax(reg_predictions[i])
            draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline='yellow')
            draw.text([xmin * width + 40, ymin * height - 10], str(i + 1), fill='yellow')
        # Take rp from player pred_label.
        else:
            rp_bb = region_proposals[pred_label - 1]
            draw.rectangle([rp_bb[1] * width, rp_bb[0] * height, rp_bb[3] * width, rp_bb[2] * height], outline='yellow')
            draw.text([rp_bb[1] * width + 40, rp_bb[0] * height - 10], str(i), fill='yellow')

    draw.text([10, 10], 'groundtruth_bb', fill='red')
    draw.text([10, 20], 'region_proposal', fill='green')
    draw.text([10, 30], 'lstm', fill='blue')
    draw.text([10, 40], 'selected_tracking_bb', fill='yellow')

    file = os.path.normpath(os.path.join(global_config.cfg['results'], 'prediction'))
    img.save(file + '_' + filename + '.jpg')


def draw_bb_and_cls_labels_and_save(image, rp_bbs, gt_bbs, labels, filename):
    """
    NOTE: The groundtruth are always correctly sorted.
    That means gt_bbs[i] is always the i-th. player.

    :param image: Shape (height, width, 3).
    :param rp_bbs: Shape (10, 4). Format [ymin, xmin, ymax, xmax].
    :param gt_bbs: Shape (10, 4). Format [x, y, w, h].
    :param labels: Shape (10).
    :param filename: Appended to filename.
    """

    height = image.shape[0]
    width = image.shape[1]

    img = Image.fromarray((image * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # Over all bbs.
    for i in range(gt_bbs.shape[0]):
        gt_label = i
        gt_bb = gt_bbs[i]

        # Wants [x0, y0, x1, y1].
        xmin, ymin, xmax, ymax = xywh_to_xmin_ymin_xmax_ymax(gt_bb)
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline='red')
        draw.text([xmin * width, ymin * height - 10], str(i + 1), fill='red')

    for i in range(rp_bbs.shape[0]):
        rp_label = labels[i]
        rp_bb = rp_bbs[i]

        # Wants [x0, y0, x1, y1].
        draw.rectangle([rp_bb[1] * width, rp_bb[0] * height, rp_bb[3] * width, rp_bb[2] * height], outline='blue')
        draw.text([rp_bb[1] * width + 20, rp_bb[0] * height - 10], str(int(rp_label)))

    file = os.path.normpath(os.path.join(global_config.cfg['results'], 'cls_gt'))
    img.save(file + '_' + filename + '.jpg')


def xywh_to_xmin_ymin_xmax_ymax(bb):
    x, y, w, h = bb

    xmin = x - w / 2.0
    ymin = y - h / 2.0
    xmax = x + w / 2.0
    ymax = y + h / 2.0

    return [xmin, ymin, xmax, ymax]


def ymin_xmin_ymax_xmax_to_xywh(bb):
    ymin, xmin, ymax, xmax = bb

    w = xmax - xmin
    h = ymax - ymin

    x = xmin + w / 2.0
    y = ymin + h / 2.0

    return [x, y, w, h]


def iou(bb1, bb2):
    """
    :param bb1: Format [x, y, w, h].
    :param bb2: Format [x, y, w, h].

    :return: iou value.
    """
    x0, y0, w0, h0 = bb1
    x1, y1, w1, h1 = bb2

    xmin0, ymin0, xmax0, ymax0 = xywh_to_xmin_ymin_xmax_ymax(bb1)
    xmin1, ymin1, xmax1, ymax1 = xywh_to_xmin_ymin_xmax_ymax(bb2)

    area0 = w0 * h0
    area1 = w1 * h1

    if area0 == 0 or area1 == 0:
        return 0

    xmin_i = max([xmin0, xmin1])
    ymin_i = max([ymin0, ymin1])
    xmax_i = min([xmax0, xmax1])
    ymax_i = min([ymax0, ymax1])

    intersection_area = (xmax_i - xmin_i) * (ymax_i - ymin_i)

    union_area = float(area0 + area1 - intersection_area)
    iou = intersection_area / union_area

    if iou < 0:
        return 0
    return iou
