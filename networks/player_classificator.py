import tensorflow as tf
import numpy as np

import util.helper
import util.global_config as global_config


def generate_cls_gt(input_pipeline_out, frcnn_out):
    target_bbs = input_pipeline_out['target_bbs']

    batch_size = target_bbs.shape[0]
    sequence_size = target_bbs.shape[1]
    PLAYERS = 10

    # The groundtruth bbs with shape (batch_size, backprop_step_size, PLAYERS=10, 4).
    target_bbs = np.reshape(target_bbs, (batch_size, sequence_size, PLAYERS, 4))

    # The RPs with shape (batch_size, sequence_length, PLAYERS=10, 4).
    rp_bbs = frcnn_out

    # The groundtruth for the classifier with shape ().
    # Each RP bb is one of the following classes: t from [0, 1, ..., PLAYERS=10].
    # 0 means take LSTM prediction.
    target_cls = np.zeros((batch_size, sequence_size, PLAYERS))

    threshold = 0.6

    for b in range(batch_size):
        for s in range(sequence_size):
            # Shape (10, 4).
            current_gt_bbs = target_bbs[b, s]

            # Shape (10, 4).
            current_rp_bbs = rp_bbs[b, s]

            # Iterate through all rp bbs in this image.
            for i in range(PLAYERS):
                # Shape (4) in format [ymin, xmin, ymax, xmax].
                current_rp_bb = current_rp_bbs[i]

                ious = np.zeros(PLAYERS)

                # Iterate through all gt bbs in this image.
                for j in range(PLAYERS):
                    # Shape (4) in format [x, y, w, h].
                    current_gt_bb = current_gt_bbs[j]

                    ious[j] = util.helper.iou(current_gt_bb,
                                              util.helper.ymin_xmin_ymax_xmax_to_xywh(current_rp_bb))

                index = np.argmax(ious)
                if ious[index] >= threshold:
                    target_cls[b, s, i] = index + 1
                else:
                    target_cls[b, s, i] = 0

    return target_cls


def arrange_classifier_inputs(region_proposals, lstm_predictions):
    # 1. last_lstm_predictions: [batch_size, sequence_length, 10, 4].
    #       - Add zero vector to have [batch_size, sequence_length + 1, 10, 4]
    #       - Optionally discard last vector.
    # 2. region_proposals (placeholder): [batch_size = 1, sequence_length, 10, 4].
    # 3. last_region_proposals: [batch_size = 1, sequence_length, 10, 4]

    # lstm_predictions Shape (batch_size, step_size, output_dimension).

    current_region_proposals = tf.reshape(region_proposals, [global_config.cfg['batch_size'],
                                                             global_config.cfg['backprop_step_size'],
                                                             40])

    start_region_proposal = tf.zeros((global_config.cfg['batch_size'], 1, 10, 4))
    last_region_proposals = tf.concat([start_region_proposal, region_proposals[:, :-1]], axis=1)
    last_region_proposals = tf.reshape(last_region_proposals, [global_config.cfg['batch_size'],
                                                               global_config.cfg['backprop_step_size'],
                                                               40])

    start_lstm_prediction = tf.zeros((global_config.cfg['batch_size'], 1, 40))
    last_lstm_predictions = tf.concat([start_lstm_prediction, lstm_predictions[:, :-1]], axis=1)

    # Stack all into input vector for classificator.
    cls_input = tf.concat([current_region_proposals, last_region_proposals, last_lstm_predictions], axis=2)

    return cls_input, last_region_proposals, last_lstm_predictions


def _shapeinfo(cls_input, cls_target):
    input_shape = cls_input.get_shape()

    batch_size = input_shape[0]
    step_size = input_shape[1]
    input_dimension = input_shape[2]

    output_dimension = cls_target.get_shape()[2]

    return batch_size, step_size, input_dimension, output_dimension


def network(cls_input, cls_target):
    """
    :param cls_input: Shape (batch_size, sequence_size, 40 * 3).
    :param cls_target: Shape (batch_size, sequence_size, 10 + 1).
    """
    batch_size, step_size, input_dimension, output_dimension = _shapeinfo(cls_input, cls_target)
    pass


def build(region_proposals, lstm_predictions, cls_target):
    cls_input, last_region_proposals, last_lstm_predictions = arrange_classifier_inputs(region_proposals,
                                                                                        lstm_predictions)

    # def classificators(cls_input):
    #     with tf.name_scope('cls_fc'):
    #         for j in range(10):
    #             slim.fully_connected(cls_input)

    return cls_input, last_region_proposals, last_lstm_predictions
