import tensorflow as tf

import input_pipeline.sequences.tfrecord_reader
import util.global_config as global_config


def build_cls_labels():
    PLAYERS = input_pipeline.sequences.tfrecord_reader.PLAYERS
    batch_size = global_config.cfg['batch_size']
    backprop_step_size = global_config.cfg['backprop_step_size']

    target_cls = tf.placeholder(tf.int32, shape=(batch_size, backprop_step_size, PLAYERS), name='target_cls')

    return target_cls


def rps():
    region_proposals = tf.placeholder(tf.float32, shape=(global_config.cfg['batch_size'],
                                                         global_config.cfg['backprop_step_size'],
                                                         10, 4), name='frcnn_inputs')

    return region_proposals


def build_lstm_input():
    PLAYERS = input_pipeline.sequences.tfrecord_reader.PLAYERS
    batch_size = global_config.cfg['batch_size']
    backprop_step_size = global_config.cfg['backprop_step_size']

    groundtruth_bbs = tf.placeholder(tf.float32,
                                     shape=(batch_size, backprop_step_size, 4 * PLAYERS),
                                     name='groundtruth_bbs')

    target_bbs = tf.placeholder(tf.float32,
                                shape=(batch_size, backprop_step_size, 4 * PLAYERS),
                                name='target_bbs')

    images = tf.placeholder(tf.float32,
                            shape=(batch_size, backprop_step_size, None, None, 3),
                            name='images')

    input_data = {
        'groundtruth_bbs': groundtruth_bbs,
        'target_bbs': target_bbs,
        'images': images
    }

    return input_data
