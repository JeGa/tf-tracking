import tensorflow as tf

import input_pipeline.sequences.tfrecord_reader
import util.global_config as global_config


def build():
    PLAYERS = input_pipeline.sequences.tfrecord_reader.PLAYERS
    batch_size = global_config.cfg['batch_size']
    backprop_step_size = global_config.cfg['backprop_step_size']

    groundtruth_bbs = tf.placeholder(tf.float32,
                                     shape=(batch_size, backprop_step_size, 4 * PLAYERS),
                                     name='groundtruth_bbs')
    target_bbs = tf.placeholder(tf.float32,
                                shape=(batch_size, backprop_step_size, 4 * PLAYERS),
                                name='taregt_bbs')
    images = tf.placeholder(tf.float32,
                            shape=(batch_size, backprop_step_size, None, None, 3),
                            name='images')

    input_data = {
        'groundtruth_bbs': groundtruth_bbs,
        'target_bbs': target_bbs,
        'images': images
    }

    return input_data
