import tensorflow as tf

import util.global_config as global_config

import input_pipeline.tfrecord_reader


def build():
    # The bb coordinates are saved un-normalized.
    input_data, input_handles = input_pipeline.tfrecord_reader.read_graph(
        global_config.cfg['input_data_training'],
        global_config.cfg['input_data_testing'])

    # with tf.variable_scope("model"):
    #    tensors = build_network(input_data['image'], input_data['groundtruth'],
    #                            input_data['gt_detections_nr'], input_data['name'],
    #                            is_training=False, reuse=None)

    # TODO summaries_training(tensors)

    # TODO: Just temporary.
    tensors = dict(input_data=input_data)

    # tensors['summary'] = tf.summary.merge_all()

    return tensors, input_handles
