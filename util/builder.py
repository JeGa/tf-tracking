import tensorflow as tf

import input_pipeline.sequences.tfrecord_reader
import util.global_config as global_config
import networks.lstm


def summaries(tensors):
    tf.summary.scalar('lstm_loss', tensors['lstm']['total_loss'])

    tensors['summary'] = tf.summary.merge_all()


def build_network(inputs, targets):
    total_loss, train_step, predictions = networks.lstm.build(inputs, targets,
                                                              global_config.cfg['state_size'],
                                                              global_config.cfg['lstm_layers'])

    lstm_tensors = {
        'inputs': inputs,
        'targets': targets,
        'total_loss': total_loss,
        'train_step': train_step,
        'predictions': predictions
    }

    return lstm_tensors


def build():
    # The bb coordinates and the images are normalized.
    input_data, input_handles = input_pipeline.sequences.tfrecord_reader.read_graph(
        global_config.cfg['input_data_training'],
        global_config.cfg['input_data_testing'])

    with tf.variable_scope('model'):
        lstm_tensors = build_network(input_data['groundtruth_bbs'], input_data['target_bbs'])

    tensors = {
        'input_data': input_data,
        'lstm': lstm_tensors
    }

    summaries(tensors)

    return tensors, input_handles
