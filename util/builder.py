import tensorflow as tf
import tensorflow.contrib.slim as slim

import input_pipeline.sequences.tfrecord_reader
import input_pipeline.sequences.placeholder
import util.global_config as global_config
import networks.lstm


def _format(input_data):
    # From [x, y, w, h] to [y_min, x_min, y_max, x_max].

    # input_data = util.helper.tfprint(input_data, [input_data])

    # Each of shape (1, 10).
    x = tf.maximum(input_data[:, :, 0], 0.0)
    y = tf.maximum(input_data[:, :, 1], 0.0)
    w = tf.maximum(input_data[:, :, 2], 0.0)
    h = tf.maximum(input_data[:, :, 3], 0.0)

    y_min = y - h / 2.0
    x_min = x - w / 2.0
    y_max = y + h / 2.0
    x_max = x + w / 2.0

    input_data = tf.stack([y_min, x_min, y_max, x_max], axis=-1)

    # TODO input_data = util.helper.tfprint(input_data, [input_data])

    return input_data


def bbdata_image(bbs, image, name):
    """
    :param bbs: Shape (40).
    :param image: Shape (height, width, channels).
    :param name: Name for the tf summary.
    """
    # Shape (1, 10, 4) and [y_min, x_min, y_max, x_max] format for bbs.
    bbs = _format(tf.expand_dims(tf.reshape(bbs, [-1, 4]), 0))

    # Shape (1, height, width, channels).
    image = tf.expand_dims(image, 0)

    image_bbs = tf.image.draw_bounding_boxes(image, bbs)

    tf.summary.image(name, image_bbs)


def add_summaries(tensors):
    tf.summary.scalar('lstm_loss', tensors['lstm']['total_loss'])

    batch = 0
    sequence_element = -1

    bbdata_image(tensors['lstm']['inputs'][batch][sequence_element],
                 tensors['input_data_placeholders']['images'][batch][sequence_element],
                 'inputs')

    bbdata_image(tensors['lstm']['targets'][batch][sequence_element],
                 tensors['input_data_placeholders']['images'][batch][sequence_element],
                 'targets')

    bbdata_image(tensors['lstm']['predictions'][batch][sequence_element],
                 tensors['input_data_placeholders']['images'][batch][sequence_element],
                 'predictions')

    tensors['summary'] = tf.summary.merge_all()


def build_network(inputs, targets):
    total_loss, train_step, predictions = networks.lstm.build(inputs, targets,
                                                              global_config.cfg['state_size'],
                                                              global_config.cfg['lstm_layers'],
                                                              learning_rate=global_config.cfg['learning_rate'])

    region_proposals = tf.placeholder(tf.float32, shape=(global_config.cfg['batch_size'],
                                                         global_config.cfg['backprop_step_size'],
                                                         10, 4), name='frcnn_inputs')

    # predictions Shape (batch_size, step_size, output_dimension).

    current_region_proposals = tf.reshape(region_proposals, [global_config.cfg['batch_size'],
                                                             global_config.cfg['backprop_step_size'],
                                                             40])

    start_region_proposal = tf.zeros((global_config.cfg['batch_size'], 1, 10, 4))
    last_region_proposals = tf.concat([start_region_proposal, region_proposals[:, :-1]], axis=1)
    last_region_proposals = tf.reshape(last_region_proposals, [global_config.cfg['batch_size'],
                                                               global_config.cfg['backprop_step_size'],
                                                               40])

    start_lstm_prediction = tf.zeros((global_config.cfg['batch_size'], 1, 40))
    lstm_predictions = tf.concat([start_lstm_prediction, predictions[:, :-1]], axis=1)

    # Stack all into input vector for classificator.
    cls_input = tf.concat([current_region_proposals, last_region_proposals, lstm_predictions], axis=2)

    def classificators(cls_input):
        with tf.name_scope('cls_fc'):
            for j in range(10):
                slim.fully_connected(cls_input)

    # 1. lstm_predictions: [batch_size, sequence_length, 10, 4].
    #       - Add zero vector to have [batch_size, sequence_length + 1, 10, 4]
    #       - Optionally discard last vector.
    # 2. region_proposals (placeholder): [batch_size = 1, sequence_length, 10, 4].
    # 3. last_region_proposals: [batch_size = 1, sequence_length, 10, 4]

    lstm_tensors = {
        'inputs': inputs,
        'targets': targets,
        'total_loss': total_loss,
        'train_step': train_step,
        'predictions': predictions,

        'last_region_proposal': last_region_proposals,
        'last_lstm_predictions': lstm_predictions
    }

    rpn_tensors = {
        'region_proposals_placeholder': region_proposals
    }

    return lstm_tensors, rpn_tensors


def build():
    # The bb coordinates and the images are normalized.
    # TODO: Move parameters inside.
    input_data, input_handles = input_pipeline.sequences.tfrecord_reader.read_graph(
        global_config.cfg['input_data_training'],
        global_config.cfg['input_data_testing'])

    input_data_placeholders = input_pipeline.sequences.placeholder.build()

    with tf.variable_scope('model'):
        lstm_tensors, rpn_tensors = build_network(input_data_placeholders['groundtruth_bbs'],
                                                  input_data_placeholders['target_bbs'])

    tensors = {
        'input_data': input_data,

        'input_data_placeholders': input_data_placeholders,
        'lstm': lstm_tensors,
        'rpn': rpn_tensors
    }

    add_summaries(tensors)

    return tensors, input_handles
