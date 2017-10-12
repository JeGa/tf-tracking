import tensorflow as tf

import input_pipeline.sequences.tfrecord_reader
import input_pipeline.sequences.placeholder
import util.global_config as global_config
import networks.lstm
import networks.player_classificator


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
    tf.summary.scalar('lstm_reg_loss', tensors['lstm']['loss'])
    tf.summary.scalar('lstm_cls_loss', tensors['cls']['loss'])
    tf.summary.scalar('total_loss', tensors['combined']['loss'])

    batch = 0
    sequence_element = -1

    bbdata_image(tensors['lstm']['inputs'][batch][sequence_element],
                 tensors['placeholders']['images'][batch][sequence_element],
                 'inputs')

    bbdata_image(tensors['lstm']['targets'][batch][sequence_element],
                 tensors['placeholders']['images'][batch][sequence_element],
                 'targets')

    bbdata_image(tensors['lstm']['predictions'][batch][sequence_element],
                 tensors['placeholders']['images'][batch][sequence_element],
                 'predictions')

    tensors['summary'] = tf.summary.merge_all()


def build_network(inputs, targets, region_proposals, target_cls):
    with tf.variable_scope('lstm_bb_regressor'):
        lstm_reg_total_loss, lstm_reg_train_step, lstm_reg_predictions = networks.lstm.build(
            inputs, targets,
            global_config.cfg['state_size'],
            global_config.cfg['lstm_layers'],
            learning_rate=global_config.cfg['learning_rate'])

    with tf.variable_scope('lstm_bb_classificator'):
        (cls_input, last_region_proposals, last_lstm_predictions,
         lstm_cls_total_loss, lstm_cls_train_step, lstm_cls_predictions) = networks.player_classificator.build(
            region_proposals,
            lstm_reg_predictions,
            target_cls)

    with tf.name_scope('total_loss'):
        total_loss = 0.5 * lstm_cls_total_loss + 0.5 * lstm_reg_total_loss

    with tf.name_scope('total_loss_training'):
        total_train_step = tf.train.AdagradOptimizer(global_config.cfg['learning_rate']).minimize(total_loss)

    lstm_tensors = {
        'inputs': inputs,
        'targets': targets,

        'loss': lstm_reg_total_loss,
        'train_step': lstm_reg_train_step,
        'predictions': lstm_reg_predictions,

        'last_region_proposal': last_region_proposals,
        'last_lstm_predictions': last_lstm_predictions
    }

    classifier_tensors = {
        'inputs': cls_input,
        'targets': target_cls,

        'loss': lstm_cls_total_loss,
        'train_step': lstm_cls_train_step,
        'predictions': lstm_cls_predictions,

        'last_region_proposals': last_region_proposals,
        'last_lstm_predictions': last_lstm_predictions
    }

    combined_loss_tensors = {
        'loss': total_loss,
        'train_step': total_train_step
    }

    return lstm_tensors, classifier_tensors, combined_loss_tensors


def build_lstm_and_classifier():
    input_data_placeholders = input_pipeline.sequences.placeholder.build_lstm_input()
    target_cls_placeholder = input_pipeline.sequences.placeholder.build_cls_labels()
    region_proposals = input_pipeline.sequences.placeholder.rps()

    # This builds the lstm and the classifier network.
    with tf.variable_scope('model'):
        lstm_tensors, classifier_tensors, combined_loss_tensors = build_network(
            input_data_placeholders['groundtruth_bbs'],
            input_data_placeholders['target_bbs'],
            region_proposals,
            target_cls_placeholder)

    tensors = {
        'placeholders': {
            'groundtruth_bbs': input_data_placeholders['groundtruth_bbs'],
            'target_bbs': input_data_placeholders['target_bbs'],
            'images': input_data_placeholders['images'],

            'target_cls': target_cls_placeholder,
            'region_proposals': region_proposals
        },

        'lstm': lstm_tensors,
        'cls': classifier_tensors,
        'combined': combined_loss_tensors
    }

    # Adds 'summary' to tensor.
    add_summaries(tensors)

    return tensors


def build_input_pipeline():
    # The bb coordinates and the images are normalized.
    input_data, input_handles = input_pipeline.sequences.tfrecord_reader.build()

    return input_data, input_handles
