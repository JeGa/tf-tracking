import tensorflow as tf

import input_pipeline.sequences.tfrecord_reader
import input_pipeline.sequences.placeholder
import util.global_config as global_config
import networks.bbreg_lstm
import networks.cls_lstm


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


def add_train_summaries(tensors):
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


def build_network(reg_inputs, reg_targets,
                  cls_region_proposals, cls_ordered_lrp, cls_targets):
    with tf.variable_scope('lstm_bb_regressor'):
        lstm_reg_total_loss, lstm_reg_predictions = networks.bbreg_lstm.build(
            reg_inputs, reg_targets,
            global_config.cfg['state_size'],
            global_config.cfg['lstm_layers'])

    with tf.variable_scope('lstm_bb_classificator'):
        (cls_input, lstm_cls_total_loss, lstm_cls_predictions) = networks.cls_lstm.build(
            cls_region_proposals,
            cls_ordered_lrp,
            lstm_reg_predictions,
            cls_targets)

    with tf.variable_scope('total_loss_weights'):
        cls_weight = tf.placeholder(tf.float32, shape=())
        reg_weight = tf.placeholder(tf.float32, shape=())

    with tf.name_scope('total_loss'):
        total_loss = cls_weight * lstm_cls_total_loss + reg_weight * lstm_reg_total_loss

    with tf.name_scope('total_loss_training'):
        regvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model/lstm_bb_regressor')
        clsvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model/lstm_bb_classificator')

        total_train_step = tf.train.AdagradOptimizer(global_config.cfg['learning_rate']).minimize(total_loss,
                                                                                                  var_list=[regvars])

    lstm_tensors = {
        'inputs': reg_inputs,
        'targets': reg_targets,

        'loss': lstm_reg_total_loss,
        'predictions': lstm_reg_predictions,
    }

    classifier_tensors = {
        'inputs': cls_input,
        'targets': cls_targets,

        'loss': lstm_cls_total_loss,
        'predictions': lstm_cls_predictions,
    }

    combined_loss_tensors = {
        'loss': total_loss,
        'train_step': total_train_step,

        'cls_weight': cls_weight,
        'reg_weight': reg_weight
    }

    return lstm_tensors, classifier_tensors, combined_loss_tensors


def build_lstm_and_classifier():
    input_data_placeholders = input_pipeline.sequences.placeholder.build_lstm_input()

    target_cls_placeholder = input_pipeline.sequences.placeholder.build_cls_labels()
    region_proposals = input_pipeline.sequences.placeholder.rps()
    cls_ordered_lrp = input_pipeline.sequences.placeholder.build_cls_input()

    # This builds the lstm and the classifier network.
    with tf.variable_scope('model'):
        lstm_tensors, classifier_tensors, combined_loss_tensors = build_network(
            input_data_placeholders['groundtruth_bbs'], input_data_placeholders['target_bbs'],
            region_proposals, cls_ordered_lrp, target_cls_placeholder)

    # Adds 'summary' to tensor.
    tensors = {
        'placeholders': {
            # ==== REG PART:

            # Give into reg lstm as input.
            'groundtruth_bbs': input_data_placeholders['groundtruth_bbs'],

            # Target for reg lstm.
            'target_bbs': input_data_placeholders['target_bbs'],

            # Not used.
            'images': input_data_placeholders['images'],

            # ==== CLS PART:

            # Build input vector for cls lstm from:
            'region_proposals': region_proposals,
            'ordered_last_region_proposals': cls_ordered_lrp,

            # Target for cls.
            'target_cls': target_cls_placeholder
        },

        'lstm': lstm_tensors,
        'cls': classifier_tensors,
        'combined': combined_loss_tensors
    }
    add_train_summaries(tensors)

    return tensors


def build_input_pipeline():
    # The bb coordinates and the images are normalized.
    input_data, input_handles = input_pipeline.sequences.tfrecord_reader.build()

    return input_data, input_handles
