import tensorflow as tf

import input_pipeline.sequences.proto_helper as proto_helper
import util.global_config as global_config

PLAYERS = 10


def _bb(bb_feature, sequence_length):
    # For the feature_list one dimension is added for the sequences.
    bbs = tf.reshape(bb_feature, [sequence_length, 4 * PLAYERS])

    # Crop the sequence if it is larger than global_config.cfg['backprop_step_size'].
    cropped_bbs = bbs[:global_config.cfg['backprop_step_size'], :]

    return cropped_bbs


def parse_proto(serialized_example):
    # The first dict contains the context key/values. The second dict contains the feature_list key/values.
    context, feature_list = tf.parse_single_sequence_example(
        serialized_example,
        context_features=proto_helper.key_to_type.context_mapping,
        sequence_features=proto_helper.key_to_type.sequence_mapping)

    sequence_length = tf.to_int32(context['sequence_length'])

    # TODO players = tf.to_int32(context['players'])

    def decode(raw_bytes):
        return tf.div(tf.to_float(tf.image.decode_jpeg(raw_bytes, channels=3)), 255.0)

    raw_bytes_sequence = feature_list['image_data']
    image_decoded = tf.map_fn(decode, raw_bytes_sequence, dtype=tf.float32,
                              back_prop=False, parallel_iterations=10)

    # Crop the sequence if it is larger than global_config.cfg['backprop_step_size'].
    cropped_image_sequence = image_decoded[:global_config.cfg['backprop_step_size'], :]

    # For the feature_list one dimension is added for the sequences.
    cropped_bbs = _bb(feature_list['bbs'], sequence_length)
    cropped_bbs_targets = _bb(feature_list['bbs_targets'], sequence_length)

    return cropped_bbs, cropped_bbs_targets, cropped_image_sequence


def create_datasets(training_record_file, validation_record_file):
    training_dataset = tf.contrib.data.TFRecordDataset([training_record_file])
    validation_dataset = tf.contrib.data.TFRecordDataset([validation_record_file])

    paddshapes = (
        [global_config.cfg['backprop_step_size'], 4 * PLAYERS],
        [global_config.cfg['backprop_step_size'], 4 * PLAYERS],
        [global_config.cfg['backprop_step_size'], -1, -1, -1])

    training_dataset = training_dataset.map(parse_proto)
    training_dataset = training_dataset.padded_batch(global_config.cfg['batch_size'],
                                                     padded_shapes=paddshapes)
    training_dataset = training_dataset.repeat(1)
    training_dataset = training_dataset.shuffle(100)

    validation_dataset = validation_dataset.map(parse_proto)
    validation_dataset = validation_dataset.padded_batch(global_config.cfg['batch_size'],
                                                         padded_shapes=paddshapes)
    validation_dataset = validation_dataset.repeat(1)

    return training_dataset, validation_dataset


def build():
    training_dataset, validation_dataset = create_datasets(global_config.cfg['input_data_training'],
                                                           global_config.cfg['input_data_testing'])

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.contrib.data.Iterator.from_string_handle(handle,
                                                           training_dataset.output_types,
                                                           training_dataset.output_shapes)

    training_iterator = training_dataset.make_initializable_iterator()
    validation_iterator = validation_dataset.make_initializable_iterator()

    training_handle = training_iterator.string_handle()
    validation_handle = validation_iterator.string_handle()

    input_handles = {
        'tr_it': training_iterator,
        'val_it': validation_iterator,
        'tr_h': training_handle,
        'val_h': validation_handle,
        'h': handle
    }

    groundtruth_bbs, target_bbs, images = iterator.get_next()

    groundtruth_bbs.set_shape((global_config.cfg['batch_size'],
                               global_config.cfg['backprop_step_size'],
                               4 * PLAYERS))

    target_bbs.set_shape((global_config.cfg['batch_size'],
                          global_config.cfg['backprop_step_size'],
                          4 * PLAYERS))

    images.set_shape((global_config.cfg['batch_size'],
                      global_config.cfg['backprop_step_size'],
                      None, None, 3))

    input_data = {
        'groundtruth_bbs': groundtruth_bbs,
        'target_bbs': target_bbs,
        'images': images
    }

    return input_data, input_handles
