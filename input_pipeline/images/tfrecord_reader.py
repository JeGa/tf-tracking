import tensorflow as tf

import input_pipeline.images.proto_helper as proto_helper
import util.global_config as global_config


def parse_proto(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features=proto_helper.key_to_type.mapping)

    image = tf.to_float(tf.image.decode_jpeg(features['image/encoded']))

    xmin = tf.sparse_tensor_to_dense(features['image/object/bbox/xmin'])
    ymin = tf.sparse_tensor_to_dense(features['image/object/bbox/ymin'])
    xmax = tf.sparse_tensor_to_dense(features['image/object/bbox/xmax'])
    ymax = tf.sparse_tensor_to_dense(features['image/object/bbox/ymax'])

    name = tf.sparse_tensor_to_dense(features['image/source_id'], default_value='')

    bbs = tf.stack([ymin, xmin, ymax, xmax], axis=1)  # (number_boxes, 4)

    labels = tf.to_float(tf.sparse_tensor_to_dense(features['image/object/class/label']))
    labels = tf.expand_dims(labels, -1)  # (number_boxes, 1)
    groundtruth = tf.concat([bbs, labels], axis=1)  # (number_boxes, 5)

    gt_detections_nr = tf.shape(groundtruth)[0]

    # TODO: Also resize the BBs accordingly.
    resize_height = global_config.cfg['resize_height']
    resize_width = global_config.cfg['resize_width']

    image = tf.image.resize_images(tf.div(image, 255), [resize_height, resize_width])
    image.set_shape([resize_height, resize_width, 3])

    return image, groundtruth, gt_detections_nr, name


def create_datasets(training_record_file, validation_record_file):
    training_dataset = tf.contrib.data.TFRecordDataset([training_record_file])
    validation_dataset = tf.contrib.data.TFRecordDataset([validation_record_file])

    paddshapes = ([480, 854, 3], [10, 5], [], [-1])

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


def read_graph(training_record_file, validation_record_file):
    training_dataset, validation_dataset = create_datasets(training_record_file, validation_record_file)

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

    image, groundtruth, gt_detections_nr, name = iterator.get_next()
    input_data = {
        'image': image,
        'groundtruth': groundtruth,
        'gt_detections_nr': gt_detections_nr,
        'name': name
    }

    return input_data, input_handles
