import tensorflow as tf


class record_features:
    @staticmethod
    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def int64_list_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def bytes_feature_from_string(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

    @staticmethod
    def bytes_list_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    @staticmethod
    def bytes_list_feature_from_string(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.encode() for v in value]))

    @staticmethod
    def float_list_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class key_to_type:
    context_mapping = {
        'sequence_id': tf.VarLenFeature(tf.string),
        'sequence_length': tf.FixedLenFeature([], tf.int64),
        'players': tf.FixedLenFeature([], tf.int64)
    }
    sequence_mapping = {
        'bbs': tf.FixedLenSequenceFeature([40], tf.float32),
        'bbs_targets': tf.FixedLenSequenceFeature([40], tf.float32),
        'image_data': tf.FixedLenSequenceFeature([], tf.string)
    }
