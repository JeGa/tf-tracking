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
    mapping = {
        # TODO: Different sized images?
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/source_id': tf.VarLenFeature(tf.string),

        # TODO: Different sized images?
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),

        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),

        'image/object/class/text': tf.VarLenFeature(tf.string),
        'image/object/class/label': tf.VarLenFeature(tf.int64)
    }
