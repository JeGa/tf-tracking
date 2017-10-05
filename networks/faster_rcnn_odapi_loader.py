import numpy as np
import tensorflow as tf
import logging


class faster_rcnn_odapi:
    def __init__(self):
        self.graph = None
        self.session = None

        self.image_tensor = None
        self.boxes = None
        self.scores = None
        self.classes = None
        self.num_detections = None

    def import_graph(self, file):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.graph = detection_graph
        self.session = tf.Session(graph=self.graph)

        self.get_tensors()

    def get_tensors(self):
        # The model expects images to have shape: [1, None, None, 3]
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

    def predict(self, image):
        """
        :param session: A tensorflow session.
        :param image: Numpy array of shape [1, None, None, 3], uint8, 0 - 255.

        :return: sorted_boxes: Shape [10, 4]
        :return: sorted_scores: Shape [10]
        """
        if self.graph is None:
            logging.warning('Graph not loaded.')
            return None

        boxes, scores, num_detections = self.session.run([self.boxes, self.scores, self.num_detections],
                                                         feed_dict={self.image_tensor: image})

        # Remove the batch dimension.

        # (300, 4)
        boxes = boxes[0]
        # (300)
        scores = scores[0]

        indices = np.argsort(-scores)

        sorted_boxes = np.zeros((10, 4))
        sorted_scores = np.zeros(10)

        # Get the 10 best predictions.
        for i in range(10):
            sorted_boxes[i, :] = boxes[indices[i]]
            sorted_scores[i] = scores[indices[i]]

        return sorted_boxes, sorted_scores
