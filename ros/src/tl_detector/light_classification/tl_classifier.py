
import tensorflow as tf
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import cv2
import time
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        self.current_light = TrafficLight.UNKNOWN
        
        SSD_GRAPH_FILE = './frozen_inference_graph.pb'        
        self.detection_graph = self.load_graph(SSD_GRAPH_FILE)
        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        
    
    def load_graph(self,graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph
        

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        with tf.Session(graph=self.detection_graph) as sess:                
            # Actual detection.
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                                feed_dict={self.image_tensor: image_np})

            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            # confidence_cutoff = 0.8
            # # Filter boxes with a confidence score less than `confidence_cutoff`
            # boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)
            
            min_score_thresh = .5
            count = 0
            count1 = 0
            # print(scores)

            for i in range(boxes.shape[0]):
                if scores is None or scores[i] > min_score_thresh:
                    count1 += 1
                    class_name = self.category_index[classes[i]]['name']

                    # Traffic light thing
                    if class_name == 'Red':
                        count += 1

            # print(count)
            if count < count1 - count:
                self.current_light = TrafficLight.GREEN
            else:
                self.current_light = TrafficLight.RED

        return self.current_light
    