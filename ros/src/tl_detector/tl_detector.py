#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
# from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree

import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.camera_image = None
        self.waypoint_tree = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        # self.light_classifier = TLClassifier()
        # self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
            
    def traffic_cb(self, msg):
        self.lights = msg.lights

        # if self.waypoints == None:
        #     # Need the waypoints for further processing
        #     return
    
        # # List of positions of the lines to stop in front of intersections
        # self.stop_line_positions = self.config['stop_line_positions']
    
        # # Associate the stop lines with the traffic lights. This is done once
        # if self.adjust == False:
        #     self.adjustLight()
        #     self.adjust = True

        # # Get the closest waypoint to the position of the car
        # if self.pose:
        #     car_wp_index = self.get_closest_waypoint(self.pose.pose.position.x,
        #                                              self.pose.pose.position.y)
        #     rospy.loginfo("car @ %s", car_wp_index)
        # else:
        #     # Cannot continue without knowing the pose of the car itself
        #     return

        # # Locate the next upcoming red traffic light stop line waypoint index
        # closest_stop_index = len(self.waypoints) - 1
        # for light in self.lights:
        #     # Green light is not an obstacle!
        #     if light.state == TrafficLight.RED:
                
        #         # Get the stop line from the light
        #         light_x = light.pose.pose.position.x
        #         light_y = light.pose.pose.position.y
        #         stop_line = self.light_stops[(light_x, light_y)]
        #         rospy.loginfo("found red @ %s",
        #                       self.get_closest_waypoint(light_x, light_y))

        #         # Get the waypoint index closest to the stop line
        #         stop_line_x = stop_line[0]
        #         stop_line_y = stop_line[1]
        #         stop_wp_index = self.get_closest_waypoint(stop_line_x,
        #                                                   stop_line_y)
      
        #         if  stop_wp_index > car_wp_index and \
        #             stop_wp_index < closest_stop_index:
        #             closest_stop_index = stop_wp_index

        # rospy.loginfo("closest stop @ %s", closest_stop_index)
        
        # # Publish the result
        # self.upcoming_red_light_pub.publish(closest_stop_index)

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_idx = self.waypoint_tree.query([x,y],1)[1]
        return closest_idx
        # min_dist = float('inf')
        # closest_waypoint_index = 0  # Index to return

        # for i, wp in enumerate(self.waypoints):
        #     dist = pow(x - wp.x, 2) + pow(y - wp.y, 2)
            
        #     # Update the minimum distance and update the index
        #     if dist < min_dist:
        #         min_dist = dist
        #         closest_waypoint_index = i
    
        # # Return the index of the closest waypoint in self.waypoints
        # return closest_waypoint_index

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        return light.state
        # if(not self.has_image):
        #     self.prev_light_loc = None
        #     return False

        # cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # #Get classification
        # return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = -1

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

        #TODO find the closest visible traffic light (if one exists)
        diff = len(self.waypoints.waypoints)
        for i, light in enumerate(self.lights):
            #Get stop line waypoint index
            line = stop_line_positions[i]
            temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
            #find closest stop line waypoint index
            d = temp_wp_idx - car_wp_idx
            if d >= 0 and d < diff:
                diff = d
                closest_light = light
                line_wp_idx = temp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light)
            return line_wp_idx, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

        # #TODO find the closest visible traffic light (if one exists)

        # if closest_light:
        #     state = self.get_light_state(closest_light)
        #     return line_wp_idx, state
        
        # return -1, TrafficLight.UNKNOWN
    
def adjustLight(self):
    """
    adjust the closest stop line position to traffic lights.
    """
    for light in self.lights:
        # Reset the minimum distance and the index we search for
        min_dist = float('inf')
        matching_index = 0
        
        for i, stop_line_position in enumerate(self.stop_line_positions):
            # Calculate the Euclidean distance
            dx = light.pose.pose.position.x - stop_line_position[0]
            dy = light.pose.pose.position.y - stop_line_position[1]
            dist = pow(dx, 2) + pow(dy, 2)
            
            # Update the minimum distance and matching index
            if dist < min_dist:
                min_dist = dist
                matching_index = i
    
        # Correlate each light position (x, y) with the closest stop line
        x = light.pose.pose.position.x
        y = light.pose.pose.position.y
        self.light_stops[(x, y)] = self.stop_line_positions[matching_index]

        rospy.loginfo("light = (%s, %s) : stop = (%s, %s)",
                        x, y, self.stop_line_positions[matching_index][0],
                        self.stop_line_positions[matching_index][1])

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
