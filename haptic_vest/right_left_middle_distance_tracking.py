import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp
from hailo_apps_infra.depth_pipeline import GStreamerDepthApp

import time
import threading

class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        # Divide camera view into zones (normalized coordinates 0-1)
        self.zone_x_min = .33
        self.zone_x_max = .66
        self.zone_y_min = 0
        self.zone_y_max = 1

        # Debouncing variables
        self.left_object_detected_frames = 0
        self.middle_object_detected_frames = 0
        self.right_object_detected_frames = 0
        self.no_object_detected_frames = 0  # Number of frames with no object detected

        self.current_object_ids = set()  # Track currently visible object IDs

    def calculate_average_object_depth(self, depth_mat):
        depth_values = np.array(depth_mat).flatten()  # Flatten the array and filter out outlier pixels
        try:
            m_depth_values = depth_values[depth_values <= np.percentile(depth_values, 95)]  # drop 5% of highest values (outliers)
        except Exception as e:
            m_depth_values = np.array([])
        if len(m_depth_values) > 0:
            average_depth = np.mean(m_depth_values)  # Calculate the average depth of the pixels
        else:
            average_depth = 0  # Default value if no valid pixels are found
        return average_depth
    
# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------  

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    # Using the user_data to count the number of frames
    user_data.increment()

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)

    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        # Track visible objects and their IDs and locations
    visible_objects = []
    visible_ids = set()
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()

        # Calculate bounding box center
        x_min = bbox.xmin()
        #box_width = bbox.width()
        #x_max = x_min + box_width
        
        x_max = bbox.xmax()

        center_x = (x_min + x_max) / 2
        # NOTE y coordinate is not used in this example, but can be used for vertical tracking if needed

        # Determine location
        if center_x < user_data.zone_x_min:
            obj_location = "LEFT"
        elif center_x > user_data.zone_x_max:
            obj_location = "RIGHT"
        elif user_data.zone_x_min <= center_x <= user_data.zone_x_max:
            # If the object is within the middle zone, we can further refine the location based on its width
            obj_location = "MIDDLE"

        visible_objects.append((label, track_id, obj_location))
        visible_ids.add(track_id)

    #------------------------------------depth
        depth_mat = detection.get_objects_typed(hailo.HAILO_DEPTH_MASK)
        if len(depth_mat) > 0:  # since depth is only on detections
            detection_average_depth = user_data.calculate_average_depth(depth_mat[0].get_data())
        else:
            detection_average_depth = 0
    #------------------------------------------


    # Print only if the set of visible IDs has changed
    if visible_ids != user_data.current_object_ids:
        if visible_objects:
            print("Objects in view:")
            for label, track_id, obj_location in visible_objects:
                print(f"  {label} (ID: {track_id}) {obj_location}" 
                      f"Average Depth: {detection_average_depth:.2f}m")
        else:
            print("No objects in view.")
        user_data.current_object_ids = visible_ids


    return Gst.PadProbeReturn.OK    

#---------------------------------------------
if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()