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

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
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

        # State tracking (users path is either clear or not)
        self.path_isClear = True  # True if the path is clear, False if there is an object in the path

        self.last_printed_path_state = None  # Track last printed state

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
    string_object_location = None

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we will use the frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get the numpy array from the buffer
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_detections_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Parse the detections
    detection_count = 0
    for detection in detections:
        # Output of cameras vision
        label = detection.get_label()
        bbox = detection.get_bbox() # xmin, ymin, width, height (coordinates for bounding box)        
        confidence = detection.get_confidence()

        # Use rectangle coordinates to draw the outline of that object
        x_min = bbox.xmin()
        y_min = bbox.ymin()
        box_width = bbox.width()
        box_height = bbox.height()

        # Calculate max coordinates
        x_max = x_min + box_width
        y_max = y_min + box_height

        # Calculate the center of the bounding box
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        location = None
        # Which location the object is in (left, middle, right)
        if center_x < user_data.zone_x_min:
            location = "left"
        elif center_x > user_data.zone_x_max:
            location = "right"
        else:
            location = "middle"
        # NOTE center_y is not used since ymin and ymax are always 0 and 1 respectively

    # Set the string to print based on the location of the object after at least 4 frames of detection
    if location == "left":
        user_data.left_object_detected_frames += 1
        user_data.middle_object_detected_frames = 0
        user_data.right_object_detected_frames = 0
        if user_data.left_object_detected_frames >= 4:
            string_object_location = "left"
            user_data.path_isClear = False # subject to change
    elif location == "middle":
        user_data.middle_object_detected_frames += 1
        user_data.left_object_detected_frames = 0
        user_data.right_object_detected_frames = 0
        if user_data.middle_object_detected_frames >= 4:
            string_object_location = "middle"
            user_data.path_isClear = False # same as above
    elif location == "right":
        user_data.right_object_detected_frames += 1
        user_data.left_object_detected_frames = 0
        user_data.middle_object_detected_frames = 0
        if user_data.right_object_detected_frames >= 4:
            string_object_location = "right"
            user_data.path_isClear = False # same as above
    else:
        # No object detected in the path
        user_data.left_object_detected_frames = 0
        user_data.middle_object_detected_frames = 0
        user_data.right_object_detected_frames = 0
        user_data.no_object_detected_frames += 1
        if user_data.no_object_detected_frames >= 4:
            string_object_location = "no object detected"
            user_data.path_isClear = True
    
        #---------------------------------------------- prints to shell (command line)       
    if user_data.use_frame:
        # Note: using imshow will not work here, as the callback function is not running in the main thread
        # Let's print the detection count to the frame
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Example of how to use the new_variable and new_function from the user_data
        # Let's print the new_variable and the result of the new_function to the frame
        cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Convert the frame to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    return Gst.PadProbeReturn.OK
    #---------------------------------------------
if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
