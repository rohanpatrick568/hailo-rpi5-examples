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
        
        # User path (normalized coordinates 0-1)
        self.zone_x_min = .33 
        self.zone_x_max = .66
        self.zone_y_min = 0
        self.zone_y_max = 1

        # Object detection state
        self.object_detected = False  # Flag to indicate if an object is detected in the user's path

        # Debouncing variables
        self.object_detected_frames = 0 # Number of frames with object detected
        self.no_object_detected_frames = 0 # Number of frames with no object detected

        # State tracking (users path is either clear or not)
        self.path_isClear = True  # True if the path is clear, False if there is an object in the path

        self.last_printed_path_state = None  # Track last printed state

    def new_function(self):  # New function example
        return "The meaning of life is: "

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
    string_to_print = ""
    path_state_to_print = None  # Track if we need to print

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    #frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)

        # Draw the detection area (user's path) as a rectangle
        x1 = int(user_data.zone_x_min * width)
        y1 = int(user_data.zone_y_min * height)
        x2 = int(user_data.zone_x_max * width)
        y2 = int(user_data.zone_y_max * height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box, thickness 2

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Parse the detections
    detection_count = 0
    object_in_path = False  # Track if any detection is in the path
    for detection in detections:
        # Output of cameras vision
        label = detection.get_label()
        bbox = detection.get_bbox() # coordinates for box (bounded box)
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
        center_x = x_min + (box_width / 2)
        center_y = y_min + (box_height / 2) #(y_min + (box_height / 2) - 0.22) * 1.83 (from YT)

        # Check if the center of the bounding box is within the user's path
        if user_data.zone_x_min <= center_x <= user_data.zone_x_max and user_data.zone_y_min <= center_y <= user_data.zone_y_max:
            object_in_path = True  # At least one object is in the path

    user_data.object_detected = object_in_path  # Set after checking all detections

    if user_data.object_detected:
        # If the path is not clear, increment the object_detected_frames counter
        user_data.object_detected_frames += 1
        user_data.no_object_detected_frames = 0
        # If the object_detected_frames counter exceeds a threshold, print a warning
        if user_data.object_detected_frames > 5:
            user_data.path_isClear = False
            path_state_to_print = "Warning: Object detected in the path!\n"
    else:
        # If the path is clear, increment the no_object_detected_frames counter
        user_data.no_object_detected_frames += 1
        user_data.object_detected_frames = 0
        # If the no_object_detected_frames counter exceeds a threshold, print a message
        if user_data.no_object_detected_frames > 5:
            user_data.path_isClear = True
            path_state_to_print = "Path is clear!\n"

    # Only print when the state changes or on the first run
    if path_state_to_print is not None and user_data.last_printed_path_state != user_data.path_isClear:
        print(path_state_to_print)
        user_data.last_printed_path_state = user_data.path_isClear

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

    if string_to_print:
        print(string_to_print)
    return Gst.PadProbeReturn.OK
    #---------------------------------------------
if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
