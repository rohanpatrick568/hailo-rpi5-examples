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

        self.current_object_ids = set()  # Track currently visible object IDs

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
    string_object_id = None
    location = None # NOTE js added

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we will use the frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get the numpy array from the buffer
        frame = get_numpy_from_buffer(buffer, format, width, height)

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

        # Optionally, draw label, ID, and location on the frame
        if user_data.use_frame and frame is not None:
            x_min_int = int(x_min)
            y_min_int = int(bbox.ymin())
            cv2.putText(
                frame,
                f"{label} ID:{track_id} {obj_location}",
                (x_min_int, y_min_int - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            x_max_int = int(x_min + bbox.width())
            y_max_int = int(bbox.ymin() + bbox.height())
            cv2.rectangle(frame, (x_min_int, y_min_int), (x_max_int, y_max_int), (0, 255, 0), 2)

    # Print only if the set of visible IDs has changed
    if visible_ids != user_data.current_object_ids:
        if visible_objects:
            print("Objects in view:")
            for label, track_id, obj_location in visible_objects:
                print(f"  {label} (ID: {track_id}) {obj_location}")
        else:
            print("No objects in view.")
        user_data.current_object_ids = visible_ids

    # Parse the detections
    detection_count = 0
    
        #---------------------------------------------- prints to shell (command line)       
    if user_data.use_frame:
        # Note: using imshow will not work here, as the callback function is not running in the main thread
        # Let's print the detection count to the frame
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Example of how to use the new_variable and new_function from the user_data
        # Let's print the new_variable and the result of the new_function to the frame
        cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Convert the frame to BGR
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            user_data.set_frame(frame)

    return Gst.PadProbeReturn.OK
    #---------------------------------------------
if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
