import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import os
import numpy as np
import hailo

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    app_callback_class,
)
from haptic_vest.detection_depth_pipeline import GStreamerDetectionDepthApp

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        # Divide camera view into zones (normalized coordinates 0-1)
        self.zone_x_min = .33 
        self.zone_x_max = .66
        self.zone_y_min = 0
        self.zone_y_max = 1  

        self.current_object_ids = set()  # Track currently visible object IDs

    def calculate_average_depth(self, depth_mat):
        depth_values = np.array(depth_mat).flatten()
        try:
            m_depth_values = depth_values[depth_values <= np.percentile(depth_values, 95)]
        except Exception:
            m_depth_values = np.array([])
        if len(m_depth_values) > 0:
            average_depth = np.mean(m_depth_values)
        else:
            average_depth = 0
        return average_depth

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------  
import numpy as np

def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    depth_masks = roi.get_objects_typed(hailo.HAILO_DEPTH_MASK)

    # Get the full-frame depth map if available
    depth_map = None
    if len(depth_masks) > 0:
        depth_map = depth_masks[0].get_data()  # This is a 2D numpy array

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
        x_max = bbox.xmax()
        y_min = bbox.ymin()
        y_max = bbox.ymax()
        center_x = (x_min + x_max) / 2

        # Determine location
        if center_x < user_data.zone_x_min:
            obj_location = "LEFT"
        elif center_x > user_data.zone_x_max:
            obj_location = "RIGHT"
        else:
            obj_location = "MIDDLE"

        # Calculate average depth in the detection bounding box
        detection_average_depth = 0
        if depth_map is not None:
            h, w = depth_map.shape
            x0 = int(x_min * w)
            x1 = int(x_max * w)
            y0 = int(y_min * h)
            y1 = int(y_max * h)
            crop = depth_map[y0:y1, x0:x1]
            if crop.size > 0:
                flat = crop.flatten()
                m_depth_values = flat[flat <= np.percentile(flat, 95)]
                if len(m_depth_values) > 0:
                    detection_average_depth = np.mean(m_depth_values)

        visible_objects.append((label, track_id, obj_location, detection_average_depth))
        visible_ids.add(track_id)

    if visible_ids != user_data.current_object_ids:
        if visible_objects:
            print("Objects in view:")
            for label, track_id, obj_location, detection_average_depth in visible_objects:
                print(f"  {label} (ID: {track_id}) {obj_location} Depth: {detection_average_depth:.2f}")
        else:
            print("No objects in view.")
        user_data.current_object_ids = visible_ids

    return Gst.PadProbeReturn.OK

#---------------------------------------------
if __name__ == "__main__":
    user_data = user_app_callback_class()
    app = GStreamerDetectionDepthApp(app_callback, user_data)
    app.run()