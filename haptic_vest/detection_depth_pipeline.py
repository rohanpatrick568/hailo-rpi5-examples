import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import os
import setproctitle
import hailo
from hailo_apps_infra.hailo_rpi_common import app_callback_class, detect_hailo_arch, get_default_parser
from hailo_apps_infra.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    TRACKER_PIPELINE,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE,
)
from hailo_apps_infra.gstreamer_app import GStreamerApp

class GStreamerDetectionDepthApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        if parser is None:
            parser = get_default_parser()
        super().__init__(parser, user_data)
        self.app_callback = app_callback

        # Detect architecture
        if self.options_menu.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError('Could not auto-detect Hailo architecture. Please specify --arch manually.')
            self.arch = detected_arch
        else:
            self.arch = self.options_menu.arch

        # Set HEF and SO paths
        if self.arch == "hailo8":
            self.detection_hef_path = os.path.join(self.current_path, '../resources/yolov8m.hef')
            self.depth_hef_path = os.path.join(self.current_path, '../resources/scdepthv3.hef')
        else:
            self.detection_hef_path = os.path.join(self.current_path, '../resources/yolov8s_h8l.hef')
            self.depth_hef_path = os.path.join(self.current_path, '../resources/scdepthv3_h8l.hef')

        self.detection_post_process_so = os.path.join(self.current_path, '../resources/libyolo_hailortpp_postprocess.so')
        self.detection_post_function_name = "filter_letterbox"
        self.depth_post_process_so = os.path.join(self.current_path, '../resources/libdepth_postprocess.so')
        self.depth_post_function_name = "filter_scdepth"

        self.batch_size = 2
        self.thresholds_str = (
            f"nms-score-threshold=0.3 "
            f"nms-iou-threshold=0.45 "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        setproctitle.setproctitle("Hailo Detection+Depth App")
        self.create_pipeline()

    def get_pipeline_string(self):
        # Source
        source_pipeline = SOURCE_PIPELINE(self.video_source, self.video_width, self.video_height)
        # Detection branch
        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.detection_hef_path,
            post_process_so=self.detection_post_process_so,
            post_function_name=self.detection_post_function_name,
            batch_size=self.batch_size,
            additional_params=self.thresholds_str,
            name='detection_inference')
        detection_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(detection_pipeline, name='inference_wrapper_detection')
        tracker_pipeline = TRACKER_PIPELINE(class_id=1)
        # Depth branch
        depth_pipeline = INFERENCE_PIPELINE(
            hef_path=self.depth_hef_path,
            post_process_so=self.depth_post_process_so,
            post_function_name=self.depth_post_function_name,
            name='depth_inference')
        depth_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(depth_pipeline, name='inference_wrapper_depth')
        # User callback and display
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        display_pipeline = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps)

        # Parallel branches using tee
        pipeline_string = (
            f"{source_pipeline} ! tee name=t "
            f"t. ! queue ! {detection_pipeline_wrapper} ! {tracker_pipeline} ! queue name=detq "
            f"t. ! queue ! {depth_pipeline_wrapper} ! queue name=depthq "
            f"detq. ! {user_callback_pipeline} ! {display_pipeline}"
        )
        return pipeline_string