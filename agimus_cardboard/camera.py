import time
import argparse
from rclpy.node import Node
from agimus_cardboard.CameraBasler import CameraBasler
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image as MsgImage
from agimus_cardboard.cardboard_parameters import cardboard_params
import yaml
import agimus_cardboard.crbtools as crb
import geometry.all as g
import numpy as np
import cv2
import cv_bridge


class Camera(Node):
    """"""

    def __init__(self, calib_file: str, simulate_file: str = None,
                 name="camera"):
        self.initialized = False
        super().__init__(name)

        self.bridge = cv_bridge.CvBridge()
        # parameters
        self.param_listener = cardboard_params.ParamListener(self)
        self.params = self.param_listener.get_params()

        serial_number = self.params.camera.serial
        exposure_us = int(self.params.camera.exposure_time * 1e6)

        self.publish_raw = self.params.camera.publish_raw
        self.publish_plane = self.params.camera.publish_plane
        self.period = self.params.camera.period
        packet_size = self.params.camera.packet_size

        # calibration
        with open(calib_file, 'r') as fh:
            calib_data = yaml.load(fh, Loader=yaml.SafeLoader)

        self.calib = crb.Calib.from_dict(calib_data)
        self.calib_u = self.calib.get_undistorted()

        rot = g.a2r(self.calib.r_vec.flatten())
        mat_h = self.calib_u.mat_k @ rot.T @ np.linalg.inv(self.calib_u.mat_k)
        self.mat_h, self.new_w, self.new_h = crb.im_fit_h(mat_h, self.calib_u.w,
                                                          self.calib_u.h)

        # publishers
        if self.publish_raw:
            self.publisher_raw_ = self.create_publisher(
                MsgImage,
                "image_raw",
                qos_profile=QoSProfile(
                    depth=2,
                    reliability=ReliabilityPolicy.RELIABLE))

        if self.publish_plane:
            self.publisher_plane_ = self.create_publisher(
                MsgImage,
                "image_plane",
                qos_profile=QoSProfile(
                    depth=2,
                    reliability=ReliabilityPolicy.RELIABLE))

        # camera or simulated image
        self._frame_no = -1
        self.img_u = None
        self.timestamp = self.get_clock().now().to_msg()

        if simulate_file is not None:
            self.simulated_image = cv2.imread(simulate_file)
            self.simulated_image = cv2.cvtColor(self.simulated_image,
                                                cv2.COLOR_RGB2GRAY)
            self.cam = None

            self.get_logger().info(
                "Camera initialized:\n" +
                f"  simulated_image={simulate_file}\n" +
                f"  raw={self.publish_raw}\n" +
                f"  plane={self.publish_plane}")
        else:
            self.simulated_image = None

            assert len(serial_number) > 0
            assert exposure_us > 0

            self.cam = CameraBasler(serial_number=serial_number,
                                    exposure_us=exposure_us,
                                    packet_size=packet_size,)

            self.get_logger().info(
                "Camera initialized:\n" +
                f"  serial={serial_number}\n" +
                f"  exposure={exposure_us}\n" +
                f"  packet_size={packet_size}\n" +
                f"  raw={self.publish_raw}\n" +
                f"  plane={self.publish_plane}")

    def capture(self):
        self._frame_no += 1
        self.timestamp = self.get_clock().now().to_msg()

        t0 = time.time()
        if self.simulated_image is not None:
            self.img = self.simulated_image
        else:
            self.img = self.cam.capture_image()
        t1 = time.time() - t0
        t2 = 0.0
        t3 = 0.0

        if self.publish_raw:
            t = time.time()

            msg = self.bridge.cv2_to_imgmsg(self.img, "mono8")
            msg.header.stamp = self.timestamp

            self.publisher_raw_.publish(msg)

            t2 = time.time() - t

        self.img_u = self.calib.transform_image_to(self.calib_u, self.img)

        if self.publish_plane:
            t = time.time()

            msg = self.bridge.cv2_to_imgmsg(self.img_u, "mono8")
            msg.header.stamp = self.timestamp

            self.publisher_plane_.publish(msg)

            t3 = time.time() - t

        self.get_logger().debug(
            f"Frame {self._frame_no} [{t1:.2f}/{t2:.2f}/{t3:.2f} s]")

        t = time.time() - t0
        if t < self.period:
            time.sleep(self.period - t1)


def main(args=None):
    node = None
    try:
        rclpy.init(args=args)

        args = rclpy.utilities.remove_ros_args(args)

        parser = argparse.ArgumentParser("camera")
        parser.add_argument("--calib-file", type=str, required=True)
        parser.add_argument("--simulate-file", type=str, required=False)
        args = parser.parse_args(args[1:])  # skip the script name

        node = Camera(**vars(args))

        while True:
            node.capture()

    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
