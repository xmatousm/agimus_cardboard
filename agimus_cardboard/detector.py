from typing import Optional
import argparse
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.node import Node
from builtin_interfaces.msg import Duration as DurationMsg
from sensor_msgs.msg import Image as MsgImage
import agimus_cardboard.crbtools as crb

from agimus_cardboard.cardboard_parameters import cardboard_params
from agimus_demos_msgs.msg import HoleNeeded
import yaml

import cv2
import numpy as np
import time
import cv_bridge
from agimus_cardboard.camera import Camera
import geometry.basic as gb
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point


class Detector(Node):
    """"""

    def __init__(self, template_file: str,
                 calib_file: Optional[str] = None,
                 robot_calib_file: Optional[str] = None,
                 simulate_file: Optional[str] = None):
        super().__init__("detector")

        self.camera_embedded = calib_file is not None

        self.bridge = cv_bridge.CvBridge()

        # parameters
        self.param_listener = cardboard_params.ParamListener(self)
        self.params = self.param_listener.get_params()

        self.publish_debug = self.params.detector.publish_debug
        self.publish_debug_reduce = self.params.detector.publish_debug_reduce

        self.opt = crb.Opt()

        # template
        self.template = crb.Template(template_file, self.opt)

        # robot calibration
        self.robot_calib = None
        if robot_calib_file is not None:
            with open(robot_calib_file, 'r') as fh:
                data = yaml.load(fh, Loader=yaml.SafeLoader)
                self.robot_calib = {
                    'rot': gb.a2r(data['rot_vec']),
                    'trn': np.array(data['trn']).reshape((3, 1)),
                    'scale': data['scale'],
                }

        self.img_u = None
        self.timestamp = None

        if self.camera_embedded:
            self.camera_node = Camera(calib_file, simulate_file,
                                      name="detector")

        else:
            self.subscriber = self.create_subscription(
                MsgImage,
                "image_plane",
                self.image_callback,
                qos_profile=QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                ),
            )

        # publisher for needed holes
        self._publisher = self.create_publisher(
            HoleNeeded,
            "hole_needed",
            qos_profile=QoSProfile(depth=2,
                                   reliability=ReliabilityPolicy.RELIABLE),
        )

        # debug publisher for markers for needed holes
        self._marker_publisher = self.create_publisher(
            Marker,
            "hole_needed_marker",
            qos_profile=QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
            ),
        )

        # publisher for debug image
        if self.publish_debug:
            self._publisher_debug = self.create_publisher(
                MsgImage,
                "image_segments",
                qos_profile=QoSProfile(depth=2,
                                       reliability=ReliabilityPolicy.RELIABLE),
            )

        self.get_logger().info(
            "Detector initialized:\n" +
            f"  camera={'embedded' if self.camera_embedded else 'topic'}\n" +
            f"  robot calib={'yes' if self.robot_calib is not None else 'no'}\n" +
            f"  debug={self.publish_debug}\n" +
            f"  template segments: {len(self.template.seg)}\n" +
            f"  template pairs: {len(self.template.pairs)}\n")

        self.mask = None
        self.t0 = self.get_clock().now().nanoseconds / 1e9
        self.t = self.t0

    def image_callback(self, msg_in: MsgImage):
        assert not self.camera_embedded
        cur_t = self.get_clock().now().nanoseconds / 1e9
        self.timestamp = msg_in.header.stamp
        self.t = self.timestamp.sec + self.timestamp.nanosec / 1e9

        self.get_logger().debug(
            f"Callback ({cur_t - self.t0:.2f} s, latency {cur_t - self.t:.2f} s)")

        self.img_u = self.bridge.imgmsg_to_cv2(msg_in, "mono8")
        self.image_process()

    def capture(self):
        assert self.camera_embedded
        self.camera_node.capture()
        self.timestamp = self.camera_node.timestamp
        self.img_u = self.camera_node.img_u

    def image_process(self):
        self.get_logger().debug(f"Processing")

        if self.mask is None:
            self.mask = (self.img_u > 0).astype(np.uint8)
            kernel = np.ones((5, 5), np.uint8)
            self.mask = cv2.erode(self.mask, kernel, iterations=1)

        t = time.time()
        seg, img_e, u = crb.detect_all_segments(self.img_u, self.opt, self.mask)
        t1 = time.time() - t

        self.get_logger().debug(f"Detection: {len(seg)} segments [{t1:.2f} s]")

        t0 = time.time()
        pairs = crb.segment_pairs(seg, self.opt)
        t2 = time.time() - t0

        self.get_logger().debug(f"Pairs: {len(pairs)} [{t2:.2f} s]")

        t0 = time.time()
        rot, trn, pair_inl, dq = self.template.match(seg, pairs, u, self.opt)
        t3 = time.time() - t0

        self.get_logger().debug(f"Match: {t3:.2f} s")
        if rot is not None:
            self.get_logger().debug(f"Pair inl: {pair_inl}")
            self.get_logger().debug(f"Pt inl: {len(dq)}")
            self.get_logger().debug(f"Pt RMS: {np.sqrt(dq.mean())}")
        else:
            self.get_logger().debug(f"R,t: None")

        t0 = time.time()

        hole_lines = []
        msg_ids = []
        msg_pose1 = []
        msg_pose2 = []

        marker = Marker()

        marker.header.frame_id = 'lbr_link_0'  # TODO
        marker.ns = "holes_needed"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        marker.scale.x = 0.01
        marker.scale.y = 0.01
        marker.scale.z = 0.01

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.lifetime = DurationMsg(sec=2)

        if rot is not None:
            rng = np.percentile(self.img_u,
                                [self.opt.diff['perc_min'],
                                 self.opt.diff['perc_max']])
            hole_lines, hole_ids = self.template.check_holes(
                self.img_u, rot, trn, self.opt, rng)

            if self.robot_calib is not None:
                robot_a = self.robot_calib['scale'] * self.robot_calib['rot']
                robot_b = self.robot_calib['trn']

                for i in range(len(hole_lines)):
                    msg_ids += [hole_ids[i]]
                    u = hole_lines[i]

                    x = robot_a @ gb.e2p(u) + robot_b

                    msg_pose1 += [x[0, 0], x[1, 0], x[2, 0]]
                    msg_pose2 += [x[0, 1], x[1, 1], x[2, 1]]

                    p1 = Point(x=x[0, 0], y=x[1, 0], z=x[2, 0])
                    p2 = Point(x=x[0, 1], y=x[1, 1], z=x[2, 1])

                    marker.points.append(p1)
                    marker.points.append(p2)

        self._marker_publisher.publish(marker)

        msg = HoleNeeded()
        msg.id = msg_ids
        msg.pose1 = msg_pose1
        msg.pose2 = msg_pose2
        msg.timestamp = self.timestamp.nanosec

        self._publisher.publish(msg)

        if self.publish_debug:
            img_debug = cv2.cvtColor(self.img_u, cv2.COLOR_GRAY2RGB)
            blue = img_debug[:,:,2]
            blue[img_e > 0] = 255

            for s in seg:
                cv2.line(img_debug, s.u1[:, 0], s.u2[:, 0], (255, 0, 0), 15)

            if rot is not None:
                for s in self.template.seg:
                    u1 = (rot @ s.u1 + trn).astype(int)
                    u2 = (rot @ s.u2 + trn).astype(int)

                    cv2.line(img_debug, u1[:, 0], u2[:, 0], (255, 255, 0), 5)

            for l in hole_lines:
                u = l.astype(int)
                cv2.line(img_debug, u[:, 0], u[:, 1], (55, 255, 0), 5)

            if self.publish_debug_reduce > 1:
                img_debug = img_debug[
                    ::self.publish_debug_reduce, ::self.publish_debug_reduce]

            msg = self.bridge.cv2_to_imgmsg(img_debug, "rgb8")

            msg.header.stamp = self.timestamp

            self._publisher_debug.publish(msg)
            t1 = time.time() - t0
            self.get_logger().debug(
                f"Publish: {len(seg)} segments [{t1:.2f} s]")

        self.get_logger().debug(f"")


def main(args=None):
    node = None
    try:
        rclpy.init(args=args)

        args = rclpy.utilities.remove_ros_args(args)

        parser = argparse.ArgumentParser("detector")
        parser.add_argument("--template-file", type=str, required=True)
        parser.add_argument("--calib-file", type=str, required=True)
        parser.add_argument("--robot-calib-file", type=str, required=False)
        parser.add_argument("--simulate-file", type=str, required=False)
        args = parser.parse_args(args[1:])  # skip the script name

        node = Detector(**vars(args))

        if node.camera_embedded:
            while True:
                node.capture()
                node.image_process()
        else:
            rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
