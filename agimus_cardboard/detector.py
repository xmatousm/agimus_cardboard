from typing import Optional
import argparse
import rclpy
from rclpy.impl.logging_severity import LoggingSeverity
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.node import Node
from builtin_interfaces.msg import Duration as DurationMsg
from sensor_msgs.msg import Image as MsgImage
import agimus_cardboard.crbtools as crb
import agimus_cardboard.draw as draw

from agimus_cardboard.detector_parameters import detector_params
from agimus_controller_mod_msgs.msg import Hole
import yaml

import cv2
import numpy as np
import time
import cv_bridge
from agimus_cardboard.camera import Camera
import geometry.basic as gb
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('qtagg')

class Detector(Node):
    """"""

    def __init__(self, template_file: str,
                 calib_file: str,
                 robot_calib_file: Optional[str] = None,
                 simulate_file: Optional[str] = None,
                 mask_file: Optional[str] = None,
                 detect_holder: bool = False,
                 camera_embedded: bool = True):
        super().__init__("detector")

        self.camera_embedded = camera_embedded

        self.bridge = cv_bridge.CvBridge()

        # parameters
        self.param_listener = detector_params.ParamListener(self)
        self.params = self.param_listener.get_params()

        self.publish_debug = self.params.detector.publish_debug
        self.publish_debug_reduce = self.params.detector.publish_debug_reduce

        self.opt = crb.Opt()
        self.is_debug = self.get_logger().is_enabled_for(LoggingSeverity.DEBUG)
        self.detect_holder = detect_holder
        # template
        with open(calib_file, 'r') as fh:
            calib_data = yaml.load(fh, Loader=yaml.SafeLoader)

        calib = crb.Calib.from_dict(calib_data)
        calib_u = calib.get_undistorted()

        with open(template_file, 'r') as fh:
            tmpl_dict = yaml.load(fh, Loader=yaml.SafeLoader)

        tmpl_m = crb.TemplateMetric.from_dict(tmpl_dict)
        tmpl = crb.Template.from_metric(calib_u, self.opt, tmpl_m)
        self.template = tmpl

        # draw template if in debug mode
        if self.is_debug:
            plt.ion()
            plt.figure(1)
            draw.template_metric(tmpl_m, plt.gca())

            plt.figure(2)
            draw.template(tmpl, plt.gca())

            plt.pause(0.01)

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
                                      mask_file,
                                      name=self.get_name() + '_' + 'camera')
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
        self._hole_needed_publisher = self.create_publisher(
            Hole,
            "hole_needed",
            qos_profile=QoSProfile(depth=2,
                                   reliability=ReliabilityPolicy.RELIABLE),
        )

        # publisher for all holes
        self._hole_publisher = self.create_publisher(
            Hole,
            "hole",
            qos_profile=QoSProfile(depth=2,
                                   reliability=ReliabilityPolicy.RELIABLE),
        )

        if self.detect_holder:
            # publisher for holder
            self._holder_part_publisher = self.create_publisher(
                Hole,
                "holder_part",
                qos_profile=QoSProfile(depth=2,
                                       reliability=ReliabilityPolicy.RELIABLE),
            )

            # debug publisher for markers for holder parts
            self._holder_part_marker_publisher = self.create_publisher(
                Marker,
                "holder_part_marker",
                qos_profile=QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                    history=HistoryPolicy.KEEP_LAST,
                ),
            )


        # debug publisher for markers for needed holes
        self._hole_needed_marker_publisher = self.create_publisher(
            Marker,
            "hole_needed_marker",
            qos_profile=QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
            ),
        )

        # debug publisher for markers for all holes
        self._hole_marker_publisher = self.create_publisher(
            Marker,
            "hole_marker",
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
        self.get_logger().debug(f"Capture")
        t0 = time.time()
        assert self.camera_embedded
        self.camera_node.capture()
        self.timestamp = self.camera_node.timestamp
        self.img_u = self.camera_node.img_u
        t1 = time.time() - t0
        self.get_logger().debug(f"Capture done [{t1:.2f} s]")

    def holes_to_robot_space_msg_marker(self, hole_lines, hole_ids,
                                        marker_ns = None,
                                        marker_color = (1.0, 1.0, 0.0)):
        msg_ids = []
        msg_pose1 = []
        msg_pose2 = []

        if marker_ns is not None:
            marker = Marker()

            marker.header.frame_id = 'lbr_link_0'  # TODO
            marker.ns = marker_ns
            marker.id = 0
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD

            marker.scale.x = 0.01
            marker.scale.y = 0.01
            marker.scale.z = 0.01

            marker.color.r = marker_color[0]
            marker.color.g = marker_color[1]
            marker.color.b = marker_color[2]
            marker.color.a = 1.0

            marker.lifetime = DurationMsg(sec=2)

        if self.robot_calib is not None:
            robot_a = self.robot_calib['scale'] * self.robot_calib['rot']
            robot_b = self.robot_calib['trn']

            for i in range(len(hole_lines)):
                msg_ids += [hole_ids[i]]
                u = hole_lines[i]

                x = robot_a @ gb.e2p(u) + robot_b

                msg_pose1 += [x[0, 0], x[1, 0], x[2, 0]]
                msg_pose2 += [x[0, 1], x[1, 1], x[2, 1]]

                if marker_ns is not None:
                    p1 = Point(x=x[0, 0], y=x[1, 0], z=x[2, 0])
                    p2 = Point(x=x[0, 1], y=x[1, 1], z=x[2, 1])
                    marker.points.append(p1)
                    marker.points.append(p2)

        msg = Hole()
        msg.id = msg_ids
        msg.pose1 = msg_pose1
        msg.pose2 = msg_pose2
        msg.timestamp = self.timestamp.nanosec

        if marker_ns is not None:
            return msg, marker
        else:
            return msg

    def image_process(self):
        self.get_logger().debug(f"Processing")

        if self.mask is None:
            self.mask = (self.img_u > 0).astype(np.uint8)
            kernel = np.ones((5, 5), np.uint8)
            self.mask = cv2.erode(self.mask, kernel, iterations=1)

        if self.detect_holder:
            holder_trn = crb.detect_holder(self.img_u)

            if holder_trn is not None:
                part_lines, part_ids = crb.holder_parts(self.img_u, holder_trn)
                self.img_u = crb.holder_remove(self.img_u, holder_trn)

                msg, marker = self.holes_to_robot_space_msg_marker(
                    part_lines, part_ids, marker_ns='holder_parts')

                self._holder_part_marker_publisher.publish(marker)
                self._holder_part_publisher.publish(msg)
                self.get_logger().debug(f"Holder: {part_ids}")


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

        hole_lines_empty, hole_ids_empty = [], []
        hole_lines_full, hole_ids_full = [], []
        if rot is not None:
            rng = np.percentile(self.img_u,
                                [self.opt.diff['perc_min'],
                                 self.opt.diff['perc_max']])
            hole_lines_full, hole_ids_full, hole_lines_empty, hole_ids_empty = \
                self.template.check_holes(self.img_u, rot, trn, self.opt, rng)

        msg, marker = self.holes_to_robot_space_msg_marker(
            hole_lines_full, hole_ids_full, marker_ns='holes_needed')

        self._hole_needed_marker_publisher.publish(marker)
        self._hole_needed_publisher.publish(msg)

        msg, marker = self.holes_to_robot_space_msg_marker(
            hole_lines_empty, hole_ids_empty, marker_ns='holes',
            marker_color=(0.0, 0.0, 1.0))

        self._hole_publisher.publish(msg)
        self._hole_marker_publisher.publish(marker)

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

            for l in hole_lines_full:
                u = l.astype(int)
                cv2.line(img_debug, u[:, 0], u[:, 1], (55, 255, 0), 5)

            for l in hole_lines_empty:
                u = l.astype(int)
                cv2.line(img_debug, u[:, 0], u[:, 1], (0, 255, 255), 5)


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

        if self.is_debug:
            plt.figure(3)
            plt.clf()
            plt.imshow(self.img_u, cmap='gray', vmin=0, vmax=255)
            draw.segments(seg)
            if rot is not None:
                draw.segments(self.template.seg, rot=rot, t=trn, linewidth=2, color='w')

            # redraw figures
            plt.gcf().canvas.draw_idle()
            plt.gcf().canvas.start_event_loop(0.01)


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
        parser.add_argument("--mask-file", type=str, required=False)
        parser.add_argument("--detect-holder", action="store_true")
        parser.add_argument("--camera-embedded", action="store_true")
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
