from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.node import Node
from rclpy.action import ActionClient
from agimus_cardboard.hole_planner_parameters import hole_planner_params

from agimus_controller_mod_msgs.msg import Hole
from agimus_controller_mod_msgs.action import TrajectoryAction
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import Point

from agimus_controller.trajectory import TrajectoryPointWeights

from agimus_controller_mod.trajectories.line_cartesian_space import \
    LineSegmentCartesianSpace

import \
    agimus_controller_mod_ros.trajectory_builders.line_cartesian_space as line_builder

from agimus_controller_mod_ros.trajectory_builders.trajectory_builder import (
    get_weights, get_all_weights
)


class HoleSelected:
    u: np.ndarray
    u1: np.ndarray
    u2: np.ndarray
    angle: float
    hole_id: int
    filled: bool

    def __init__(self, u1, u2, hole_id, filled,
                 normalize_angle: bool = False):
        self.u1 = u1
        self.u2 = u2
        self.hole_id = hole_id
        self.filled = filled
        self.u = (u1 + u2) / 2

        self.angle = float(np.arctan2(u1[1] - u2[1], u1[0] - u2[0]))

        if normalize_angle and (self.angle > np.pi / 2 or
                                self.angle < -np.pi / 2):
            self.u1, self.u2 = u2, u1
            self.angle = float(np.arctan2(u2[1] - u1[1], u2[0] - u1[0]))


@dataclass
class GoalParam:
    speed: Optional[float]
    duration: Optional[float]
    goal_tolerance: float
    goal_tolerance_boost: float
    goal_weight_boost: float
    weights: TrajectoryPointWeights

    def __init__(self, sub_params,
                 nq, ee_frame_name: str,
                 default=None):
        self.speed = sub_params.speed
        self.duration = sub_params.duration
        self.goal_tolerance = sub_params.goal_tolerance
        self.goal_tolerance_boost = sub_params.goal_tolerance_boost
        self.goal_weight_boost = sub_params.goal_weight_boost

        # not defined weights are copied from the default
        self.weights = get_all_weights(
            sub_params, nq,
            ee_frame_name,
            default)


class HolePlannerBase(Node):
    """"""

    def __init__(self, name: str, robot_name: str, hole_topic_list: list[str]):
        super().__init__(name)

        self._point_id = 0
        self._last_rpy = [0., 0., 0.]
        self.rps = 1.0  # radians per second for r, p, y (independently)

        # parameters, attributes
        self.param_listener = hole_planner_params.ParamListener(self)
        self.params = self.param_listener.get_params()

        self.freeze_holes_working = self.params.freeze_holes
        self.freeze_holes = False

        assert len(self.params.init_pose) == 3, "init pose length must be 3"
        self.init_pose = np.array(self.params.init_pose)
        self.ee_frame_name = self.params.ee_frame_name

        nq = 7  # TODO

        self.goal_param: dict[str, GoalParam] = {
            # change of weights at the same pos
            #'normal_weights': GoalParam(self.params.weights, nq,
            #                            self.ee_frame_name,
            #                            default=self.params.weights),
            #    duration=0.5,  # TODO min duration -> param
            'normal': GoalParam(self.params.params_normal, nq,
                                self.ee_frame_name),
            'prepare': GoalParam(self.params.params_prepare, nq,
                                self.ee_frame_name,
                                 default=self.params.params_normal),
            'pre_hole': GoalParam(self.params.params_pre_hole, nq,
                                 self.ee_frame_name,
                                 default=self.params.params_normal),
            'hole': GoalParam(self.params.params_hole, nq,
                                 self.ee_frame_name,
                                 default=self.params.params_normal),
        }

        self.seg_line = LineSegmentCartesianSpace(self.ee_frame_name)
        # some self.seg_line parameters set later using goal parameters

        self.seg_line.reg_q = self.params.reg_q

        # subscribers for detected holes
        self._holes = {}  # buffer for hole message
        self._subscribers = {}
        for k in hole_topic_list:
            assert not k in self._subscribers, f"Dubplicite key: {k}"
            self._holes[k] = None
            self._subscribers[k] = self.create_subscription(
                Hole, k,
                lambda msg, topic=k: self._holes_callback(topic, msg),
                qos_profile=QoSProfile(depth=2,
                                       reliability=ReliabilityPolicy.RELIABLE),
            )

        # debug publisher for working area marker
        self.area_color = {
            'idle': (1.0, 1.0, 1.0)
        }

        self._marker_publisher = self.create_publisher(
            Marker,
            "working_area_marker",
            qos_profile=QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
            ),
        )

        # client for trajectory actions
        self._action_client = ActionClient(self, TrajectoryAction,
                                           f'{robot_name}/trajectory_goal')

    def _holes_callback(self, topic, msg_in: Hole):
        if self.freeze_holes:
            save_filled = self._holes[topic][3]

        self._holes[topic] = [msg_in.id, msg_in.pose1, msg_in.pose2,
                              msg_in.filled]

        if self.freeze_holes and len(self._holes[topic][0]):
            self._holes[topic][3] = save_filled

        self.get_logger().debug(f'{topic} {self._holes[topic][0]}')

    def clean_holes(self, topic: Optional[str] = None):
        if self.freeze_holes:
            return

        if topic is None:
            for topic in self._holes:
                self._holes[topic] = None
        else:
            self._holes[topic] = None

    def one_point(self, p, angle, key: str,
                  dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):
        goal = TrajectoryAction.Goal()
        g = goal.goal
        gpar = self.goal_param[key]

        self.seg_line.x_to = p.copy()
        self.seg_line.x_to[0] += dx
        self.seg_line.x_to[1] += dy
        self.seg_line.x_to[2] += dz
        self.seg_line.goal_tolerance = gpar.goal_tolerance
        self.seg_line.velocity = gpar.speed
        self.seg_line.duration = gpar.duration
        self.seg_line.weights = gpar.weights
        self.seg_line.goal_tolerance_boost = gpar.goal_tolerance_boost
        self.seg_line.goal_weight_boost = gpar.goal_weight_boost

        line_builder.LineCartesianSpace().to_goal(self.seg_line, g)

        g.rot_rpy = [0.0, 3.1415, angle]  # TODO move into to_goal

        return goal

    def send_point_nowait(self, goal, name):
        self._point_id += 1

        rpy_delta = np.max(np.abs(np.array(self._last_rpy) -
                                  np.array(goal.goal.rot_rpy)))

        if rpy_delta > 0.0:
            rot_duration_min = rpy_delta / self.rps
            if goal.goal.duration < 0:
                goal.goal.duration = rot_duration_min
            else:
                goal.goal.duration = max(rot_duration_min, goal.goal.duration)

        self._last_rpy = goal.goal.rot_rpy

        self.get_logger().info(f"Sending goal {self._point_id}:{name}")

        goal.goal.id = self._point_id
        # send goal
        goal_future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_future)
        goal_handle = goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            rclpy.shutdown()

        self.get_logger().debug('Goal accepted')
        result_future = goal_handle.get_result_async()
        return result_future

    def wait_for_result(self, result_future) -> TrajectoryAction.Result:
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        self.get_logger().debug(f'Result: {result}')
        return result

    def send_point(self, goal, name) -> float:
        result_future = self.send_point_nowait(goal, name)
        result = self.wait_for_result(result_future)
        return result.distance

    def send_one_point(self, p, angle, key: str, name: str,
                       dx: float = 0.0, dy: float = 0.0,
                       dz: float = 0.0) -> float:
        return self.send_point(self.one_point(p, angle, key, dx, dy, dz), name)

    def _in_range(self, u, angle) -> bool:
        if not (self.params.x_min < u[0] < self.params.x_max):
            return False
        if not (self.params.y_min < u[1] < self.params.y_max):
            return False
        if not -3.0 < angle < 3.0:  # TODO better
            return False
        return True

    def _select_hole(self, topic: str, filled=None, hole_id=None,
                     normalize_angle: bool = False
                     ) -> Optional[HoleSelected]:
        ids, u1s, u2s, is_filled = self._holes[topic]
        if len(ids) < 1:
            return None

        u1s = np.array(u1s).reshape((-1, 3)).T
        u2s = np.array(u2s).reshape((-1, 3)).T

        for i in range(len(ids)):
            if filled is not None and is_filled[i] != filled:
                continue

            if hole_id is not None and ids[i] != hole_id:
                continue

            hs = HoleSelected(u1s[:, i], u2s[:, i], ids[i], is_filled[i],
                              normalize_angle)

            if self._in_range(hs.u1, hs.angle) and self._in_range(hs.u2,
                                                                  hs.angle):
                return hs

        return None

    def read_select_hole_optional(self, topic: str, filled=None,
                                  hole_id=None, normalize_angle: bool = False,
                                  ) -> Optional[HoleSelected]:

        # waits for the message, returns selected hole or none
        while self._holes[topic] is None:
            self.get_logger().warning(f'Waiting for a {topic}')
            rclpy.spin_once(self)

        hole_sel = self._select_hole(topic, filled=filled, hole_id=hole_id,
                                     normalize_angle=normalize_angle)

        return hole_sel

    def read_select_hole(self, topic: str, filled=None,
                         hole_id=None, normalize_angle: bool = False,
                         ) -> HoleSelected:
        # waits for the message and for the selected hole
        while True:
            hole_sel = self.read_select_hole_optional(
                topic, filled=filled, hole_id=hole_id,
                normalize_angle=normalize_angle)

            if hole_sel is not None:
                return hole_sel
            else:
                self.get_logger().debug(f'Cleaning {topic}')
                self.clean_holes(topic)

    def current_hole_ids(self, topic: str) -> list[int]:
        return self._holes[topic][0]

    def publish_working_area(self, key: str):
        marker = Marker()

        marker.header.frame_id = 'lbr_link_0'  # TODO
        marker.ns = "working_area"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.01

        marker.color.r = self.area_color[key][0]
        marker.color.g = self.area_color[key][1]
        marker.color.b = self.area_color[key][2]
        marker.color.a = 1.0

        marker.lifetime = DurationMsg(sec=2)
        zw = 0.1  # TODO
        marker.points.append(
            Point(x=self.params.x_min, y=self.params.y_min, z=zw))
        marker.points.append(
            Point(x=self.params.x_min, y=self.params.y_max, z=zw))
        marker.points.append(
            Point(x=self.params.x_max, y=self.params.y_max, z=zw))
        marker.points.append(
            Point(x=self.params.x_max, y=self.params.y_min, z=zw))
        marker.points.append(
            Point(x=self.params.x_min, y=self.params.y_min, z=zw))

        self._marker_publisher.publish(marker)

    def process_one(self):
        raise NotImplementedError

    def do_work(self):
        self.get_logger().info("Init")

        # working area markers
        self.publish_working_area('idle')
        self._action_client.wait_for_server()

        # move to the initial pose
        self._point_id = -1
        self._last_rpy = [0., 0., 0.]
        self.send_one_point(self.init_pose, 0.0, 'normal', 'out')

        # clean all detected holes; they should be visible now
        self.clean_holes()

        while True:
            self.publish_working_area('idle')
            self._point_id = 0
            self.process_one()
