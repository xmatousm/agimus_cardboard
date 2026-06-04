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

from agimus_controller_mod.trajectories.line_cartesian_space import \
    LineSegmentCartesianSpace

import \
    agimus_controller_mod_ros.trajectory_builders.line_cartesian_space as line_builder

from agimus_controller_mod_ros.trajectory_builders.trajectory_builder import (
    get_weights, get_all_weights
)

class HoleInsertPlanner(Node):
    """"""

    def __init__(self):
        super().__init__("hole_insert_planner")

        # parameters
        self.param_listener = hole_planner_params.ParamListener(self)
        self.params = self.param_listener.get_params()

        self.delta_z = self.params.delta_z
        init_pose = self.params.init_pose
        assert len(init_pose) == 3, "init pose length must be 3"
        self.init_pose = np.array(init_pose)
        self.ee_frame_name = self.params.ee_frame_name

        self.weights = get_all_weights(self.params, 7, self.ee_frame_name)

        self.seg_line = LineSegmentCartesianSpace(
            self.ee_frame_name,
            self.weights,
            goal_tolerance_boost=self.params.goal_tolerance_boost,
            goal_weight_boost=self.params.goal_weight_boost)

        self.goal_param = {
            'normal': {
                'speed': self.params.speed,
                'duration': 1.0, # TODO min duration -> param
                'goal_tolerance': self.params.goal_tolerance,
                'w_pose': get_weights(self.params.w_pose, 6),
            },
            'prepare': {
                'speed': self.params.speed,
                'duration': None,
                'goal_tolerance': self.params.goal_tolerance_prepare,
                'w_pose': get_weights(self.params.w_pose_prepare, 6)
            },
            'hole':{
                'speed': self.params.speed_hole,
                'duration': None,
                'goal_tolerance': self.params.goal_tolerance_prepare / 3,
                'w_pose': get_weights(self.params.w_pose_hole, 6),
            },
        }

        self.x_min = self.params.x_min
        self.x_max = self.params.x_max
        self.y_min = self.params.y_min
        self.y_max = self.params.y_max

        self.subscriber = self.create_subscription(
            Hole,
            "hole",
            self.hole_callback,
            qos_profile=QoSProfile(depth=2,
                                   reliability=ReliabilityPolicy.RELIABLE),
        )

        # debug publisher for working area
        self._marker_publisher = self.create_publisher(
            Marker,
            "working_area_marker",
            qos_profile=QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
            ),
        )

        self.hole_needed = None
        self.ids = []
        self.p1s = []
        self.p2s = []

        self.action_client = ActionClient(self, TrajectoryAction,
                                          'trajectory_goal')

    def hole_callback(self, msg_in: Hole):
        self.hole_needed = [msg_in.id, msg_in.pose1, msg_in.pose2]
        self.get_logger().debug(f'Hole {self.hole_needed}')

    def one_point(self, p, angle, key: str):
        goal = TrajectoryAction.Goal()
        g = goal.goal
        gpar = self.goal_param[key]

        self.seg_line.x_to = p
        self.seg_line.goal_tolerance = gpar['goal_tolerance']
        self.seg_line.velocity = gpar['speed']
        self.seg_line.duration = gpar['duration']

        line_builder.LineCartesianSpace().to_goal(self.seg_line, g)

        g.rot_rpy = [0.0, 3.1415, angle]  # TODO move into to_goal
        g.w_pose = list(gpar['w_pose'])

        return goal

    def send_point(self, goal, pid, name):
        self.get_logger().error(f"Sending goal {pid}:{name}")

        goal.goal.id = pid
        # send goal
        goal_future = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_future)
        goal_handle = goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            rclpy.shutdown()

        self.get_logger().info('Goal accepted')

        # wait for result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        self.get_logger().info(f'Result: {result}')

    def read_holes_needed(self, wait=True):
        # always wait for a message; if required, wait for the nonempty hole
        self.ids = []
        while len(self.ids) == 0:
            while self.hole_needed is None:
                rclpy.spin_once(self)

            self.ids, self.p1s, self.p2s = self.hole_needed
            self.hole_needed = None

            if not self.holes_in_range():
                self.get_logger().error('Some hole not in range')
                self.ids, self.p1s, self.p2s = [], [], []

            if not wait:
                return

    def holes_in_range(self) -> bool:
        for i in range(len(self.ids)):
            if not (self.x_min < self.p1s[i * 3] < self.x_max):
                return False
            if not (self.y_min < self.p1s[i * 3 + 1] < self.y_max):
                return False
            if not (self.x_min < self.p2s[i * 3] < self.x_max):
                return False
            if not (self.y_min < self.p2s[i * 3 + 1] < self.y_max):
                return False
        return True

    def select_hole(self, select_id):
        inx = None
        # select the proper hole or return None
        for i in range(len(self.ids)):
            if self.ids[i] == select_id:
                inx = i
                break

        if inx is None:  # proper hole not found
            return None, None, None

        p1 = np.array(self.p1s[inx * 3:inx * 3 + 3])
        p2 = np.array(self.p2s[inx * 3:inx * 3 + 3])

        angle = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])

        if angle > np.pi / 2 or angle < -np.pi / 2:
            p1, p2 = p2, p1
            angle = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])

        # TODO treat orientation
        return p1, p2, angle

    def process_one(self, select_id):
        p1, p2, angle = self.select_hole(select_id)
        p_mid = (p1 + p2) / 2.0
        p_up = p_mid.copy()
        p_dn = p_mid.copy()

        p_up[2] += self.delta_z
        p_mid[2] += 0.5 * self.delta_z

        self.publish_working_area(1.0, 1.0, 1.0)

        # move above the hole beginning
        self.send_point(self.one_point(p_up, angle, 'normal'), 1, "up-")

        # increase weights then move down
        self.send_point(self.one_point(p_up, angle, 'prepare'), 2, "up")
        self.send_point(self.one_point(p_mid, angle, 'hole'), 3, "mid")
        self.send_point(self.one_point(p_dn, angle, 'hole'), 3, "dn")

        # move up, decrease weights
        self.send_point(self.one_point(p_up, angle, 'prepare'), 4, "up")
        self.send_point(self.one_point(p_up, angle, 'normal'), 5, "up-2")

    def publish_working_area(self, r, g, b):
        marker = Marker()

        marker.header.frame_id = 'lbr_link_0'  # TODO
        marker.ns = "working_area"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.01

        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0

        marker.lifetime = DurationMsg(sec=2)
        zw = 0.1  # TODO
        marker.points.append(Point(x=self.x_min, y=self.y_min, z=zw))
        marker.points.append(Point(x=self.x_min, y=self.y_max, z=zw))
        marker.points.append(Point(x=self.x_max, y=self.y_max, z=zw))
        marker.points.append(Point(x=self.x_max, y=self.y_min, z=zw))
        marker.points.append(Point(x=self.x_min, y=self.y_min, z=zw))

        self._marker_publisher.publish(marker)

    def process(self):
        self.get_logger().info("Processing")

        # working area markers
        self.publish_working_area(1.0, 1.0, 1.0)
        self.action_client.wait_for_server()

        # move to the initial pose
        self.send_point(self.one_point(self.init_pose, 0.0, 'normal'), 0, "out")

        self.publish_working_area(1.0, 0.0, 0.0)

        self.hole_needed = None
        self.read_holes_needed(wait=True)
        assert len(self.ids) > 0

        ids = self.ids
        for current_id in ids:
            self.process_one(current_id)

        self.get_logger().info("Done")


def main(args=None):
    node = None
    try:
        rclpy.init(args=args)
        node = HoleInsertPlanner()
        while True:
            node.process()

    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
