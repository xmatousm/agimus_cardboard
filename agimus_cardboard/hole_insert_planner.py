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

from agimus_controller_mod_ros import node_utils as utils
from std_srvs.srv import SetBool


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
                'duration': 2.0,  # TODO min duration -> param
                'goal_tolerance': self.params.goal_tolerance,
                'w_pose': get_weights(self.params.w_pose, 6),
            },
            'prepare': {
                'speed': self.params.speed,
                'duration': None,
                'goal_tolerance': self.params.goal_tolerance_prepare,
                'w_pose': get_weights(self.params.w_pose_prepare, 6)
            },
            'hole': {
                'speed': self.params.speed_hole,
                'duration': None,
                'goal_tolerance': 0.0,
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

        self.subscriber = self.create_subscription(
            Hole,
            "holder_part",
            self.holder_part_callback,
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
        self.holder_part = None
        self.ids = []
        self.p1s = []
        self.p2s = []

        self.action_client = ActionClient(self, TrajectoryAction,
                                          'trajectory_goal')

        self.srv_gripper = utils.service_client(self, SetBool,
                                                "schunk_gripper/activate")

    def hole_callback(self, msg_in: Hole):
        self.hole_needed = [msg_in.id, msg_in.pose1, msg_in.pose2]
        self.get_logger().debug(f'Hole {self.hole_needed}')

    def holder_part_callback(self, msg_in: Hole):
        self.holder_part = [msg_in.id, msg_in.pose1, msg_in.pose2]
        self.get_logger().debug(f'Part {self.holder_part}')

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

    def _in_range(self, u, angle) -> bool:
        if not (self.x_min < u[0] < self.x_max):
            return False
        if not (self.y_min < u[1] < self.y_max):
            return False
        if not -3.0 < angle < 3.0:
            return False
        return True

    def _select_hole(self, data_in):
        ids, u1s, u2s = data_in
        if len(ids) < 1:
            return None, None

        u1s = np.array(u1s).reshape((-1, 3)).T
        u2s = np.array(u2s).reshape((-1, 3)).T

        for i in range(len(ids)):
            u = (u1s[:, i] + u2s[:, i]) / 2
            angle = np.arctan2(u1s[1, i] - u2s[1, i],
                               u1s[0, i] - u2s[0, i])
            if self._in_range(u, angle):
                return u, angle

        return None, None

    def read_select_hole(self):
        while True:
            self.publish_working_area(1.0, 0.0, 0.0)
            self.get_logger().info(f'Waiting for a hole')
            while self.hole_needed is None:
                rclpy.spin_once(self)

            u, angle = self._select_hole(self.hole_needed)
            self.clean_holes()
            if u is not None:
                self.publish_working_area(0.0, 1.0, 0.0)
                self.get_logger().info(f'Hole angle: {angle}')
                return u, angle


    def clean_holes(self):
        self.hole_needed = None

    def read_select_part(self):
        while True:
            self.publish_working_area(1.0, 0.5, 0.0)
            self.get_logger().info(f'Waiting for a part')
            while self.holder_part is None:
                rclpy.spin_once(self)

            u, angle = self._select_hole(self.holder_part)
            self.clean_parts()
            if u is not None:
                self.publish_working_area(0.0, 0.0, 1.0)
                self.get_logger().info(f'Part angle: {angle}')
                return u, angle

    def clean_parts(self):
        self.holder_part = None

    def gripper(self, state: bool):
        self.get_logger().info(f'Setting gripper: {state}')

        request = SetBool.Request()
        request.data = state
        self.srv_gripper.wait_for_service()
        future = self.srv_gripper.call_async(request)
        rclpy.spin_until_future_complete(self, future)

    def process_one(self):

        self.get_logger().info("Processing part")

        self.publish_working_area(1.0, 1.0, 1.0)

        # take one part
        pu, angle = self.read_select_part()

        pu_up = pu.copy()
        pu_up[2] += self.delta_z
        pu[2] -= 0.01  # TODO param - some pressure to correctly grab the part

        # move above the part
        self.send_point(self.one_point(pu_up, angle, 'normal'), 1, "part up-")

        # cardboard is visible now, clean holes, so the actual data will be used next
        self.clean_holes()

        self.gripper(True)

        # increase weights then move down
        self.send_point(self.one_point(pu_up, angle, 'prepare'), 2, "part up+")
        self.send_point(self.one_point(pu, angle, 'hole'), 4, "part down")

        # grab
        self.gripper(False)

        # move up then decrease weights
        self.send_point(self.one_point(pu_up, angle, 'hole'), 5, "part up+")
        self.send_point(self.one_point(pu_up, angle, 'normal'), 6, "part up")

        self.get_logger().info("Processing hole")

        p_mid, angle_h = self.read_select_hole()
        p_up = p_mid.copy()
        p_dn = p_mid.copy()

        p_up[2] += self.delta_z
        p_mid[2] += 0.5 * self.delta_z
        p_dn[2] -= 0.01  # TODO param - additional pressure

        if angle > 0.0 > angle_h or angle < 0.0 < angle_h:
            # move to zero angle first
            self.send_point(self.one_point(p_up, 0.0, 'normal'), 1, "rot")

        # move above the hole beginning
        self.send_point(self.one_point(p_up, angle_h, 'normal'), 1, "up-")

        # parts are visible now, clean them, the actual data will be used next
        self.clean_parts()

        # increase weights then move down
        self.send_point(self.one_point(p_up, angle_h, 'prepare'), 2, "up+")
        self.send_point(self.one_point(p_mid, angle_h, 'prepare'), 3, "mid+")
        self.send_point(self.one_point(p_dn, angle_h, 'hole'), 3, "dn")

        # release
        self.gripper(True)

        # move up, decrease weights
        self.send_point(self.one_point(p_up, angle_h, 'hole'), 4, "up")
        self.send_point(self.one_point(p_up, angle_h, 'normal'), 5, "up-")

        if angle > 0.0 > angle_h or angle < 0.0 < angle_h:
            # move to zero angle again
            self.send_point(self.one_point(p_up, 0.0, 'normal'), 1, "rot")


        self.publish_working_area(1.0, 1.0, 1.0)
        self.get_logger().info("Done")

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
        self.get_logger().info("Init")

        # working area markers
        self.publish_working_area(1.0, 1.0, 1.0)
        self.action_client.wait_for_server()

        # move to the initial pose
        self.send_point(self.one_point(self.init_pose, 0.0, 'normal'), 0, "out")
        self.gripper(False)
        self.gripper(True)

        self.clean_holes()
        self.clean_parts()

        while True:
            self.process_one()


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
