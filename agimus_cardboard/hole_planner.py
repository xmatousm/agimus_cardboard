import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.node import Node
from rclpy.action import ActionClient
from agimus_cardboard.hole_planner_parameters import hole_planner_params

from agimus_demos_msgs.msg import HoleNeeded
from agimus_demos_msgs.action import TrajectoryGoal
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import Point

class HolePlanner(Node):
    """"""

    def __init__(self):
        super().__init__("hole_planner")

        # parameters
        self.param_listener = hole_planner_params.ParamListener(self)
        self.params = self.param_listener.get_params()

        self.delta_z = self.params.delta_z
        init_pose = self.params.init_pose
        assert len(init_pose) == 3, "init pose length must be 3"
        self.init_pose = np.array(init_pose)
        self.speed = self.params.speed
        self.speed_hole = self.params.speed_hole
        self.ee_frame_name = self.params.ee_frame_name

        self.w_q = self.get_weights(self.params.w_q, 7)
        self.w_qdot = self.get_weights(self.params.w_qdot, 7)
        self.w_qddot = self.get_weights(self.params.w_qddot, 7)
        self.w_robot_effort = self.get_weights(self.params.w_robot_effort, 7)
        self.w_pose = self.get_weights(self.params.w_pose, 6)
        self.w_pose_prepare = self.get_weights(self.params.w_pose_prepare, 6)
        self.w_pose_hole = self.get_weights(self.params.w_pose_hole, 6)

        self.goal_tolerance = self.params.goal_tolerance
        self.goal_tolerance_boost = self.params.goal_tolerance_boost
        self.goal_weight_boost = self.params.goal_weight_boost
        self.goal_tolerance_prepare = self.params.goal_tolerance_prepare

        self.saw_length = self.params.saw_length
        self.saw_height = self.params.saw_height

        self.saw_begin = self.params.saw_begin
        self.saw_end = self.params.saw_end

        self.x_min = self.params.x_min
        self.x_max = self.params.x_max
        self.y_min = self.params.y_min
        self.y_max = self.params.y_max

        self.subscriber = self.create_subscription(
            HoleNeeded,
            "hole_needed",
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

        self.action_client = ActionClient(self, TrajectoryGoal,
                                          'trajectory_goal')

    def get_weights(
            self, weights: list[float], size: int
    ) -> list[float]:
        """
        Return weights with right size if user sent only one value, otherwise
        directly returns weights.
        """
        if len(weights) == 1:
            return weights * size
        else:
            assert len(weights) == size
            return weights

    def hole_callback(self, msg_in: HoleNeeded):
        self.hole_needed = [msg_in.id, msg_in.pose1, msg_in.pose2]
        self.get_logger().debug(f'Hole needed  {self.hole_needed}')

    def one_point(self, p, angle, speed, tol, w_pose, duration=None):
        goal = TrajectoryGoal.Goal()
        g = goal.goal
        g.id = 0
        g.frame_name = "lbr_link_tool"
        g.trajectory_type = "line_cartesian_space"
        g.w_q = self.w_q
        g.w_qdot = self.w_qdot
        g.w_qddot = self.w_qddot
        g.w_robot_effort = self.w_robot_effort
        g.rot_rpy = [0.0, 3.1415, angle]
        g.speed = speed
        g.duration = -1.0 if duration is None else duration
        g.pose = [p[0], p[1], p[2]]
        g.w_pose = w_pose
        g.min_distance = tol
        g.goal_tolerance_boost = self.goal_tolerance_boost
        g.goal_weight_boost = self.goal_weight_boost

        return goal

    def one_point_saw(self, p, angle, speed, w_pose):
        goal = TrajectoryGoal.Goal()
        g = goal.goal
        g.id = 0
        g.frame_name = "lbr_link_tool"
        g.trajectory_type = "saw_line_cartesian_space"
        g.w_q = self.w_q
        g.w_qdot = self.w_qdot
        g.w_qddot = self.w_qddot
        g.w_robot_effort = self.w_robot_effort
        g.rot_rpy = [0.0, 3.1415, angle]
        g.speed = speed
        g.duration = -1.0
        g.pose = [p[0], p[1], p[2]]
        g.w_pose = w_pose
        g.s1 = self.saw_length
        g.v1 = [0.0, 0.0, self.saw_height]

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

        # beginning and ending shift
        vec = p2 - p1
        vec = vec / np.linalg.norm(vec)

        return p1 + vec * self.saw_begin, p2 - vec * self.saw_end, angle

    def process_one(self, select_id):
        p1, p2, angle0 = self.select_hole(select_id)

        if p1 is None:
            self.get_logger().error(f'Hole {select_id} disappeared at the beginning.')
            return

        self.publish_working_area(0.0, 1.0, 0.0)

        # move above the hole beginning
        p1[2] += self.delta_z
        g = self.one_point(p1, angle0, self.speed,
                           self.goal_tolerance, self.w_pose)
        self.send_point(g, 1, "1")

        # read/move again to accept the last changes
        self.read_holes_needed(wait=False)
        p1, p2, angle = self.select_hole(select_id)

        if p1 is None:
            self.get_logger().error(f'Hole {select_id} disappeared during refinement.')
            self.publish_working_area(1.0, 0.0, 0.0)
            return

        p1[2] += self.delta_z
        rps = 0.5
        duration = np.abs(angle - angle0) / 0.5

        g = self.one_point(p1, angle, self.speed,
                           self.goal_tolerance, self.w_pose, duration)
        self.send_point(g, 2, "1a")

        # increase weights then move down
        g = self.one_point(p1, angle, self.speed,
                           self.goal_tolerance_prepare, self.w_pose_prepare)
        self.send_point(g, 3, "1+")
        p1[2] -= self.delta_z
        g = self.one_point(p1, angle, self.speed,
                           self.goal_tolerance_prepare, self.w_pose_hole)
        self.send_point(g, 4, "1D")

        # saw in the hole
        g = self.one_point_saw(p2, angle, self.speed_hole, self.w_pose_hole)
        self.send_point(g, 5, "2D")

        # finishing point
        p2[2] += self.saw_height
        g = self.one_point(p2, angle, self.speed_hole,
                           self.goal_tolerance_prepare, self.w_pose_hole)
        self.send_point(g, 5, "2D")

        p2[2] -= self.saw_height
        g = self.one_point(p2, angle, self.speed_hole,
                           self.goal_tolerance_prepare, self.w_pose_hole)
        self.send_point(g, 5, "2D")

        # move up, decrease weights
        p2[2] += self.delta_z
        g = self.one_point(p2, angle, self.speed,
                           self.goal_tolerance, self.w_pose_prepare)
        self.send_point(g, 6, "2+")

        g = self.one_point(p2, angle, self.speed,
                           self.goal_tolerance, self.w_pose)
        self.send_point(g, 7, "2")

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
        zw = 0.1 # TODO
        marker.points.append(Point(x=self.x_min, y=self.y_min, z=zw))
        marker.points.append(Point(x=self.x_min, y=self.y_max, z=zw))
        marker.points.append(Point(x=self.x_max, y=self.y_max, z=zw))
        marker.points.append(Point(x=self.x_max, y=self.y_min, z=zw))
        marker.points.append(Point(x=self.x_min, y=self.y_min, z=zw))

        self._marker_publisher.publish(marker)

    def process(self):
        self.get_logger().info("Processing")

        # working area markers
        self.publish_working_area(1.0, 0.0, 0.0)
        self.action_client.wait_for_server()

        # move to the initial pose
        g = self.one_point(self.init_pose, 0.0, self.speed,
                           self.goal_tolerance, self.w_pose)
        self.send_point(g, 0, "out")

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
        node = HolePlanner()
        while True:
            node.process()

    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
