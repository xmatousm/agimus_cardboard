import numpy as np
from agimus_controller_mod_ros import node_utils as utils
from .hole_planner_base import HolePlannerBase, HoleSelected

from agimus_controller_mod_msgs.action import TrajectoryAction
from agimus_controller_mod.trajectories.saw_line_cartesian_space import \
    SawLineSegmentCartesianSpace

import \
    agimus_controller_mod_ros.trajectory_builders.saw_line_cartesian_space as saw_builder

class HolePlanner(HolePlannerBase):
    """"""

    def __init__(self):
        super().__init__("hole_planner", ["hole"])

        self.area_color['hole_wait'] = (1.0, 0.0, 0.0)
        self.area_color['hole_process'] = (0.0, 1.0, 1.0)
        self.area_color['hole_finished'] = (0.5, 0.5, 0.5)

        gpar = self.goal_param['hole']

        self.seg_saw = SawLineSegmentCartesianSpace(
            self.ee_frame_name,
            self.weights,
            tooth_length=self.params.saw_length,
            tooth_tip=np.array([0.0, 0.0, self.params.saw_height]))
        self.seg_saw.velocity = gpar['speed']
        self.seg_saw.weights.w_end_effector_poses[self.ee_frame_name] = \
            gpar['w_pose']

        self.saw_height = self.params.saw_height
        self.saw_begin = self.params.saw_begin
        self.saw_end = self.params.saw_end

    def one_point_saw(self, p, angle):
        goal = TrajectoryAction.Goal()
        g = goal.goal

        self.seg_saw.x_to = p
        saw_builder.SawLineCartesianSpace().to_goal(self.seg_saw, g)

        g.rot_rpy = [0.0, 3.1415, angle]  # TODO move into to_goal

        return goal

    def modify_hole_endpoints(self, hole: HoleSelected) -> None:
        # beginning and ending shift
        vec = hole.u2 - hole.u1
        vec = vec / np.linalg.norm(vec)
        hole.u1 = hole.u1 + vec * self.saw_begin
        hole.u2 = hole.u2 - vec * self.saw_end

    def process_one_hole(self, select_id):
        dz_up = self.params.delta_z

        self.publish_working_area('hole_wait')
        hole = self.read_select_hole_optional(
            'hole', hole_id=select_id, filled=True, normalize_angle=True)

        if hole is None:
            self.get_logger().warning(f'Hole {select_id} not available.')
            return

        self.modify_hole_endpoints(hole)

        # move above the hole beginning
        self.publish_working_area('hole_process')
        self.send_one_point(hole.u1, hole.angle, "normal", "1u-", dz=dz_up)

        # read/move again to accept the last position changes, ignore fill
        self.clean_holes('hole')
        self.publish_working_area('hole_wait')
        hole = self.read_select_hole_optional('hole', hole_id=select_id,
                                              normalize_angle=True)

        if hole is None:
            self.get_logger().error(
                f'Hole {select_id} disappeared during refinement.')
            return

        self.modify_hole_endpoints(hole)

        # move above the hole beginning again
        self.publish_working_area('hole_process')
        p1, p2, angle = hole.u1, hole.u2, hole.angle
        self.send_one_point(p1, angle, "normal", "1u-", dz=dz_up)

        self.send_one_point(p1, angle, 'normal', "1a", dz=dz_up)

        # increase weights then move down
        self.send_one_point(p1, angle, 'prepare', "1 up", dz=dz_up)
        self.send_one_point(p1, angle, "hole", "1 down")

        # saw in the hole
        g = self.one_point_saw(p2, angle)
        self.send_point(g, "2 - saw")

        # finishing point (up and down)
        self.send_one_point(p2, angle, "hole", "2 up", dz=dz_up)
        self.send_one_point(p2, angle, "hole", "2 down")

        # move up, decrease weights
        self.send_one_point(p2, angle, "prepare", "2 up", dz=dz_up)
        self.send_one_point(p2, angle, "normal", "2 up -", dz=dz_up)


    def process_one(self):
        self.get_logger().info("One cardboard")

        # ensure there is at least one not filled hole
        self.publish_working_area('hole_wait')
        self.read_select_hole('hole', filled=True, normalize_angle=True)
        ids = self.current_hole_ids('hole')
        print( ">>>", ids)
        for current_id in ids:
            self.process_one_hole(current_id)

        self.publish_working_area('hole_finished')

        # go back to the initial pose
        self.send_one_point(self.init_pose, 0.0, 'normal', 'out')

        # clean all detected holes; they should be visible now
        self.clean_holes()

        self.get_logger().info("Done")


def main(args=None):
    utils.init_spin_node(args, HolePlanner)
