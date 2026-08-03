import argparse
import numpy as np
import rclpy
from agimus_controller_mod_ros import node_utils as utils
from std_srvs.srv import SetBool
from .hole_planner_base import HolePlannerBase

from agimus_controller_mod.trajectories.shake_insert import ShakeInsert
from agimus_controller_mod_msgs.action import TrajectoryAction

import \
    agimus_controller_mod_ros.trajectory_builders.shake_insert as shake_builder


class HoleInsertPlanner(HolePlannerBase):
    """"""

    def __init__(self, robot_name: str, mode: str = 'full'):
        # only_holes = True - no parts are grabbed, only navigate to holes
        # to check for accuracy

        super().__init__("hole_insert_planner", robot_name,
                         ['hole', 'holder_part'])

        self.mode = mode
        self.srv_gripper = utils.service_client(
            self, SetBool, f"{robot_name}/schunk_gripper/activate")
        self.area_color['hole_wait'] = (1.0, 0.0, 0.0)
        self.area_color['hole_process'] = (0.0, 1.0, 1.0)
        self.area_color['part_wait'] = (1.0, 0.5, 0.0)
        self.area_color['part_process'] = (0.0, 0.0, 1.0)
        self.area_color['finished'] = (0.5, 0.5, 0.5)

        gpar = self.goal_param['hole']

        # gripper test
        self.gripper(False)
        self.gripper(True)

        if self.mode == 'holes':
            self.gripper(False)

        self.get_logger().info(f"Mode: {self.mode}")
        self.seg_shake = ShakeInsert(self.ee_frame_name)
        self.seg_shake.weights = self.weights
        self.seg_shake.weights.w_end_effector_poses[self.ee_frame_name] = \
            gpar['w_pose']

    def gripper(self, state: bool):
        self.get_logger().debug(f'Setting gripper: {state}')

        request = SetBool.Request()
        request.data = state
        self.srv_gripper.wait_for_service()
        future = self.srv_gripper.call_async(request)
        rclpy.spin_until_future_complete(self, future)

    def one_shake(self, p, angle: float, dz: float, amount: float,
                  duration: float):
        goal = TrajectoryAction.Goal()
        g = goal.goal

        self.seg_shake.shake_amount = amount
        self.seg_shake.x_from = p
        self.seg_shake.x_to = p
        self.seg_shake.delta_z = dz
        self.seg_shake.duration = duration

        shake_builder.ShakeInsert().to_goal(self.seg_shake, g)

        g.rot_rpy = [0.0, 3.1415, angle]  # TODO move into to_goal
        return goal

    def process_one(self):

        dz_up = self.params.delta_z

        # take one part
        self.publish_working_area('part_wait')
        if self.mode == 'holes':
            part_sel = self.read_select_hole('holder_part')
        else:
            part_sel = self.read_select_hole('holder_part', filled=True)

        pu, angle = part_sel.u, part_sel.angle
        self.publish_working_area('part_process')

        self.get_logger().info(
            f"Processing part {part_sel.hole_id} ({int(angle / np.pi * 180)} deg)")

        # move above the part
        self.send_one_point(pu, angle, 'normal', "part up-", dz=dz_up)

        # cardboard is visible now, clean holes, so the actual data will be used next
        self.clean_holes()

        # close the gripper, increase weights, then move down, shake, and grab
        if not self.mode == 'holes':
            self.gripper(True)
        self.send_one_point(pu, angle, 'prepare', "part up+", dz=dz_up)
        self.send_one_point(pu, angle, 'hole', "part down",
                            dz=-self.params.dz_part)

        if self.params.shake_part != 0.0:
            g = self.one_shake(pu, angle, dz=-self.params.dz_part,
                               amount=self.params.shake_part,
                               duration=self.params.shake_duration_part)
            self.send_point(g, "dn_shake")

        if not self.mode == 'holes':
            self.gripper(False)

        # move up then decrease weights
        self.send_one_point(pu, angle, 'hole', "part up+", dz=dz_up)
        self.send_one_point(pu, angle, 'normal_weights', "part up", dz=dz_up)

        # take one hole
        self.publish_working_area('hole_wait')
        hole_sel = self.read_select_hole('hole', filled=False)
        self.publish_working_area('hole_process')

        p, angle_h = hole_sel.u, hole_sel.angle
        self.get_logger().info(f"Processing hole {angle_h}")

        # point in between; when we have angles with different signs, always
        # go through zero and not 2*pi
        p_half = (p + pu) / 2
        angle_half = (angle_h + angle) / 2
        self.send_one_point(p_half, angle_half, 'normal', 'half', dz=dz_up)

        # parts are visible, refresh for a check
        self.clean_holes('holder_part')

        # move above the hole beginning
        self.send_one_point(p, angle_h, 'normal', "up-", dz=dz_up)

        if self.mode == 'holes':
            part_sel.filled = False
        else:
            # check if the part is not there (and clean parts for the next run)
            self.publish_working_area('part_wait')
            part_sel = self.read_select_hole('holder_part',
                                             hole_id=part_sel.hole_id)

        self.publish_working_area('hole_process')

        if part_sel.filled:
            self.get_logger().error(f"Grab failed, part: {part_sel.hole_id}")
        else:
            if self.mode == 'parts':
                # just throw it
                self.gripper(True)
            else:
                # increase weights, move down, shake a bit
                self.send_one_point(p, angle_h, 'prepare', "up+", dz=dz_up)
                self.send_one_point(p, angle_h, 'prepare', "mid+", dz=dz_up / 2)
                self.send_one_point(p, angle_h, 'hole', "dn",
                                    dz=-self.params.dz_hole)

                if self.params.shake_hole != 0.0:
                    g = self.one_shake(p, angle_h, dz=-self.params.dz_hole,
                                       amount=self.params.shake_hole,
                                       duration=self.params.shake_duration_hole)
                    self.send_point(g, "dn_shake")

                if not self.mode == 'holes':
                    # release
                    self.gripper(True)

                # move up, decrease weights
                self.send_one_point(p, angle_h, 'hole', "up", dz=dz_up)
                self.send_one_point(p, angle_h, 'normal_weights', "up-",
                                    dz=dz_up)

        # half-way back
        self.send_one_point(p_half, angle_half, 'normal', "half", dz=dz_up)

        self.publish_working_area('finished')
        self.get_logger().info("Done")


def main(args=None):
    parser = argparse.ArgumentParser("detector")
    parser.add_argument("--mode", type=str)
    parser.add_argument("--robot_name", type=str)

    utils.init_spin_node(args, HoleInsertPlanner, parser)
