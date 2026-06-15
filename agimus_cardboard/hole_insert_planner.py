import numpy as np
import rclpy
from agimus_controller_mod_ros import node_utils as utils
from std_srvs.srv import SetBool
from .hole_planner_base import HolePlannerBase, HoleSelected

class HoleInsertPlanner(HolePlannerBase):
    """"""

    def __init__(self):
        super().__init__("hole_insert_planner", ['hole', 'holder_part'])

        self.srv_gripper = utils.service_client(self, SetBool,
                                                "schunk_gripper/activate")
        self.area_color['hole_wait'] = (1.0, 0.0, 0.0)
        self.area_color['hole_process'] = (0.0, 1.0, 1.0)
        self.area_color['part_wait'] = (1.0, 0.5, 0.0)
        self.area_color['part_process'] = (0.0, 0.0, 1.0)
        self.area_color['finished'] = (0.5, 0.5, 0.5)

        # gripper test
        self.gripper(False)
        self.gripper(True)

    def gripper(self, state: bool):
        self.get_logger().debug(f'Setting gripper: {state}')

        request = SetBool.Request()
        request.data = state
        self.srv_gripper.wait_for_service()
        future = self.srv_gripper.call_async(request)
        rclpy.spin_until_future_complete(self, future)

    def process_one(self):

        dz_up = self.params.delta_z
        dz_dn_part = - 0.01  # TODO param - some pressure to correctly grab the part
        dz_dn_hole = - 0.01  # TODO param - additional pressure

        # take one part
        self.publish_working_area('part_wait')
        part_sel = self.read_select_hole('holder_part', filled=True)
        pu, angle = part_sel.u, part_sel.angle
        self.publish_working_area('part_process')

        self.get_logger().info(
            f"Processing part {part_sel.hole_id} ({int(angle / np.pi * 180)} deg)")

        # move above the part
        self.send_one_point(pu, angle, 'normal', "part up-", dz=dz_up)

        # cardboard is visible now, clean holes, so the actual data will be used next
        self.clean_holes()

        # close the gripper, increase weights, then move down and grab
        self.gripper(True)
        self.send_one_point(pu, angle, 'prepare', "part up+", dz=dz_up)
        self.send_one_point(pu, angle, 'hole', "part down", dz=dz_dn_part)
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

        # check if the part is not there (and clean parts for the next run)
        self.publish_working_area('part_wait')
        part_sel = self.read_select_hole('holder_part', hole_id=part_sel.hole_id)
        self.publish_working_area('hole_process')

        if part_sel.filled:
            self.get_logger().error(f"Grab failed, part: {part_sel.hole_id}")
        else:
            # increase weights, move down, and release
            self.send_one_point(p, angle_h, 'prepare', "up+", dz=dz_up)
            self.send_one_point(p, angle_h, 'prepare', "mid+", dz=dz_up / 2)
            self.send_one_point(p, angle_h, 'hole', "dn", dz=dz_dn_hole)
            self.gripper(True)

            # move up, decrease weights
            self.send_one_point(p, angle_h, 'hole', "up", dz=dz_up)
            self.send_one_point(p, angle_h, 'normal_weights', "up-", dz=dz_up)

        # half-way back
        self.send_one_point(p_half, angle_half, 'normal', "half", dz=dz_up)

        self.publish_working_area('finished')
        self.get_logger().info("Done")

def main(args=None):
    utils.init_spin_node(args, HoleInsertPlanner)
