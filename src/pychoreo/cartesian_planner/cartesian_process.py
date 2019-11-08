import random
import warnings

from pybullet_planning import multiply, set_pose, get_movable_joints
from pybullet_planning import get_collision_fn, get_floating_body_collision_fn

# EE domain (can be directions, or directly poses)
# EE gen fn

# tool poses gen fn

### extrusion
# additive process: no attachment
# end point 0 -> end point 1
# EE directions (theta, phi)
# two continuous domains (0, 2pi)
# linear interpolation -> path poses
# collision fn: no attachment, built objects in the sequence
# need additional approach pathes

### pick and place
# additive process: no attachment
# two processes
# pick approach - attach - retreat
# place approach - attach - retreat
# EE behavior controlled by two tri-tuples of EE poses
# two index lists with the same length
# slerp interpolation
# the four sub processes have four different attachment / collision objects settings


def _null_ee_pose_gen_fn(path_points):
    raise NotImplementedError('ee_pose_gen_fn not specified!')
    # yield path_poses

def _null_sample_ik_fn(pose):
    raise NotImplementedError('sample_ik_fn not specified!')
    # raise Warning('sample_ik_fn not specified!')
    # yield conf

def _null_collision_fn(conf):
    raise NotImplementedError('collision fn not specified!')
    # raise Warning('collision fn not specified!')
    # return False

class CartesianTrajectory(object):
    def __init__(self, process_name='',
        robot=None, ik_joint_names=[], path_points=[],
        ee_pose_gen_fn=_null_ee_pose_gen_fn, sample_ik_fn=_null_sample_ik_fn, collision_fn=_null_collision_fn,
        pointwise_collision_fns={}, element_identifier=None):

        self._process_name = process_name
        self._robot = robot
        self._ik_joint_names = ik_joint_names
        self._path_pts = path_points
        self._ee_pose_gen_fn = ee_pose_gen_fn
        self._sample_ik_fn = sample_ik_fn
        self._collision_fn = collision_fn
        self._pointwise_collision_fns = pointwise_collision_fns
        self._reverse = False
        self._element_id = element_identifier

    @property
    def robot(self):
        return self._robot

    @robot.setter
    def robot(self, robot_):
        self._robot = robot_

    @property
    def ik_joint_names(self):
        return self._ik_joint_names

    @ik_joint_names.setter
    def ik_joint_names(self, ik_joint_names_):
        self._ik_joint_names = ik_joint_names_

    @property
    def path_points(self):
        if not self.reverse:
            return self._path_pts
        else:
            return self._path_pts[::-1]

    @path_points.setter
    def path_points(self, path_points_):
        self._path_pts = path_points_

    @property
    def dof(self):
        return len(self._ik_joint_names)

    @property
    def process_name(self):
        return self._process_name

    @process_name.setter
    def process_name(self, process_name_):
        self._process_name = process_name_

    @property
    def element_identifier(self):
        return self._element_id

    @element_identifier.setter
    def element_identifier(self, e_id):
        self._element_id = e_id

    @property
    def ee_pose_gen_fn(self):
        return self._ee_pose_gen_fn

    @ee_pose_gen_fn.setter
    def ee_pose_gen_fn(self, ee_pose_gen_fn_):
        self._ee_pose_gen_fn = ee_pose_gen_fn_

    @property
    def sample_ik_fn(self):
        return self._sample_ik_fn

    @sample_ik_fn.setter
    def sample_ik_fn(self, sample_ik_fn_):
        self._sample_ik_fn = sample_ik_fn_

    @property
    def collision_fn(self):
        return self._collision_fn

    @collision_fn.setter
    def collision_fn(self, collision_fn_):
        self._collision_fn = collision_fn_

    @property
    def pointwise_collision_fns(self):
        return self._pointwise_collision_fns

    @pointwise_collision_fns.setter
    def pointwise_collision_fns(self, pointwise_collision_fns_):
        self._pointwise_collision_fns = pointwise_collision_fns_

    @property
    def reverse(self):
        return self._reverse

    @reverse.setter
    def reverse(self, reverse_):
        self._reverse = reverse_

    def sample_ee_poses(self, tool_from_root=None):
        ee_poses = next(self.ee_pose_gen_fn)
        if tool_from_root:
            ee_poses = [multiply(p, tool_from_root) for p in ee_poses]
        if not self.reverse:
            return ee_poses
        else:
            return ee_poses[::-1]

    def get_ik_sols(self, ee_poses, check_collision=True, get_all=True, pt_ids=[]):
        if get_all:
            pt_ids = range(len(ee_poses))
        full_jt_list = []
        for pt_id in pt_ids:
            jt_list = self.sample_ik_fn(ee_poses[pt_id])
            if check_collision:
                jt_list = [jts for jts in jt_list if jts and not self.collision_fn(jts)]
                if pt_id in self.pointwise_collision_fns:
                    jt_list = [jts for jts in jt_list if not self.pointwise_collision_fns[pt_id](jts)]
            full_jt_list.append(jt_list)
        return full_jt_list

    def __repr__(self):
        return 'cart process - {}'.format(self.process_name)

##################################################

def prune_ee_feasible_directions(cartesian_process, free_pose_map, ee_pose_map_fn, ee_body,
    tool_from_root=None, collision_objects=[], workspace_bodies=[], check_ik=False):
    # only take the positional part
    way_points = [p[0] for p in cartesian_process.sample_ee_poses()]

    from pybullet_planning import wait_for_user

    ee_collision_fn = get_floating_body_collision_fn(ee_body, collision_objects,
                                       ws_bodies=workspace_bodies,
                                       ws_disabled_body_link_pairs={})

    fmap_ids = list(range(len(free_pose_map)))
    random.shuffle(fmap_ids)
    for i in fmap_ids:
        if free_pose_map[i]:
            direction_pose = ee_pose_map_fn(i)
            way_poses = [(pt, direction_pose[1]) for pt in way_points]
            for pose in way_poses:
                # transform TCP to EE tool base link
                if tool_from_root:
                    pose = multiply(pose, tool_from_root)
                # check pairwise collision between the EE and collision objects
                is_colliding = ee_collision_fn(pose)
                if not is_colliding and check_ik:
                    raise NotImplementedError
                if is_colliding:
                    free_pose_map[i] = 0
                    break
                # wait_for_user()
    # print(free_pose_map)
    return free_pose_map
