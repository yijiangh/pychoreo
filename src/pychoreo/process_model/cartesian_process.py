import random
import warnings
from copy import copy
from itertools import product, tee

from pybullet_planning import multiply, set_pose, get_movable_joints, joints_from_names
from pybullet_planning import get_collision_fn, get_floating_body_collision_fn

from pychoreo.process_model.gen_fn import CartesianPoseGenFn

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


def _NULL_EE_POSE_GEN_FN():
    raise NotImplementedError('ee_pose_gen_fn not specified!')
    yield None

_NULL_CARTESIAN_GEN_FN = CartesianPoseGenFn([], _NULL_EE_POSE_GEN_FN())

def _NULL_SAMPLE_IK_FN(pose):
    raise NotImplementedError('sample_ik_fn not specified!')
    # raise Warning('sample_ik_fn not specified!')
    # yield conf

def _NULL_COLLISION_FN(conf):
    raise NotImplementedError('collision fn not specified!')
    # raise Warning('collision fn not specified!')
    # return False

class CartesianSubProcess(object):
    def __init__(self, sub_process_name='',
                 collision_fn=_NULL_COLLISION_FN, pointwise_collision_fns={}):
        self._sub_process_name = sub_process_name
        self._collision_fn = collision_fn
        self._pointwise_collision_fns = pointwise_collision_fns
        self._path_point_size = -1
        self._traj = None

    @property
    def sub_process_name(self):
        return self._sub_process_name

    @sub_process_name.setter
    def sub_process_name(self, _sub_process_name):
        self._sub_process_name = _sub_process_name

    @property
    def path_point_size(self):
        return self._path_point_size

    @path_point_size.setter
    def path_point_size(self, _path_point_size):
        if _path_point_size <= 0:
            raise ValueError('Path point size (now: {}) must be bigger than 0!'.format(_path_point_size))
        self._path_point_size = _path_point_size

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
    def trajectory(self):
        return self._traj

    @trajectory.setter
    def trajectory(self, trajectory_):
        self._traj = trajectory_

    def __repr__(self):
        return '|sp::*{}*/#pts:{}|'.format(self.sub_process_name, self.path_point_size)

class CartesianProcess(object):
    def __init__(self, process_name='',
        robot=None, ik_joint_names=[], sub_process_list=[],
        ee_pose_gen_fn=_NULL_EE_POSE_GEN_FN, sample_ik_fn=_NULL_SAMPLE_IK_FN,
        element_identifier=None):

        self._process_name = process_name
        self._robot = robot
        self._ik_joint_names = ik_joint_names
        self._sub_process_list = sub_process_list
        self._ee_pose_gen_fn = ee_pose_gen_fn
        self._sample_ik_fn = sample_ik_fn
        self._element_id = element_identifier
        self._trajectory = None

        # TODO: add a trigger to "renew" the sample ee iterator in case we want to sample the poses again

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
    def ik_joints(self):
        return joints_from_names(self.robot, self.ik_joint_names)

    @property
    def sub_process_list(self):
        return self._sub_process_list

    @sub_process_list.setter
    def sub_process_list(self, sub_process_list_):
        self._sub_process_list = sub_process_list_

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
        """the identifier of the element involved with the Cartesian process, usually indices of element
        in a list or dict. For example, in the case of element extrusion, we use (start_node_id, end_node_id)
        as the element identifier, which can be directly fed into the dict of element bodies to get the collision
        bodies.

        """
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

    def reset_ee_pose_gen_fn(self):
        self.ee_pose_gen_fn.reset()

    def sample_ee_poses(self, tool_from_root=None, copy_iter=False):
        if not copy_iter:
            ee_poses = next(self.ee_pose_gen_fn.gen)
        else:
            _, ee_pose_gen_fn = tee(self.ee_pose_gen_fn.gen)
            try:
                ee_poses = next(ee_pose_gen_fn)
            except StopIteration:
                ee_pose_gen = copy(self.ee_pose_gen_fn)
                ee_pose_gen.reset()
                ee_poses = next(ee_pose_gen.gen)
        assert len(ee_poses) == len(self.sub_process_list), 'sampled ee poses size ({}) not equal to the number of sub_processes ({})!'.format(len(ee_poses), len(self.sub_process_list))
        for sp_poses, sp in zip(ee_poses, self.sub_process_list):
            sp.path_point_size = len(sp_poses)
        if tool_from_root:
            ee_poses = [[multiply(p, tool_from_root) for p in sub_p] for sub_p in ee_poses]
        return ee_poses

    def exhaust_iter(self, tool_from_root=None):
        self.ee_pose_gen_fn.reset()
        while True:
            try:
                yield self.sample_ee_poses(tool_from_root=tool_from_root)
            except StopIteration:
                break

    def get_ik_sols(self, ee_poses, check_collision=True, diagnosis=False):
        assert len(ee_poses) == len(self.sub_process_list), 'sampled ee poses size ({}) not equal to the number of sub_processes ({})!'.format(len(ee_poses), len(self.sub_process_list))
        sp_pt_ids = list(zip(range(len(self.sub_process_list)), [list(range(len(sp_poses))) for sp_poses in ee_poses]))
        full_jt_list = [[] for _ in range(len(sp_pt_ids))]
        for sp_id, pt_ids in sp_pt_ids:
            for pt_id in pt_ids:
                jt_list = self.sample_ik_fn(ee_poses[sp_id][pt_id])
                if check_collision:
                    jt_list = [jts for jts in jt_list if jts and not self.sub_process_list[sp_id].collision_fn(jts, diagnosis=diagnosis)]
                    if pt_id in self.sub_process_list[sp_id].pointwise_collision_fns:
                        jt_list = [jts for jts in jt_list if not self.sub_process_list[sp_id].pointwise_collision_fns[pt_id](jts)]
                full_jt_list[sp_id].append(jt_list)
        return full_jt_list

    def __repr__(self):
        return 'cart process-{}|E#{}|sp#{}'.format(self.process_name, self.element_identifier, len(self.sub_process_list))

##################################################

def prune_ee_feasible_directions(cartesian_process, free_pose_map, ee_pose_map_fn, ee_body,
                                 check_all=True, pt_ids=[],
                                 self_collisions=True, disabled_collisions={},
                                 obstacles=[], extra_disabled_collisions={},
                                 tool_from_root=None, check_ik=False):
    # only take the positional part
    sampled_poses = cartesian_process.sample_ee_poses(copy_iter=True)
    way_points = [[p[0] for p in sp_poses] for sp_poses in sampled_poses]
    ee_collision_fn = get_floating_body_collision_fn(ee_body, obstacles,
                                                     disabled_collisions=disabled_collisions)

    assert len(way_points) == len(cartesian_process.sub_process_list), 'sampled ee poses must have the same number of lists with the number of sub_processes!'
    if check_all:
        sp_pt_ids = list(zip(range(len(cartesian_process.sub_process_list)), [list(range(len(sp_poses))) for sp_poses in way_points]))

    fmap_ids = list(range(len(free_pose_map)))
    random.shuffle(fmap_ids)
    for i in fmap_ids:
        if free_pose_map[i]:
            direction_pose = ee_pose_map_fn(i)
            way_poses = [[(pt, direction_pose[1]) for pt in sp_way_points] for sp_way_points in way_points]
            for sp_id, pt_ids in sp_pt_ids:
                for pt_id in pt_ids:
                    pose = way_poses[sp_id][pt_id]
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
    return free_pose_map
