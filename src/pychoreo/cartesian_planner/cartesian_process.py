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

class CartesianTrajectory(object):
    def __init__(self, process_name,
        ee_pose_gen_fn, ee_pose_interp_fn, sample_ik_fn,
        collision_fn, pointwise_collision_fns={}):
        self._process_name = process_name
        self._ee_pose_gen_fn = ee_pose_gen_fn
        self._ee_pose_interp_fn = ee_pose_interp_fn
        self._sample_ik_fn = sample_ik_fn
        self._collision_fn = collision_fn
        self._pointwise_collision_fns = pointwise_collision_fns

    @property
    def process_name(self):
        return self._process_name

    @property
    def ee_pose_gen_fn(self):
        return self._ee_pose_gen_fn

    @property
    def ee_pose_interp_fn(self):
        return self._ee_pose_interp_fn

    @property
    def sample_ik_fn(self):
        return self._sample_ik_fn

    @property
    def collision_fn(self):
        return self._collision_fn

    @property
    def pointwise_collision_fns(self):
        return self._pointwise_collision_fns

    def gen_ee_poses(self):
        ee_landmarks = next(self.ee_pose_gen_fn)
        assert len(ee_landmarks) > 1, 'ee generation function must return at least two ee poses for interpolation.'
        full_path = []
        for i in range(len(ee_landmarks)-1):
            full_path.extend(self.ee_pose_interp_fn(ee_landmarks[i], ee_landmarks[i+1]))
        return ee_landmarks

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
