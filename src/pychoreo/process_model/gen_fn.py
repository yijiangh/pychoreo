from itertools import tee
from pybullet_planning import INF

class GenFn(object):
    def __init__(self, gen_fn):
        self._archived_gen_fn, self._current_gen_fn = tee(gen_fn)

    def reset(self):
        self._archived_gen_fn, self._current_gen_fn = tee(self._archived_gen_fn)

    def update_gen_fn(self, new_gen_fn):
        self._archived_gen_fn, self._current_gen_fn = tee(new_gen_fn)

    @property
    def gen(self):
        return self._current_gen_fn

class CartesianPoseGenFn(GenFn):
    def __init__(self, base_path_pts, pose_gen_fn):
        super(CartesianPoseGenFn, self).__init__(pose_gen_fn)
        self._base_path_pts = base_path_pts

    @property
    def base_path_pts(self):
        return self._base_path_pts
