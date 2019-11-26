from itertools import tee
from pybullet_planning import INF

class GenFn(object):
    def __init__(self, gen_fn):
        self._archived_gen_fn, self._current_gen_fn = tee(gen_fn)

    def reset(self):
        self._archived_gen_fn, self._current_gen_fn = tee(self._archived_gen_fn)

    @property
    def gen(self):
        return self._current_gen_fn

####################################
# Cartesian pose generator

def get_pose_gen_fn(sample_gen_fn, compose_pose_fn, **kwargs):
    for gen_sample in sample_gen_fn:
        yield compose_pose_fn(gen_sample, **kwargs)

class CartesianPoseGenFn(GenFn):
    def __init__(self, sample_gen_fn, compose_pose_fn, **kwargs):
        # static data used by the sample or compose fn that is not sampling-dependent
        self._static_data = {}
        self._static_data.update(kwargs)
        super(CartesianPoseGenFn, self).__init__(get_pose_gen_fn(sample_gen_fn, compose_pose_fn, **kwargs))

    @property
    def static_data(self):
        return self._static_data
