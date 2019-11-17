import time
import pytest
from copy import deepcopy, copy
from itertools import tee

from pychoreo.process_model.gen_fn import GenFn
from pychoreo.utils.stream_utils import get_random_direction_generator

def enum_gen(op_list):
    for val in op_list:
        yield val

@pytest.mark.gen
def test_enumerate_gen_fn():
    a = [1,2,3,4,5,6]
    enum_gen_fn = GenFn(enum_gen(a))
    print(next(enum_gen_fn.gen))
    print(next(enum_gen_fn.gen))

    print('reset')
    enum_gen_fn.reset()
    for i in enum_gen_fn.gen:
        print(i)

    print('reset 2-nd')
    enum_gen_fn.reset()
    for i in enum_gen_fn.gen:
        print(i)

    print('copy reset')
    copy_enum_gen = copy(enum_gen_fn)
    copy_enum_gen.reset()
    with pytest.raises(StopIteration):
        next(enum_gen_fn.gen)
    for i in copy_enum_gen.gen:
        print(i)
    with pytest.raises(StopIteration):
        next(copy_enum_gen.gen)

    print('forever gen')
    start_time = time.time()
    randir_gen = GenFn(get_random_direction_generator())
    while time.time() - start_time < 1:
        g = next(randir_gen.gen)

    _, copy_randir_gen = tee(randir_gen.gen)
    start_time = time.time()
    while time.time() - start_time < 1:
        gg = next(copy_randir_gen)
