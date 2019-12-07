import pytest
from fixtures.extrusion import extrusion_problem_path, extrusion_robot_data, extrusion_end_effector
from fixtures.picknplace import picknplace_problem_path, picknplace_robot_data, picknplace_end_effector, picknplace_tcp_def


def pytest_addoption(parser):
    parser.addoption('--viewer', action='store_true', help='Enables the pybullet viewer')
    # TODO: not working now...
    parser.addoption('--solve_method', default='sparse_ladder_graph', choices=['ladder_graph', 'sparse_ladder_graph'], \
        help='pychoreo Cartesian planner.')

@pytest.fixture
def viewer(request):
    return request.config.getoption("--viewer")

@pytest.fixture
def solve_method(request):
    return request.config.getoption("--solve_method")

# @pytest.fixture
# def solve_method(request):
#     return request.param

# def pytest_generate_tests(metafunc):
#     default_solve_methods = ['ladder_graph', 'sparse_ladder_graph']
#     solve_method_opt = metafunc.config.getoption('solve_method') or default_solve_methods
#     if 'solve_method' in metafunc.fixturenames:
#         metafunc.parametrize('solve_method', solve_method_opt, indirect=True)
