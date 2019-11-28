import pytest
from fixtures.extrusion import extrusion_problem_path, extrusion_robot_data, extrusion_end_effector
from fixtures.picknplace import picknplace_problem_path, picknplace_robot_data, picknplace_end_effector, picknplace_tcp_def


def pytest_addoption(parser):
    parser.addoption('--viewer', action='store_true', help='Enables the pybullet viewer')

@pytest.fixture
def viewer(request):
    return request.config.getoption("--viewer")
