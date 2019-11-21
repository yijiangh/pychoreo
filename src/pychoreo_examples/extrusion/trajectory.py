import warnings
from pybullet_planning import is_connected, body_from_name, joints_from_names
from pychoreo.process_model.trajectory import Trajectory

class PrintTrajectory(Trajectory):
    def __init__(self, robot, joints, traj_path, element, is_reverse=False, tag=''):
        super(PrintTrajectory, self).__init__(robot, joints, traj_path, tag=tag)
        self.is_reverse = is_reverse
        self._element = element
        self.n1, self.n2 = reversed(element) if self.is_reverse else element
        self.tag = tag

    @classmethod
    def from_trajectory(cls, traj, element, is_reverse=False, tag=''):
        isinstance(traj, Trajectory)
        return cls(traj.robot, traj.joints, traj.traj_path, element, is_reverse, tag)

    @property
    def element(self):
        return self._element

    @property
    def directed_element(self):
        return (self.n1, self.n2)

    def reverse(self):
        return self.__class__(self.robot, self.joints, self.traj_path[::-1], self.element, not self.is_reverse)

    def to_data(self, include_robot_data=False, include_link_path=False):
        data = super(PrintTrajectory, self).to_data(include_robot_data, include_link_path)
        data['traj_type'] = self.__class__.__name__
        data['element'] = self.element
        data['is_reverse'] = self.is_reverse
        return data

    @classmethod
    def from_data(cls, data):
        # TODO: trying to use the following but fail...
        # traj = super(PrintBufferTrajectory, cls).from_data(data)
        try:
            if not is_connected():
                raise ValueError('pybullet not connected!')
            robot = body_from_name(data['robot_name'])
            joints = joints_from_names(robot, data['joints_name'])
        except ValueError:
            warnings.warn('Pybullet environment not connected or body/joints not found, robot and joints kept as names.')
            robot = data['robot_name']
            joints = data['joints_name']
        traj_path = data['traj_path']
        return cls(robot, joints, traj_path, data['element'], data['is_reverse'], data['tag'])

    def __repr__(self):
        return 'PrintTraj: n{}->n{}'.format(self.n1, self.n2)

class PrintBufferTrajectory(Trajectory):
    def __init__(self, robot, joints, traj_path, element, is_reverse=False, tag='approach'):
        super(PrintBufferTrajectory, self).__init__(robot, joints, traj_path)
        self._element = element
        self.is_reverse = is_reverse
        self.n1, self.n2 = reversed(element) if self.is_reverse else element
        self.tag = tag

    @classmethod
    def from_trajectory(cls, traj, element, is_reverse=False, tag='approach'):
        assert isinstance(traj, Trajectory)
        return cls(traj.robot, traj.joints, traj.traj_path, element, is_reverse, tag)

    @property
    def element(self):
        return self._element

    @property
    def related_node(self):
        if self.tag == 'approach':
            return self.n1
        elif self.tag == 'retreat':
            return self.n2

    def to_data(self, include_robot_data=False, include_link_path=False):
        data = super(PrintBufferTrajectory, self).to_data(include_robot_data, include_link_path)
        data['traj_type'] = self.__class__.__name__
        data['element'] = self.element
        data['is_reverse'] = self.is_reverse
        return data

    @classmethod
    def from_data(cls, data):
        # TODO: trying to use the following but fail...
        # traj = super(PrintBufferTrajectory, cls).from_data(data)
        try:
            if not is_connected():
                raise ValueError('pybullet not connected!')
            robot = body_from_name(data['robot_name'])
            joints = joints_from_names(robot, data['joints_name'])
        except ValueError:
            warnings.warn('Pybullet environment not connected or body/joints not found, robot and joints kept as names.')
            robot = data['robot_name']
            joints = data['joints_name']
        traj_path = data['traj_path']
        return cls(robot, joints, traj_path, data['element'], data['is_reverse'], data['tag'])

    def __repr__(self):
        return '{}_traj: ->n{}, #E{}'.format(self.tag, self.related_node, self.element)
