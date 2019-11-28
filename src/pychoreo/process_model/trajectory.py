import warnings
from pybullet_planning import BodySaver, is_connected
from pybullet_planning import tform_from_pose
from pybullet_planning import has_link, link_from_name, get_link_pose, get_body_name, get_link_name, body_from_name
from pybullet_planning import set_joint_positions, get_joint_names, joints_from_names

class Trajectory(object):
    def __init__(self, robot, joints, traj_path, tag=''):
        self.robot = robot
        self.joints = joints
        self.traj_path = traj_path
        self.path_from_link = {}
        self._tag = ''

    @property
    def tag(self):
        return self._tag

    @tag.setter
    def tag(self, tag_):
        self._tag = tag_

    def get_link_path(self, link_name):
        # This is essentially doing forward kinematics for the specified link
        assert has_link(self.robot, link_name)
        link = link_from_name(self.robot, link_name)
        if link not in self.path_from_link:
            with BodySaver(self.robot):
                self.path_from_link[link] = []
                for conf in self.traj_path:
                    set_joint_positions(self.robot, self.joints, conf)
                    self.path_from_link[link].append(get_link_pose(self.robot, link))
        return self.path_from_link[link]

    def reverse(self):
        raise NotImplementedError()

    def iterate(self):
        for conf in self.traj_path[1:]:
            set_joint_positions(self.robot, self.joints, conf)
            yield

    def to_data(self, include_robot_data=False, include_link_path=False):
        data = {}
        if include_robot_data:
            robot_name = get_body_name(self.robot)
            joints_name = get_joint_names(self.robot, self.joints)
            data['robot_name'] = robot_name
            data['joints_name'] = [jt_name.decode(encoding='UTF-8') for jt_name in joints_name]
        data['traj_type'] = self.__class__.__name__
        data['tag'] = self.tag
        data['traj_path'] = self.traj_path
        if include_link_path:
            data['link_path'] = {get_link_name(self.robot, link) : [tform_from_pose(p).tolist() for p in lpath] for link, lpath in self.path_from_link.items()}
        return data

    @classmethod
    def from_data(cls, data):
        if not is_connected():
            warnings.warn('Pybullet environment not connected or body/joints not found, robot and joints kept as names.')
            robot = data['robot_name']
            joints = data['joints_name']
        else:
            robot = body_from_name(data['robot_name'])
            joints = joints_from_names(robot, data['joints_name'])
        traj_path = data['traj_path']
        tag = data['tag']
        # TODO: parse path_link as well
        return cls(robot, joints, traj_path, tag)

    def __repr__(self):
        return 'Traj{}|len#{}'.format(self.tag, len(self.traj_path))

class MotionTrajectory(Trajectory):
    def __init__(self, robot, joints, traj_path, attachments=None):
        super(MotionTrajectory, self).__init__(robot, joints, traj_path)
        self._attachments = attachments or []

    @property
    def attachments(self):
        return self._attachments

    @classmethod
    def from_trajectory(cls, traj, attachments=[]):
        isinstance(traj, Trajectory)
        return cls(traj.robot, traj.joints, traj.traj_path, attachments)

    def reverse(self):
        return self.__class__(self.robot, self.joints, self.traj_path[::-1], self.attachments)

    def to_data(self, include_robot_data=False, include_link_path=False):
        data = super(MotionTrajectory, self).to_data(include_robot_data, include_link_path)
        data['traj_type'] = self.__class__.__name__
        return data

    # TODO: from_data, parse attachments

    def __repr__(self):
        return 'MotionTraj(#J {}, #pth {})'.format(len(self.joints), len(self.traj_path))
