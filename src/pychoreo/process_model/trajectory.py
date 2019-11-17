from pybullet_planning import BodySaver
from pybullet_planning import has_link, link_from_name, get_link_pose
from pybullet_planning import set_joint_positions

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
        assert has_link(self.robot, link_from_name)
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

    def __repr__(self):
        return 'Traj{}|len#{}'.format(self.tag, len(self.traj_path))

class MotionTrajectory(Trajectory):
    def __init__(self, robot, joints, traj_path, attachments=[]):
        super(MotionTrajectory, self).__init__(robot, joints, traj_path)
        self.attachments = attachments

    @classmethod
    def from_trajectory(cls, traj, attachments=[]):
        isinstance(traj, Trajectory)
        return cls(traj.robot, traj.joints, traj.traj_path, attachments)

    def reverse(self):
        return self.__class__(self.robot, self.joints, self.traj_path[::-1], self.attachments)

    def __repr__(self):
        return 'MotionTraj(#J {}, #pth {})'.format(len(self.joints), len(self.traj_path))
