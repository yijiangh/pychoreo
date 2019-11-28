import warnings
from pybullet_planning import is_connected, body_from_name, joints_from_names
from pybullet_planning import Attachment
from pychoreo.process_model.trajectory import Trajectory

class PicknPlaceBufferTrajectory(Trajectory):
    def __init__(self, robot, joints, traj_path, attachments=None, tag='approach', element_id=None, element_info=None):
        super(PicknPlaceBufferTrajectory, self).__init__(robot, joints, traj_path)
        self._element_id = element_id
        self._element_info = element_info
        self._attachments = attachments or []
        self.tag = tag

    @classmethod
    def from_trajectory(cls, traj, attachments=[], tag='approach', element_id=None, element_info=None):
        assert isinstance(traj, Trajectory)
        return cls(traj.robot, traj.joints, traj.traj_path, attachments, tag, element_id, element_info)

    @property
    def element_id(self):
        return self._element_id

    @property
    def element_info(self):
        return self._element_info

    @property
    def attachments(self):
        return self._attachments

    @attachments.setter
    def attachments(self, attachs_):
        self._attachments = attachs_

    def to_data(self, include_robot_data=False, include_link_path=False):
        data = super(PicknPlaceBufferTrajectory, self).to_data(include_robot_data, include_link_path)
        data['traj_type'] = self.__class__.__name__
        data['element_id'] = self.element_id
        data['element_info'] = self.element_info
        data['attachments'] = [at.to_data() for at in self.attachments]
        return data

    @classmethod
    def from_data(cls, data):
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
        attachments = [Attachment.from_data(at_data) for at_data in data['attachments']]
        return cls(robot, joints, traj_path, attachments, data['tag'], data['element_id'], data['element_info'])

    def __repr__(self):
        return '{}:{}-{}'.format(self.__class__.__name__, self.tag, self.element_info)
