import warnings
from pybullet_planning import is_connected, body_from_name, joints_from_names
from pybullet_planning import Attachment
from pychoreo.process_model.trajectory import Trajectory

class PicknPlaceBufferTrajectory(Trajectory):
    def __init__(self, robot, joints, traj_path, tag='approach', ee_attachments=None, attachments=None, element_id=None, element_info=None):
        super(PicknPlaceBufferTrajectory, self).__init__(robot, joints, traj_path, tag, ee_attachments, attachments)
        self._element_id = element_id
        self._element_info = element_info

    @property
    def element_id(self):
        return self._element_id

    @property
    def element_info(self):
        return self._element_info

    def to_data(self, include_robot_data=False, include_link_path=False):
        data = super(PicknPlaceBufferTrajectory, self).to_data(include_robot_data, include_link_path)
        data['traj_type'] = self.__class__.__name__
        data['element_id'] = self.element_id
        data['element_info'] = self.element_info
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
        ee_attachments = [Attachment.from_data(at_data) for at_data in data['ee_attachments']]
        attachments = [Attachment.from_data(at_data) for at_data in data['attachments']]
        return cls(robot, joints, traj_path, data['tag'], ee_attachments, attachments, data['element_id'], data['element_info'])

    def __repr__(self):
        return '{}:{}-{}'.format(self.__class__.__name__, self.tag, self.element_info)
