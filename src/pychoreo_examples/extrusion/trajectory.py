from pychoreo.process_model.trajectory import Trajectory

class PrintTrajectory(Trajectory):
    def __init__(self, robot, joints, traj_path, element, is_reverse=False):
        super(PrintTrajectory, self).__init__(robot, joints, traj_path)
        # self.tool_path = tool_path
        #assert len(self.path) == len(self.tool_path)
        self.is_reverse = is_reverse
        self._element = element
        self.n1, self.n2 = reversed(element) if self.is_reverse else element

    @classmethod
    def from_trajectory(cls, traj, element, is_reverse=False):
        isinstance(traj, Trajectory)
        return cls(traj.robot, traj.joints, traj.traj_path, element, is_reverse)

    @property
    def element(self):
        return self._element

    @property
    def directed_element(self):
        return (self.n1, self.n2)

    # def get_link_path(self, link_name=None):
    #     if link_name == EE_LINK_NAME:
    #         return self.tool_path
    #     return super(PrintTrajectory, self).get_link_path(link_name)

    def reverse(self):
        return self.__class__(self.robot, self.joints, self.traj_path[::-1], self.element, not self.is_reverse)

    def __repr__(self):
        return 'PrintTraj: n{}->n{}'.format(self.n1, self.n2)

# TODO: model approach / retreat path here
