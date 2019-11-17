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

PRINT_BUFFER_TAGS = ['approach', 'retreat']

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
    def tag(self):
        return self._tag

    @tag.setter
    def tag(self, tag_):
        if tag_ not in PRINT_BUFFER_TAGS:
            raise ValueError('Unknown tag for PrintBufferTrajectory, must be one of : {}'.format(PRINT_BUFFER_TAGS))
        else:
            self._tag = tag_

    @property
    def related_node(self):
        if self.tag == 'approach':
            return self.n1
        elif self.tag == 'retreat':
            return self.n2

    # def get_link_path(self, link_name=None):
    #     if link_name == EE_LINK_NAME:
    #         return self.tool_path
    #     return super(PrintTrajectory, self).get_link_path(link_name)

    def __repr__(self):
        return '{}_traj: ->n{}, #E{}'.format(self.tag, self.related_node, self.element)
