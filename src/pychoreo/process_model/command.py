
class Command(object):
    # container for multiple Trajectory
    def __init__(self, trajectories=[], colliding=set()):
        self.trajectories = list(trajectories)
        self.colliding = set(colliding)

    def get_trajectory(self, TrajectoryType):
        for traj in self.trajectories:
            if isinstance(traj, TrajectoryType):
                return traj
        return None

    @property
    def start_conf(self):
        return self.trajectories[0].path[0]

    @property
    def end_conf(self):
        return self.trajectories[-1].path[-1]

    def reverse(self):
        return self.__class__([traj.reverse() for traj in reversed(self.trajectories)],
                              colliding=self.colliding)

    def iterate(self):
        for trajectory in self.trajectories:
            for output in trajectory.iterate():
                yield output

    def __repr__(self):
        return 'command[{}]'.format(','.join(map(repr, self.trajectories)))
