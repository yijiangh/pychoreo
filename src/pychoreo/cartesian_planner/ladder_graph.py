from pybullet_planning import INF

class LadderGraphEdge(object):
    def __init__(self, idx=None, cost=-INF):
        self.idx = idx # the id of the destination vert
        self.cost = cost
        # TODO: we ignore the timing constraint here

    def __repr__(self):
        return 'E idx{0}, cost{1}'.format(self.idx, self.cost)


class LadderGraphRung(object):
    def __init__(self, id=None, data=[], edges=[], collision_fn=None):
        self.id = id
        self.data = data # joint_data: joint values are stored in one contiguous list
        self.edges = edges

        #TODO: collision bodies or collision fn
        self.collision_fn = collision_fn

    def __repr__(self):
        return 'id {0}, data {1}, edge num {2}'.format(self.id, len(self.data), len(self.edges))


class LadderGraph(object):
    def __init__(self, dof):
        assert(dof != 0)
        self.dof = dof
        self.rungs = []

    def get_dof(self):
        return self.dof

    def get_rung(self, rung_id):
        assert(rung_id < len(self.rungs))
        return self.rungs[rung_id]

    def get_edges(self, rung_id):
        return self.get_rung(rung_id).edges

    def get_edge_sizes(self):
        return [len(r.edges) for r in self.rungs]

    def get_data(self, rung_id):
        return self.get_rung(rung_id).data

    def get_rungs_size(self):
        return len(self.rungs)

    @property
    def size(self):
        return self.get_rungs_size()

    def get_rung_vert_size(self, rung_id):
        """count the number of vertices in a rung"""
        return int(len(self.get_rung(rung_id).data) / self.dof)

    def get_vert_size(self):
        """count the number of vertices in the whole graph"""
        return sum([self.get_rung_vert_size(r_id) for r_id in range(self.get_rungs_size())])

    def get_vert_sizes(self):
        return [self.get_rung_vert_size(r_id) for r_id in range(self.get_rungs_size())]

    def get_vert_data(self, rung_id, vert_id):
        return self.get_rung(rung_id).data[self.dof * vert_id : self.dof * (vert_id+1)]

    def resize(self, rung_number):
        if self.size == 0:
            self.rungs = [LadderGraphRung(id=None, data=[], edges=[]) for i in range(rung_number)]
            return
        if self.size > 0 and self.size < rung_number:
            # fill in the missing ones with empty rungs
            self.rungs.extend([LadderGraphRung(id=None, data=[], edges=[]) for i in range(rung_number - self.size)])
            return
        elif self.size > rung_number:
            self.rungs = [r for i, r in enumerate(self.rungs) if i < rung_number]
            return

    def clear(self):
        self.rungs = []

    # assign fns
    def assign_rung(self, r_id, sol_lists):
        rung = self.get_rung(r_id)
        rung.id = r_id
        rung.data = [jt for jt_l in sol_lists for jt in jt_l]
        assert(len(rung.data) % self.dof == 0)

    def assign_edges(self, r_id, edges):
        # edges_ref = self.get_edges(r_id)
        self.get_rung(r_id).edges = edges

    def __repr__(self):
        return 'g tot_r_size:{0}, v_sizes:{1}, e_sizes:{2}'.format(self.size, self.get_vert_sizes(), self.get_edge_sizes())

    # insert_rung, clear_rung_edges (maybe not needed at all)

class EdgeBuilder(object):
    """edge builder for ladder graph, construct edges for fully connected biparte graph"""
    def __init__(self, n_start, n_end, dof, upper_tm=None, joint_vel_limits=None):
        self.result_edges_ = [[] for i in range(n_start)]
        self.edge_scratch_ = [LadderGraphEdge(idx=None, cost=None) for i in range(n_end)] # preallocated space to work on
        self.dof_ = dof
        self.delta_buffer_ = [0 for i in range(dof)]
        self.count_ = 0
        self.has_edges_ = False

    def consider(self, st_jt, end_jt, index):
        """index: to_id"""
        # TODO check delta joint val exceeds the joint_vel_limits
        cost = 0
        for i in range(self.dof_):
            cost += abs(st_jt[i] - end_jt[i])

        # cost = sum(self.delta_buffer_)
        assert(self.count_ < len(self.edge_scratch_))
        self.edge_scratch_[self.count_].cost = cost
        self.edge_scratch_[self.count_].idx = index
        self.count_ += 1

    def next(self, i):
        #TODO: want to do std::move here to transfer memory...
        self.result_edges_[i] = deepcopy(self.edge_scratch_)
        self.has_edges_ = self.has_edges_ or self.count_ > 0
        self.count_ = 0

    @property
    def result(self):
        return self.result_edges_

    @property
    def has_edges(self):
        return self.has_edges_

