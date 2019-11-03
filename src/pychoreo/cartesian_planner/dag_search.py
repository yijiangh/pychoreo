
class SolutionRung(object):
    def __init__(self):
        self.distance = []
        self.predecessor = []

    def extract_min(self):
        min_dist = min(self.distance)
        min_id = self.distance.index(min_dist)
        return min_dist, min_id

    def __len__(self):
        assert(len(self.distance) == len(self.predecessor))
        return len(self.distance)
    #
    # def __repr__(self):
    #     return 'min dist: {0}, min pred id: {1}'.format(self.extract_min())

class DAGSearch(object):
    def __init__(self, graph):
        assert(isinstance(graph, LadderGraph))
        self.graph = graph
        self.solution = [SolutionRung() for i in range(graph.get_rungs_size())]

        # allocate everything we need
        for i in range(graph.get_rungs_size()):
            n_verts = graph.get_rung_vert_size(i)
            assert(n_verts > 0)
            self.solution[i].distance = [0] * n_verts
            self.solution[i].predecessor = [0] * n_verts

    def distance(self, r_id, v_id):
        return self.solution[r_id].distance[v_id]

    def predecessor(self, r_id, v_id):
        return self.solution[r_id].predecessor[v_id]

    def run(self):
        """forward cost propagation"""
        # TODO: add st_conf cost to SolutionRung 0
        # first rung init to 0
        self.solution[0].distance = [0] * len(self.solution[0])

        # other rungs init to inf
        for j in range(1, len(self.solution)):
            self.solution[j].distance = [np.inf] * len(self.solution[j])

        for r_id in range(0, len(self.solution)-1):
            n_verts = self.graph.get_rung_vert_size(r_id)
            next_r_id = r_id + 1
            # for each vert in the out edge list
            for v_id in range(n_verts):
                u_cost = self.distance(r_id, v_id)
                edges = self.graph.get_edges(r_id)[v_id]
                for edge in edges:
                    dv = u_cost + edge.cost
                    if dv < self.distance(next_r_id, edge.idx):
                        self.solution[next_r_id].distance[edge.idx] = dv
                        self.solution[next_r_id].predecessor[edge.idx] = v_id

        return min(self.solution[-1].distance)

    def shortest_path(self):
        _, min_val_id = self.solution[-1].extract_min()
        path_idx = [0] * len(self.solution)

        current_v_id = min_val_id
        count = len(path_idx) - 1
        while count >= 0:
            path_idx[count] = current_v_id
            current_v_id = self.predecessor(count, current_v_id)
            count -= 1

        sol = []
        for r_id, v_id in enumerate(path_idx):
            data = self.graph.get_vert_data(r_id, v_id)
            sol.append(data)

        return sol
