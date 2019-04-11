import math
import numpy as np
from profilehooks import timecall, profile

def tuple_l2norm(p1, p2):
    return math.sqrt(sum([pow((float(p1_i) - float(p2_i)),2) for p1_i, p2_i in zip(p1, p2)]))

# a spatial assembly network for easier node/eleement query
class AssemblyJoint(object):
    def __init__(self, node_point, node_id, is_grounded=False):
        self.node_point = node_point
        self.node_id = node_id
        self.neighbor_e_ids = []
        self.is_grounded = is_grounded

    def add_neighbor_element(self, e_id):
        self.neighbor_e_ids.append(e_id)

class AssemblyElement(object):
    def __init__(self, node_ids, layer_id=0, e_id=-1, element_body=None):
        self.e_id = e_id
        self.node_ids = node_ids
        self.layer_id = layer_id
        self.element_body=element_body
        self.is_grounded = False
        self.to_ground_dist = np.inf

    def __repr__(self):
        return 'e{0}, {1}, {2}, {3}'.format(self.e_id, self.node_ids, self.layer_id, self.element_body)

class AssemblyNetwork(object):
    def __init__(self, node_points, elements, grounded_nodes):
        self.assembly_joints = {}
        self.assembly_elements = {}
        self.layer_element_ids = {}
        for node_id, node in enumerate(node_points):
            is_grounded = True if node_id in grounded_nodes else False
            self.insert_joint(node, is_grounded)
        for e in elements:
            self.insert_element(e)

    def get_size_of_joints(self):
        return len(self.assembly_joints)

    def get_size_of_elements(self):
        return len(self.assembly_elements)

    def get_size_of_grounded_elements(self):
        return len([e for e in self.assembly_elements.values() if e.is_grounded])

    def get_element_end_point_ids(self, e_id):
        return self.assembly_elements[e_id].node_ids

    def get_end_points(self, e_id):
        return [self.assembly_joints[v_id].node_point for v_id in self.get_element_end_point_ids(e_id)]

    def get_element_length(self, e_id):
        return tuple_l2norm(*self.get_end_points(e_id))

    def get_element_body(self, e_id):
        return self.assembly_elements[e_id].element_body

    def get_element_neighbor(self, element_id, end_node_id=None):
        end_ids = list(self.assembly_elements[element_id].node_ids) if end_node_id is None else [end_node_id]
        ngbh_ids = []
        for end_node_id in end_ids:
            ngbh_ids.extend([e_id for e_id in self.assembly_joints[end_node_id].neighbor_e_ids if e_id != element_id])
        return ngbh_ids

    def get_node_neighbor(self, node_id):
        return self.assembly_joints[node_id].neighbor_e_ids

    def get_layer_element_ids(self, layer_id):
        assert(self.layer_element_ids.has_key(layer_id))
        return self.layer_element_ids[layer_id]

    def get_layers(self):
        return self.layer_element_ids.keys()

    def is_element_grounded(self, e_id):
        return self.assembly_elements[e_id].is_grounded

    def get_element_to_ground_dist(self, e_id):
        return self.assembly_elements[e_id].to_ground_dist

    # insert fns
    def insert_joint(self, node_point, is_grounded=False):
        """insert assembly_joint
        :param: (a 3-list float) nodal position
        """
        v_n = self.get_size_of_joints()
        for a_jt in self.assembly_joints.values():
            # check duplicates
            assert tuple_l2norm(node_point, a_jt.node_point)>1e-4, 'duplicated point inserted!'
        self.assembly_joints[v_n] = AssemblyJoint(node_point, v_n, is_grounded)

    def insert_element(self, element):
        assert element.e_id not in self.assembly_elements.keys(), \
               'element already in the list!'
        assert all(node_id in self.assembly_joints.keys() for node_id in element.node_ids), \
               'node id not exist in the assembly_joints'
        # add node-neighbor
        for node_id in element.node_ids:
            self.assembly_joints[node_id].add_neighbor_element(element.e_id)
        # set groundedness
        element.is_grounded = True \
            if any(self.assembly_joints[node_id].is_grounded for node_id in element.node_ids) else False

        if self.layer_element_ids.has_key(element.layer_id):
            self.layer_element_ids[element.layer_id].append(element.e_id)
        else:
            self.layer_element_ids[element.layer_id] = [element.e_id]

        self.assembly_elements[element.e_id] = element

    def print_neighbors(self):
        for e in self.assembly_elements.values():
            print('grounded:{}'.format(e.is_grounded))
            print('neighbor e of e{0}:{1}'.format(e.e_id, self.get_element_neighbor(e.e_id)))
            for end_id in e.node_ids:
                print('neighbor e of end{0}/e{1}:{2}'.format(
                    end_id, e.e_id, self.get_element_neighbor(e.e_id, end_id)))

        for v in self.assembly_joints.values():
            print('neighbor e of v{0}: {1}'.format(v.node_id, self.get_node_neighbor(v.node_id)))

    def dijkstra(self, src_e_id, sub_graph=None):
        def min_distance(e_size, dist, visited_set):
            # return -1 if all the unvisited vertices' dist = inf (disconnected)
            min = np.inf
            min_index = -1
            for e_id in range(e_size):
                if dist[e_id] < min and not visited_set[e_id]:
                    min = dist[e_id]
                    min_index = e_id
            # assert(min_index > -1)
            return min_index

        if self.is_element_grounded(src_e_id):
            return 0
        e_size = self.get_size_of_elements()
        dist = [np.inf] * e_size
        dist[src_e_id] = 0
        visited_set = [False] * e_size

        for k in range(e_size):
            if sub_graph:
                if k not in sub_graph:
                    continue
            e_id = min_distance(e_size, dist, visited_set)
            if e_id == -1:
                # all unvisited ones are inf distance (not connected)
                break

            visited_set[e_id] = True

            nbhd_e_ids = set(self.get_element_neighbor(e_id))
            if sub_graph:
                nbhd_e_ids = nbhd_e_ids.intersection(sub_graph)

            for n_e_id in nbhd_e_ids:
                if not visited_set[n_e_id] and dist[n_e_id] > dist[e_id] + 1:
                    dist[n_e_id] = dist[e_id] + 1

        # get smallest dist to the grounded elements
        grounded_dist = [dist[e.e_id] for e in self.assembly_elements.values() if e.is_grounded]
        return min(grounded_dist)

    @timecall
    def compute_traversal_to_ground_dist(self, sub_graph=None):
        considered_e_ids = self.assembly_elements.keys() if not sub_graph else sub_graph
        for e in considered_e_ids:
            self.assembly_elements[e].to_ground_dist = self.dijkstra(e, sub_graph)

    def __repr__(self):
        return 'assembly_net: #joint:{0}, #element:{1}'.format(self.get_size_of_joints(), self.get_size_of_elements())