import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'choreo'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'conrob_pybullet'))

import pytest

import choreo
from choreo.extrusion.extrusion_utils import load_extrusion, create_elements
from choreo.assembly_datastructure import AssemblyNetwork
from conrob_pybullet.ss_pybullet.pybullet_tools.utils import connect, disconnect

class Test_extrusion_utils(object):
    """ Test class for extrusion utils fn. """
    def test_parse_extrusion_instance(self):
        problem_name = 'simple_frame'
        elements, node_points, ground_nodes, file_path = load_extrusion(problem_name, parse_layers=True)
        node_ids = list(range(len(node_points)))

        assert len(elements) == 19
        assert len(node_points) == 12
        assert len(ground_nodes) == 4
        for e in elements:
            assert all(n in node_ids for n in e.node_ids), 'element end node id not in vertices lists!'

    def test_create_extrusion_frame_assembly_network_wo_bodies(self):
        problem_name = 'simple_frame'
        elements, node_points, ground_nodes, file_path = load_extrusion(problem_name, parse_layers=True)

        asn = AssemblyNetwork(node_points, elements, ground_nodes)
        asn.compute_traversal_to_ground_dist()

        for e in asn.elements.values():
            if e.is_grounded:
                assert e.to_ground_dist == 0
            else:
                assert 1 <= e.to_ground_dist and e.to_ground_dist <= 2

        assert asn.get_size_of_joints() == 12
        assert asn.get_size_of_elements() == 19
        assert asn.get_size_of_grounded_elements() == 4

    def test_create_element_bodies(self):
        problem_name = 'simple_frame'
        elements, node_points, ground_nodes, file_path = load_extrusion(problem_name, parse_layers=True)
        connect(use_gui=False)
        # create collision bodies
        bodies = create_elements(node_points, [tuple(e.node_ids) for e in elements])

        assert len(bodies) == 19

        disconnect()
