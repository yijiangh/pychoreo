from __future__ import print_function

import json
import os
import random
import time
from copy import copy
from collections import defaultdict
from itertools import product
import numpy as np
from termcolor import cprint

##################################################

def is_ground(element, ground_nodes):
    return any(n in ground_nodes for n in element)


def get_node_neighbors(elements):
    node_neighbors = defaultdict(set)
    for e in elements:
        n1, n2 = e
        node_neighbors[n1].add(e)
        node_neighbors[n2].add(e)
    return node_neighbors


def get_element_neighbors(elements):
    node_neighbors = get_node_neighbors(elements)
    element_neighbors = defaultdict(set)
    for e in elements:
        n1, n2 = e
        element_neighbors[e].update(node_neighbors[n1])
        element_neighbors[e].update(node_neighbors[n2])
        element_neighbors[e].remove(e)
    return element_neighbors

##################################################

def extrusion_direction_routing(element_sequence, elements, node_points, grounded_node_ids, heuristic='max_valence'):
    reverse_flags = {e : False for e in elements}
    current_node_neighbors = defaultdict(set)
    for seq_id, e in enumerate(element_sequence):
        if e not in elements:
            e = e[::-1]
            element_sequence[seq_id] = e
        assert e in elements
        n1, n2 = e
        current_node_neighbors[n1].add(e)
        current_node_neighbors[n2].add(e)

        if len(current_node_neighbors[n2]) > 1 and len(current_node_neighbors[n1]) == 1:
            # n1 is newly created, n2 connected
            reverse_flags[e] = True
            cprint('Routing {}:{}, forced direction flipped.'.format(seq_id, e), 'yellow')
        else:
            # if grounded, always start with the grounded node
            if n1 in grounded_node_ids or n2 in grounded_node_ids:
                if n2 in grounded_node_ids:
                    reverse_flags[e] = True
            else:
                if heuristic == 'max_valence' and len(current_node_neighbors[n1]) < len(current_node_neighbors[n2]) \
                    and len(current_node_neighbors[n2]) > 1:
                    cprint('Routing {}:{}, direction flipped by valence heuristic'.format(seq_id, e), 'green')
                # prefer starting with the node with a larger valence
                    reverse_flags[e] = True
                elif heuristic == 'z' and node_points[n1][2] > node_points[n2][2] \
                    and len(current_node_neighbors[n2]) > 1:
                    cprint('Routing {}:{} direction flipped by z heuristic'.format(seq_id, e), 'green')
                    reverse_flags[e] = True
    return reverse_flags
