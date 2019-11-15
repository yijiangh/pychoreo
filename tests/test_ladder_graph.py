import pytest

from pychoreo.cartesian_planner.ladder_graph import LadderGraph, EdgeBuilder
from pychoreo.cartesian_planner.ladder_graph import append_ladder_graph, concatenate_graph_vertically
from pychoreo.cartesian_planner.dag_search import DAGSearch
from pychoreo.cartesian_planner.postprocessing import divide_list_chunks

def test_ladder_graph_search():
    with pytest.raises(ValueError):
       LadderGraph(-1)

    dof = 6
    graph = LadderGraph(dof)

    with pytest.raises(ValueError):
        dag_search = DAGSearch(graph)

    # TODO: test correspondence between ladder graph rungs & solution rungs
