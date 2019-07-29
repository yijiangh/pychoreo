import pytest
import numpy as np
import random
from numpy.testing import assert_equal, assert_almost_equal
from choreo.interlock.utils import computeFeasibleRegionFromBlockDir
from conrob_pybullet.ss_pybullet.pybullet_tools.utils import Euler, Pose, \
    multiply, tform_point
from scipy.optimize import linear_sum_assignment
from scipy.linalg import solve_triangular, norm

def assert_approx_equal_vectors(vec1, vec2, unitize=False, tol_digit=6, exact_eq=False):
    assert len(vec1) == len(vec2)
    assert tol_digit > 0
    for i in range(len(vec1)):
        e1 = vec1[i] / np.linalg.norm(vec1) if unitize else vec1[i]
        e2 = vec2[i] / np.linalg.norm(vec2) if unitize else vec2[i]
        assert_almost_equal(e1, e2, tol_digit)
        if exact_eq:
            assert_equal(e1, e2)


def is_approx_equal_vectors(vec1, vec2, unitize=False, tol_digit=6, exact_eq=False):
    assert len(vec1) == len(vec2)
    assert tol_digit > 0
    for i in range(len(vec1)):
        e1 = vec1[i] / np.linalg.norm(vec1) if unitize else vec1[i]
        e2 = vec2[i] / np.linalg.norm(vec2) if unitize else vec2[i]
        is_close = np.isclose(e1, e2, atol=10**(-1 * tol_digit))
        if exact_eq:
            is_close = assert_equal(e1, e2)
        if not is_close:
            return is_close
    return is_close


def vec_list_matching(vec_list1, vec_list2):
    """compute minimum distance matching between two lists of vectors

    Parameters
    ----------
    vec_list1 : a list of n-tuples
    vec_list2 : a list of n-tuples

    Returns
    -------
    list1_ids: a list of int
        index of the optimal matching in list1, list1_ids[i] is matched to
        list2_ids[i]

    list2_ids: a list of int

    """
    assert len(vec_list1) == len(vec_list2)
    num = len(vec_list1)
    costm = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            costm[i,j] = np.linalg.norm(np.array(vec_list1[i]) - np.array(vec_list2[j]))
    list1_ids, list2_ids = linear_sum_assignment(costm)
    return list1_ids, list2_ids


def compute_from_rot_block_dirs(bdirs, gt_fdirs, random_rot=False, verbose=True):
    """ apply random transformation on blocking directions
    and ground truth feasible directions """
    if random_rot:
        r1 = random.uniform(-np.pi, np.pi)
        r2 = random.uniform(-np.pi, np.pi)
        r3 = random.uniform(-np.pi, np.pi)
        tform = Pose(euler=Euler(roll=r1, pitch=r2, yaw=r3))
        r_bdirs = [tform_point(tform, bdir) for bdir in bdirs]
        r_gt_fdirs = [tform_point(tform, fdir) for fdir in gt_fdirs]
        res_f_rays = computeFeasibleRegionFromBlockDir(r_bdirs, verbose)
        return res_f_rays, r_gt_fdirs
    else:
        res_f_rays = computeFeasibleRegionFromBlockDir(bdirs, verbose)
        return res_f_rays, gt_fdirs


def check_matched_direction_lists_equal(v_list1, v_list2, tol_digit=6, exact_eq=False, unitize=True):
    """ match two lists first then check corresponding equality """
    l1_ids, l2_ids = vec_list_matching(v_list1, v_list2)
    for id1, id2 in zip(l1_ids, l2_ids):
        assert_approx_equal_vectors(v_list1[id1], v_list2[id2],
                                    unitize=unitize, tol_digit=6, exact_eq=False)


def list_matched_direction_lists_equal(v_list1, v_list2, tol_digit=6, exact_eq=False, unitize=True):
    """ match two lists first then check corresponding equality """
    l1_ids, l2_ids = vec_list_matching(v_list1, v_list2)
    eq_list = []
    for id1, id2 in zip(l1_ids, l2_ids):
        is_close = is_approx_equal_vectors(v_list1[id1], v_list2[id2],
                    unitize=unitize, tol_digit=6, exact_eq=False)
        eq_list.append(is_close)
    return l1_ids, l2_ids, eq_list

######################################################

@pytest.mark.parametrize("random_rot, tol_digit", [(False, 30), (True, 10)])
def test_MortiseTenon_feasible_region_from_block_dirs(random_rot, tol_digit):
    print('\nmortise-tenon test...')
    block_dirs = [[1, 0, 0], [-1, 0, 0],\
                [0, 1, 0], [0, -1, 0],\
                [0, 0, 1]]
    gt_feasible_dirs = [(0,0,-1)] # ground truth
    f_rays, gt_feasible_dirs = compute_from_rot_block_dirs(block_dirs, gt_feasible_dirs, random_rot=random_rot)
    check_matched_direction_lists_equal(f_rays, gt_feasible_dirs, tol_digit=tol_digit)
    print('mortise-tenon passed.\n=========')


@pytest.mark.parametrize("random_rot, tol_digit", [(False, 30), (True, 10)])
def test_TriangleWedge_feasible_region_from_block_dirs(random_rot, tol_digit):
    # triangle wedge
    print('\ntriangle wedge test...')
    block_dirs = [[1, 0, -1], [-1, 0, -1], \
                  [0, 1, 0], [0, -1, 0]]
    gt_feasible_dirs = [(1,0,1), (-1,0,1)] # ground truth
    f_rays, gt_feasible_dirs = compute_from_rot_block_dirs(block_dirs, gt_feasible_dirs, random_rot=random_rot)
    check_matched_direction_lists_equal(f_rays, gt_feasible_dirs, tol_digit=tol_digit)
    print('triangle wedge passed.\n=========')


@pytest.mark.parametrize("random_rot, tol_digit", [(False, 30), (True, 10)])
def test_HalfHalved_feasible_region_from_block_dirs(random_rot, tol_digit):
    print('\nhalf-halved test...')
    block_dirs = [[0, 1, 0], [0, -1, 0], [0, 0, 1]]
    gt_feasible_dirs = [(0,0,-1), (-1,0,0), (1,0,0)] # ground truth
    # block_dirs = [[0, -1, 1], [0, 1, -1], [0, 1, 1]]
    # gt_feasible_dirs = [(1,0,0), (-1,0,0), (0,-1,-1)] # ground truth

    f_rays, gt_feasible_dirs = compute_from_rot_block_dirs(block_dirs, gt_feasible_dirs, random_rot=random_rot)
    # print('rays: {}'.format(f_rays))
    # print('ground truth: {}'.format(gt_feasible_dirs))

    if not random_rot:
        check_matched_direction_lists_equal(f_rays, gt_feasible_dirs, tol_digit=tol_digit)
    else:
        m_id1, _, eq_list = list_matched_direction_lists_equal(f_rays, gt_feasible_dirs, tol_digit=tol_digit)
        neq_ids = [id for id, e in enumerate(eq_list) if not e]
        assert len(neq_ids) == 1
        skewed_ray = f_rays[m_id1[neq_ids[0]]]

        Q, R = np.linalg.qr(np.vstack(gt_feasible_dirs).T)
        proj = Q.T.dot(skewed_ray)
        res = skewed_ray - Q.dot(proj)
        assert np.linalg.matrix_rank(np.vstack(gt_feasible_dirs)) == 2
        assert np.linalg.matrix_rank(np.vstack(f_rays)) == 2
        assert_almost_equal(norm(res), 0.0, tol_digit)
    print('half-halved passed.\n=========')


@pytest.mark.parametrize("random_rot, tol_digit", [(False, 30), (True, 10)])
def test_Sandwich_feasible_region_from_block_dirs(random_rot, tol_digit):
    print('\nSandwich test...')
    block_dirs = [[0, 0, 1], [0, 0, -1]]
    gt_feasible_dirs = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0)] # ground truth

    f_rays, gt_feasible_dirs = compute_from_rot_block_dirs(block_dirs, gt_feasible_dirs, random_rot=random_rot)

    if not random_rot:
        check_matched_direction_lists_equal(f_rays, gt_feasible_dirs, tol_digit=tol_digit)
    else:
        m_id1, _, eq_list = list_matched_direction_lists_equal(f_rays, gt_feasible_dirs, tol_digit=tol_digit)
        neq_ids = [id for id, e in enumerate(eq_list) if not e]
        assert len(neq_ids) == 4

        Q, R = np.linalg.qr(np.vstack(gt_feasible_dirs).T)
        for i in range(len(neq_ids)):
            skewed_ray = f_rays[m_id1[neq_ids[i]]]
            proj = Q.T.dot(skewed_ray)
            res = skewed_ray - Q.dot(proj)
            assert np.linalg.matrix_rank(np.vstack(gt_feasible_dirs)) == 2
            assert np.linalg.matrix_rank(np.vstack(f_rays)) == 2
            assert_almost_equal(norm(res), 0.0, tol_digit)
    print('Sandwich passed.\n=========')


# dovetail

# victor's joint

#############################################
# visualization

def rgb_to_hex(rgb):
    """Return color as '0xrrggbb' for the given color values."""
    red = hex(int(255*rgb[0])).lstrip('0x')
    green = hex(int(255*rgb[1])).lstrip('0x')
    blue = hex(int(255*rgb[2])).lstrip('0x')
    return '0x{0:0>2}{1:0>2}{2:0>2}'.format(red, green, blue)

import argparse
from time import sleep
import meshcat
import meshcat.geometry as g

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--random', action='store_true')
    parser.add_argument('-tol', '--tol_digit', default=6)
    args = parser.parse_args()

    print('\nhalf-halved test...')
    # block_dirs = [[0, 1, 0], [0, -1, 0], [0, 0, 1]]
    # gt_feasible_dirs = [(0,0,-1), (-1,0,0), (1,0,0)] # ground truth

    block_dirs = [[0, -1, 1], [0, 1, -1], [0, 1, 1]]
    gt_feasible_dirs = [(1,0,0), (-1,0,0), (0,-1,-1)] # ground truth

    f_rays, gt_feasible_dirs = compute_from_rot_block_dirs(block_dirs, gt_feasible_dirs, random_rot=args.random)
    print('rays: {}'.format(f_rays))
    print('ground truth: {}'.format(gt_feasible_dirs))
    # check_matched_direction_lists_equal(f_rays, gt_feasible_dirs, tol_digit=args.tol_digit)
    print('half-halved passed.\n=========')

    vis = meshcat.Visualizer()
    try:
        vis.open()
    except:
        vis.url()

    red = np.array([1.0, 0, 0])
    blue = np.array([0, 0, 1.0])
    white = np.array([1.0, 1.0, 1.0])
    pink = np.array([255.0, 20.0, 147.0]) / 255
    black = [0, 0, 0]

    l1_ids, l2_ids = vec_list_matching(f_rays, gt_feasible_dirs)
    origin = np.zeros((1,3))
    for i in range(len(f_rays)):
        ray = f_rays[l1_ids[i]]
        gt_v = gt_feasible_dirs[l2_ids[i]]

        print('ray : {}'.format(ray))
        vis['ray' + str(i)].set_object(
            g.Line(g.PointsGeometry(np.vstack((origin, ray)).T),
                   g.MeshBasicMaterial(rgb_to_hex(pink))))
        # sleep(1)
        input()

        print('gt_v : {}'.format(gt_v))
        vis['gt_v' + str(i)].set_object(
            g.Line(g.PointsGeometry(np.vstack((origin, gt_v)).T),
                   g.MeshBasicMaterial(rgb_to_hex(white))))
        # sleep(1)
        input()

    # vis.close()

if __name__ == '__main__':
    main()
