import numpy as np
import os
import time
import argparse

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

from .assembly_datastructure import AssemblyNetwork
from choreo.extrusion_utils import load_extrusion
from .choreo_utils import read_seq_json, EEDirection, make_print_pose, read_csp_log_json

from conrob_pybullet.ss_pybullet.pybullet_tools.utils import add_line, Euler, Pose, multiply, Point, tform_point

try:
    user_input = raw_input
except NameError:
    user_input = input


# meshcat util fn
def rgb_to_hex(rgb):
    """Return color as '0xrrggbb' for the given color values."""
    red = hex(int(255*rgb[0])).lstrip('0x')
    green = hex(int(255*rgb[1])).lstrip('0x')
    blue = hex(int(255*rgb[2])).lstrip('0x')
    return '0x{0:0>2}{1:0>2}{2:0>2}'.format(red, green, blue)


def meshcat_visualize_assembly_sequence(meshcat_vis, assembly_network, element_id_sequence, seq_poses, \
                                        scale=1.0, time_step=1, direction_len=0.01):
    # EE direction color
    dir_color = [1, 0, 0]

    ref_pt = np.zeros(3)
    for k in element_id_sequence.keys():
        e_id = element_id_sequence[k]
        p1, p2 = assembly_network.get_end_points(e_id)

        if k == 0:
            ref_pt = p1

        p1 = ref_pt + (p1 - ref_pt) * scale
        p2 = ref_pt + (p2 - ref_pt) * scale

        e_mid = (np.array(p1) + np.array(p2)) / 2

        seq_ratio = float(k)/(len(element_id_sequence)-1)
        color = np.array([0, 0, 1])*(1-seq_ratio) + np.array([1, 0, 0])*seq_ratio
        # TODO: add_text(str(k), position=e_mid, text_size=text_size)
        # print('color {0} -> {1}'.format(color, rgb_to_hex(color)))

        vertices = np.vstack((p1, p2)).T

        mc_key = 'ase_seq_' + str(k)
        meshcat_vis[mc_key].set_object(g.LineSegments(g.PointsGeometry(vertices), g.MeshBasicMaterial(color=rgb_to_hex(color))))

        if seq_poses is not None:
            assert(seq_poses.has_key(k))
            dir_vertices = np.zeros([3, 2*len(seq_poses[k])])
            for i, ee_dir in enumerate(seq_poses[k]):
                assert(isinstance(ee_dir, EEDirection))
                cmap_pose = multiply(Pose(point=e_mid), make_print_pose(ee_dir.phi, ee_dir.theta))
                origin_world = e_mid
                axis = np.zeros(3)
                axis[2] = 1
                axis_world = tform_point(cmap_pose, direction_len*axis)

                dir_vertices[:, 2*i] = np.array(origin_world)
                dir_vertices[:, 2*i+1] = np.array(axis_world)

            mc_dir_key = 'as_dir_' + str(k)
            meshcat_vis[mc_dir_key + 'line'].set_object(
                g.LineSegments(g.PointsGeometry(dir_vertices),
                               g.MeshBasicMaterial(color=rgb_to_hex(dir_color))))

            # meshcat_vis[mc_dir_key].set_object(g.Points(
            #     g.PointsGeometry(dir_vertices),
            #     g.PointsMaterial(size=0.01)))

        time.sleep(time_step)


def meshcat_visualize_csp_log(meshcat_vis, assembly_network, assign_history, scale=1.0, time_step=1):
    # EE direction color
    color = [1, 0, 0]
    remain_color = [1, 1, 0]

    ref_pt = np.zeros(3)
    ref_pt, _ = assembly_network.get_end_points(0)
    full_e_ids = set(range(assembly_network.get_size_of_elements()))

    for k in assign_history.keys():
        e_ids = assign_history[k]
        vertices = np.zeros([3, 2*len(e_ids)])
        for i, e in enumerate(e_ids):
            p1, p2 = assembly_network.get_end_points(e)
            p1 = ref_pt + (p1 - ref_pt) * scale
            p2 = ref_pt + (p2 - ref_pt) * scale
            vertices[:, 2*i] = np.array(p1)
            vertices[:, 2*i+1] = np.array(p2)

        if int(k) > 0:
            meshcat_vis['assign_history_' + str(k-1)].delete()
            meshcat_vis['assign_history_' + str(k-1) + '_remain'].delete()

        mc_dir_key = 'assign_history_' + str(k)
        meshcat_vis[mc_dir_key].set_object(
            g.LineSegments(g.PointsGeometry(vertices),
                           g.MeshBasicMaterial(color=rgb_to_hex(color))))

        remain_e_ids = list(full_e_ids.difference(e_ids))
        vertices = np.zeros([3, 2*len(remain_e_ids)])
        for i, e in enumerate(remain_e_ids):
            p1, p2 = assembly_network.get_end_points(e)
            p1 = ref_pt + (p1 - ref_pt) * scale
            p2 = ref_pt + (p2 - ref_pt) * scale
            vertices[:, 2*i] = np.array(p1)
            vertices[:, 2*i+1] = np.array(p2)
            meshcat_vis[mc_dir_key+'_remain'].set_object(
                g.LineSegments(g.PointsGeometry(vertices),
                               g.MeshBasicMaterial(color=rgb_to_hex(remain_color), opacity=0.3)))

        time.sleep(time_step)


def main():
    parser = argparse.ArgumentParser()
    # four-frame | simple_frame | djmm_test_block | mars_bubble | sig_artopt-bunny | topopt-100 | topopt-205 | topopt-310 | voronoi
    parser.add_argument('-p', '--problem', default='simple_frame', help='The name of the problem to solve')
    parser.add_argument('-cl', '--check_log', default='none', help='Check csp log, b for backward log, f for forward log, default none')
    parser.add_argument('-s', '--scale', type=float, default=5.0, help='The vis scale')
    parser.add_argument('-dt', '--delta_t', type=float, default=0.5, help='Vis time step')
    args = parser.parse_args()
    print('Arguments:', args)

    # Create a new visualizer
    vis = meshcat.Visualizer()
    try:
        vis.open()
    except:
        vis.url()

    elements, node_points, ground_nodes = load_extrusion(args.problem, parse_layers=True)
    node_order = list(range(len(node_points)))

    # vert indices sanity check
    ground_nodes = [n for n in ground_nodes if n in node_order]
    elements = [element for element in elements if all(n in node_order for n in element.node_ids)]
    assembly_network = AssemblyNetwork(node_points, elements, ground_nodes)

    # TODO: safeguarding
    if args.check_log != 'none':
        check_back_log = True if args.check_log == 'b' else False
        assign_history = read_csp_log_json(args.problem, check_backward_search=check_back_log)
        meshcat_visualize_csp_log(vis, assembly_network, assign_history,
                                  scale=args.scale, time_step=args.delta_t)

    else:
        element_seq, seq_poses = read_seq_json(args.problem)
        meshcat_visualize_assembly_sequence(vis, assembly_network, element_seq, seq_poses,
                                            scale=args.scale, time_step=args.delta_t)

    user_input('ctrl-c to exit...')
    vis.delete()

if __name__ == '__main__':
    main()