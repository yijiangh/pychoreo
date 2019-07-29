import numpy as np
import os
import sys
import time
import argparse

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

from .assembly_datastructure import AssemblyNetwork
from choreo.extrusion.extrusion_utils import load_extrusion
from choreo.choreo_utils import read_seq_json, EEDirection, make_print_pose, read_csp_log_json
from conrob_pybullet.ss_pybullet.pybullet_tools.utils import add_line, Euler, Pose, multiply, Point, tform_point
from pyconmech import stiffness_checker
from choreo.deformed_frame_viz import meshcat_visualize_deformed

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


def meshcat_visualize_assembly_sequence(meshcat_vis, assembly_network, element_id_sequence, seq_poses,
                                        stiffness_checker=None, viz_pose=False, viz_deform=False,
                                        scale=1.0, time_step=1, direction_len=0.01, exagg=1.0):
    # EE direction color
    dir_color = [0, 1, 0]

    disc = 10 # disc for deformed beam

    ref_pt, _ = assembly_network.get_end_points(0)
    existing_e_ids = []

    if stiffness_checker:
        t_tol, r_tol = stiffness_checker.get_nodal_deformation_tol()

    for k in element_id_sequence.keys():
        e_id = element_id_sequence[k]
        existing_e_ids.append(e_id)

        if not viz_deform and stiffness_checker:
            print(existing_e_ids)
            assert(stiffness_checker.solve(existing_e_ids))
            max_t, max_r = stiffness_checker.get_max_nodal_deformation()
            print("max_t: {0} / {1}, max_r: {2} / {3}".format(max_t, t_tol, max_r, r_tol))

            orig_shape = stiffness_checker.get_original_shape(disc=disc, draw_full_shape=False)
            beam_disp = stiffness_checker.get_deformed_shape(exagg_ratio=exagg, disc=disc)
            meshcat_visualize_deformed(meshcat_vis, beam_disp, orig_shape, disc, scale=scale)
        else:
            p1, p2 = assembly_network.get_end_points(e_id)
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

            if seq_poses is not None and viz_pose:
                assert(k in seq_poses)
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
                                   g.MeshBasicMaterial(color=rgb_to_hex(dir_color), opacity=0.1)))

                # meshcat_vis[mc_dir_key].set_object(g.Points(
                #     g.PointsGeometry(dir_vertices),
                #     g.PointsMaterial(size=0.01)))
        time.sleep(time_step)


def meshcat_visualize_csp_log(meshcat_vis, assembly_network, assign_history, stiffness_checker=None, scale=1.0, time_step=1):
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
    parser.add_argument('-cl', '--check_log', default='none',
                        help='e.g. backward_sp, forward_random. It will search for <problem>_backward_sp_csp_log.json')
    parser.add_argument('-vd', '--viz_deform', action='store_true', help='visualize deformation, default to false')
    parser.add_argument('-vp', '--viz_pose', action='store_true', help='visualize feasible ee poses, default to false')
    parser.add_argument('-s', '--scale', type=float, default=5.0, help='The vis scale')
    parser.add_argument('-dt', '--delta_t', type=float, default=0.5, help='Vis time step')
    parser.add_argument('-dl', '--dir_len', type=float, default=0.05, help='EE direction len')
    args = parser.parse_args()
    print('Arguments:', args)

    # Create a new visualizer
    vis = meshcat.Visualizer()
    try:
        vis.open()
    except:
        vis.url()

    try:
        elements, node_points, ground_nodes, shape_file_path = load_extrusion(args.problem, parse_layers=True)
        node_order = list(range(len(node_points)))

        print('file path: {0}'.format(shape_file_path))

        # sc = None
        # if args.viz_deform:
        sc = stiffness_checker(json_file_path=shape_file_path, verbose=False)
        sc.set_self_weight_load(True)

        # vert indices sanity check
        ground_nodes = [n for n in ground_nodes if n in node_order]
        elements = [element for element in elements if all(n in node_order for n in element.node_ids)]
        assembly_network = AssemblyNetwork(node_points, elements, ground_nodes)

        # TODO: safeguarding
        if args.check_log != 'none':
            assign_history = read_csp_log_json(args.problem, specs=args.check_log)
            meshcat_visualize_csp_log(vis, assembly_network, assign_history,
                                      scale=args.scale, time_step=args.delta_t,
                                      stiffness_checker=sc)

        else:
            element_seq, seq_poses = read_seq_json(args.problem)
            meshcat_visualize_assembly_sequence(vis, assembly_network, element_seq, seq_poses,
                                                stiffness_checker=sc, viz_pose=args.viz_pose, viz_deform=args.viz_deform,
                                                scale=args.scale, direction_len=args.dir_len, time_step=args.delta_t)
    except:
        vis.delete()

    # TODO: better exit to release meshcat port
    user_input('ctrl-c to exit...')

if __name__ == '__main__':
    main()
