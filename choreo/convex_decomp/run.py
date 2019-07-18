from __future__ import print_function
import sys, os
import argparse
import json
from random import random

from conrob_pybullet.ss_pybullet.pybullet_tools.utils import connect, disconnect, wait_for_user, LockRenderer, \
    has_gui, remove_body, set_camera_pose, get_movable_joints, set_joint_positions, \
    wait_for_duration, point_from_pose, get_link_pose, link_from_name, add_line, user_input,\
    HideOutput, load_pybullet, create_obj, draw_pose, add_body_name, get_pose, \
    pose_from_tform, invert, multiply, set_pose, plan_joint_motion, get_joint_positions, \
    add_fixed_constraint, remove_fixed_constraint, Attachment, create_attachment, \
    pairwise_collision, set_color
from choreo.choreo_utils import parse_transform

try:
    from py_vhacd import compute_convex_decomp
except ImportError as e:
    print('\x1b[6;30;43m' + '{}'.format(e) + '\x1b[0m')
    raise ImportError

PICKNPLACE_DIR = 'C:\\Users\\harry\\Documents\\pb-construction\\pychoreo\\assembly_instances\\picknplace'
PICKNPLACE_FILENAMES = {
    'toggle_rebar_cage_1': 'toggle_rebar_cage_1.json'
}

scale_map = {
             'millimeter' : 1e-3,
             'meter' : 1.0,
            }


def add_sub_id_tag(obj_key, sub_mesh_id, suffix='.obj'):
    return obj_key + '_' + str(sub_mesh_id) + suffix


def extract_file_name(str_key):
    key_sep = str_key.split('.')
    return key_sep[0]


def rebuild_pkg_w_convex_collision_objects(instance_name, decomp_res=300000, verbose=True, write_log=False):
    instance_directory = os.path.join(PICKNPLACE_DIR, instance_name)
    print('Name: {}'.format(instance_name))
    json_file_path = os.path.join(instance_directory, 'json', PICKNPLACE_FILENAMES[instance_name])
    with open(json_file_path, 'r') as f:
        json_data = json.loads(f.read())

    mesh_dir = os.path.join(instance_directory, 'meshes', 'collision')

    # decompose element geometries
    for e in json_data['sequenced_elements']:
        for so_id, so in e['element_geometry_file_names'].items():
            so_full = so['full_obj']
            obj_name = extract_file_name(so_full)
            input_path = os.path.join(mesh_dir, so_full)
            output_path = input_path
            log_path = os.path.join(mesh_dir, obj_name + '.log') if write_log else ''

            print('computing: {}'.format(obj_name))
            success, mesh_verts, _ = compute_convex_decomp(input_path, output_path, log_path, resolution=decomp_res, verbose=verbose)
            assert(0 == success)
            n_convex_hulls = len(mesh_verts)
            print('V-HACD done! # of convex hulls: {}'.format(n_convex_hulls))

            so['convex_decomp'] = []
            for i in range(n_convex_hulls):
                so['convex_decomp'].append(add_sub_id_tag(obj_name, i))

    # decompose static collision objects
    for so_name, so_dict in json_data['static_obstacles'].items():
        for so_id, so in so_dict.items():
            so_full = so['full_obj']
            obj_name = extract_file_name(so_full)
            input_path = os.path.join(mesh_dir, so_full)
            output_path = input_path
            log_path = os.path.join(mesh_dir, obj_name + '.log') if write_log else ''

            print('computing: {}'.format(obj_name))
            success, mesh_verts, _ = compute_convex_decomp(input_path, output_path, log_path, resolution=decomp_res, verbose=verbose)
            assert(0 == success)
            n_convex_hulls = len(mesh_verts)
            print('V-HACD done! # of convex hulls: {}'.format(n_convex_hulls))

            so['convex_decomp'] = []
            for i in range(n_convex_hulls):
                so['convex_decomp'].append(add_sub_id_tag(obj_name, i))

    # overwrite data
    with open(json_file_path, 'w') as outfile:
        json.dump(json_data, outfile, indent=4)

    return json_data

################################
def main():
    parser = argparse.ArgumentParser()
    # toggle_rebar_cage_1 |
    parser.add_argument('-p', '--problem', default='toggle_rebar_cage_1', help='The name of the problem to rebuild')
    parser.add_argument('-res', '--res', default=100000, help='voxel resolution for V-HACD, default=100000')
    parser.add_argument('-v', '--viewer', action='store_true', help='Enables the pybullet viewer')
    parser.add_argument('-nrb', '--not_rebuild', action='store_false', help='not rebuild pkg, parse an existing one')
    parser.add_argument('-q', '--quiet', action='store_false', help='verbose output')
    args = parser.parse_args()
    print('Arguments:', args)

    instance_directory = os.path.join(PICKNPLACE_DIR, args.problem)
    if args.not_rebuild:
        json_data = rebuild_pkg_w_convex_collision_objects(args.problem, decomp_res=int(args.res), verbose=not args.quiet)
    else:
        json_file_path = os.path.join(instance_directory, 'json', PICKNPLACE_FILENAMES[args.problem])
        with open(json_file_path, 'r') as f:
            json_data = json.loads(f.read())

    # visualization
    connect(use_gui=args.viewer)
    obj_directory = os.path.join(instance_directory, 'meshes', 'collision')
    scale = scale_map[json_data['unit']]

    # element geometry
    for json_element in json_data['sequenced_elements']:
        index = json_element['order_id']
        # TODO: transform geometry based on json_element['parent_frame']

        obj_from_ee_grasp_poses = [pose_from_tform(parse_transform(json_tf)) \
                                    for json_tf in json_element['grasps']['ee_poses']]
        # pick_grasp_plane is at the top of the object with z facing downwards

        # ee_from_obj = invert(world_from_obj_pick) # Using pick frame
        pick_parent_frame = \
        pose_from_tform(parse_transform(json_element['assembly_process']['pick']['parent_frame']))
        world_from_obj_pick = \
        multiply(pick_parent_frame, pose_from_tform(parse_transform(json_element['assembly_process']['pick']['object_target_pose'])))

        place_parent_frame = \
        pose_from_tform(parse_transform(json_element['assembly_process']['place']['parent_frame']))
        world_from_obj_place = \
        multiply(place_parent_frame, pose_from_tform(parse_transform(json_element['assembly_process']['place']['object_target_pose'])))

        draw_pose(world_from_obj_pick, length=0.04)
        draw_pose(world_from_obj_place, length=0.04)

        so_dict = json_element['element_geometry_file_names']
        for sub_id, so in so_dict.items():
            pick_full_body =  create_obj(os.path.join(obj_directory, so['full_obj']), scale=scale, color=(0, 0, 1, 0.4))
            add_body_name(pick_full_body, 'e_' + str(index))
            set_pose(pick_full_body, world_from_obj_place)

            for cvd_obj in so['convex_decomp']:
                obj_name = extract_file_name(cvd_obj)
                pick_cvd_body =  create_obj(os.path.join(obj_directory, cvd_obj), scale=scale, color=(random(), random(), random(), 0.6))
                # add_body_name(obstacle_from_name[obj_name], obj_name)
                set_pose(pick_cvd_body, world_from_obj_place)

    # static collision
    obstacle_from_name = {}
    for so_name, so_dict in json_data['static_obstacles'].items():
        for sub_id, so in so_dict.items():
            obj_name = so_name + '_' + sub_id + '_full'
            obstacle_from_name[obj_name] =  create_obj(os.path.join(obj_directory, so['full_obj']),
                                   scale=scale, color=(0, 0, 1, 0.4))
            add_body_name(obstacle_from_name[obj_name], obj_name)

            for cvd_obj in so['convex_decomp']:
                obj_name = extract_file_name(cvd_obj)
                obstacle_from_name[obj_name] =  create_obj(os.path.join(obj_directory, cvd_obj), scale=scale, color=(random(), random(), random(), 0.6))
                # add_body_name(obstacle_from_name[obj_name], obj_name)

    wait_for_user()

if __name__ == '__main__':
    main()
