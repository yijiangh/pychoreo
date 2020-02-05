import os
import json
import datetime
from collections import defaultdict, OrderedDict
from termcolor import cprint

from pybullet_planning import is_connected

def get_saved_file_path(save_dir, overwrite=True, shape_file_path='', file_tag=''):
    if os.path.exists(shape_file_path):
        with open(shape_file_path, 'r') as f:
            shape_data = json.loads(f.read())
        if 'model_name' in shape_data:
            file_name = shape_data['model_name']
        else:
            file_name = shape_file_path.split('.json')[-2].split(os.sep)[-1]
    else:
        file_name = 'pychoreo_result'
        overwrite = False

    full_save_path = os.path.join(save_dir, '{}_result_{}{}.json'.format(file_name, file_tag, '_'+str(datetime.datetime.now()) if not overwrite else ''))
    return full_save_path

def export_trajectory(save_dir, trajs, ee_link_name=None, overwrite=True, shape_file_path='', assembly_type='',
    indent=None, include_robot_data=True, include_link_path=True, file_tag='', verbose=False):
    if include_robot_data and include_link_path:
        assert is_connected(), 'needs to be connected to a pybullet client to get robot/FK data'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = OrderedDict()
    data['assembly_type'] = assembly_type
    data['write_time'] = str(datetime.datetime.now())

    data['trajectory'] = []
    for cp_id, cp_trajs in enumerate(trajs):
        for sp_traj in cp_trajs:
            if ee_link_name and include_link_path:
                ee_link_path = sp_traj.get_link_path(ee_link_name)
        traj_data = []
        for sp_traj in cp_trajs:
            traj_data.append(sp_traj.to_data(include_robot_data=True, include_link_path=include_link_path))
        data['trajectory'].append(traj_data)

    full_save_path = get_saved_file_path(save_dir, overwrite=overwrite, shape_file_path=shape_file_path, file_tag=file_tag)

    with open(full_save_path, 'w') as f:
        json.dump(data, f, indent=indent)
    if verbose:
        cprint('Trajectory saved to {}'.format(full_save_path), 'green')
