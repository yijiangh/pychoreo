import os
import json
import datetime
from collections import defaultdict, OrderedDict

from pybullet_planning import is_connected

def export_trajectory(save_dir, trajs, ee_link_name=None, overwrite=True, shape_file_path='', indent=None, include_robot_data=True, include_link_path=True):
    if include_robot_data and include_link_path:
        assert is_connected(), 'needs to be connected to a pybullet client to get robot/FK data'

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

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = OrderedDict()
    data['assembly_type'] = 'extrusion'
    data['file_name'] = file_name
    data['write_time'] = str(datetime.datetime.now())

    data['trajectory'] = []
    for cp_id, cp_trajs in enumerate(trajs):
        for sp_traj in cp_trajs:
            if ee_link_name and include_link_path:
                ee_link_path = sp_traj.get_link_path(ee_link_name)
        data['trajectory'].append([sp_traj.to_data(include_robot_data=True, include_link_path=True) for sp_traj in cp_trajs])

    full_save_path = os.path.join(save_dir, '{}_result_{}.json'.format(file_name,  '_'+data['write_time'] if not overwrite else ''))
    with open(full_save_path, 'w') as f:
        json.dump(data, f, indent=indent)
