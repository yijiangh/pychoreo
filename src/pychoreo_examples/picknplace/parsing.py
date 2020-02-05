import json

from pychoreo.process_model.trajectory import MotionTrajectory
from pychoreo_examples.picknplace.trajectory import PicknPlaceBufferTrajectory

##################################################

def parse_saved_trajectory(file_path):
    with open(file_path, 'r')  as f:
        data = json.load(f)
    try:
        print('file_name: {} | write_time: {} | '.format(file_path, data['write_time']))
    except:
        pass
    full_traj = []
    for proc_traj_data in data['trajectory']:
        proc_traj_recon = []
        for sp_traj_data in proc_traj_data:
            if sp_traj_data['traj_type'] == 'MotionTrajectory':
                sp_traj = MotionTrajectory.from_data(sp_traj_data)
            if sp_traj_data['traj_type'] == 'PicknPlaceBufferTrajectory':
                sp_traj = PicknPlaceBufferTrajectory.from_data(sp_traj_data)
            proc_traj_recon.append(sp_traj)
        full_traj.append(proc_traj_recon)
    return full_traj
