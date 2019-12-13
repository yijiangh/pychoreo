import warnings
from termcolor import cprint

from pybullet_planning import set_joint_positions
from pybullet_planning import plan_joint_motion, get_collision_fn

from pychoreo.process_model.trajectory import MotionTrajectory

def solve_transition_between_extrusion_processes(robot, ik_joints, print_trajs, element_bodies, initial_conf,
                                                 obstacles=[], return2idle=True,
                                                 self_collisions=True, disabled_collisions={},
                                                 **kwargs):
    built_obstacles = []
    trans_traj = []
    for seq_id in range(len(print_trajs)+1):
        if seq_id < len(print_trajs):
            print('transition seq #{}/{}'.format(seq_id, len(print_trajs)-1))
            # if not print_trajs[seq_id-1].traj_path or not print_trajs[seq_id].traj_path:
            #     warnings.warn('print trajectory {} or {} not found, skip'.format(seq_id-1, seq_id))
            #     continue
            if seq_id != 0:
                tr_start_conf = print_trajs[seq_id-1][-1].traj_path[-1]
            else:
                tr_start_conf = initial_conf
            tr_end_conf = print_trajs[seq_id][0].traj_path[0]
        elif return2idle:
            print('plan for returning to idle position.')
            tr_start_conf = print_trajs[-1][-1].traj_path[-1]
            tr_end_conf = initial_conf
        else:
            break
        # TODO: can use robot, joints from the trajectory class itself as well
        set_joint_positions(robot, ik_joints, tr_start_conf)
        tr_path = plan_joint_motion(robot, ik_joints, tr_end_conf,
                                    obstacles=obstacles + built_obstacles,
                                    self_collisions=self_collisions, disabled_collisions=disabled_collisions,
                                    **kwargs)
        if not tr_path:
            cprint('seq #{} cannot find transition path'.format(seq_id), 'green', 'on_red')
            print('Diagnosis...')
            from pybullet_planning import MAX_DISTANCE
            max_distance = kwargs['max_distance'] if 'max_distance' in kwargs else MAX_DISTANCE
            print('max_distance: ', max_distance)
            cfn = get_collision_fn(robot, ik_joints, obstacles=obstacles + built_obstacles, attachments=[],
                                   self_collisions=True, disabled_collisions=disabled_collisions,
                                   max_distance=max_distance)

            print('start pose:')
            cfn(tr_start_conf, diagnosis=True)

            print('end pose:')
            cfn(tr_end_conf, diagnosis=True)
        trans_traj.append(MotionTrajectory(robot, ik_joints, tr_path))
        # add printed element to the built obstacles
        if seq_id < len(print_trajs):
            built_obstacles.append(element_bodies[tuple(print_trajs[seq_id][0].element)])
    return trans_traj
