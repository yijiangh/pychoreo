
from pybullet_planning import set_joint_positions
from pybullet_planning import plan_joint_motion, get_collision_fn

from pychoreo.process_model.trajectory import MotionTrajectory

def solve_transition_between_extrusion_processes(robot, ik_joints, print_trajs, element_bodies, initial_conf,
                                                 obstacles=[], return2idle=True, self_collisions=True, disabled_collisions={},
                                                 weights=None, resolutions=None, custom_limits={}):
    built_obstacles = []
    trans_traj = []
    for seq_id in range(len(print_trajs)+1):
        if seq_id < len(print_trajs):
            print('transition seq #{}'.format(seq_id))
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
                                    weights=weights, resolutions=resolutions, custom_limits=custom_limits)
        if not tr_path:
            print('seq #{} cannot find transition path'.format(seq_id))
            print('Diagnosis...')

            cfn = get_collision_fn(robot, ik_joints, obstacles=obstacles + built_obstacles, attachments=[],
                                   self_collisions=True, disabled_collisions=disabled_collisions)

            print('start pose:')
            cfn(tr_start_conf, diagnosis=True)

            print('end pose:')
            cfn(tr_end_conf)
        trans_traj.append(MotionTrajectory(robot, ik_joints, tr_path))
        # add printed element to the built obstacles
        if seq_id < len(print_trajs):
            built_obstacles.append(element_bodies[print_trajs[seq_id][0].element])
    return trans_traj
