import warnings
from termcolor import cprint

from pybullet_planning import set_joint_positions, set_color, has_gui
from pybullet_planning import plan_joint_motion, get_collision_fn

from pychoreo.process_model.trajectory import MotionTrajectory

def solve_transition_between_extrusion_processes(robot, ik_joints, print_trajs, element_bodies, initial_conf,
                                                 obstacles=[], return2idle=True,
                                                 self_collisions=True, disabled_collisions={},
                                                 extra_disabled_collisions={},
                                                 ids_for_resolve=None, partial_process=False,
                                                 **kwargs):
    if ids_for_resolve is None:
        print('Resolving all the transition trajectories.')
        full_len = len(print_trajs)+1 if return2idle else len(print_trajs)
        ids_for_resolve = list(range(full_len))
    else:
        if len(ids_for_resolve) == 0:
            full_size = len(print_trajs) if not return2idle else len(print_trajs)+1
            ids_for_resolve = list(range(full_size))
            print('Given resolve ids empty, resolve for all, Sure?')
            input()
        else:
            ids_for_resolve = sorted(ids_for_resolve)
            print('Only the following ids will be resolved: {}'.format(ids_for_resolve))
            if return2idle and len(print_trajs) not in ids_for_resolve:
                print('return2idle index {} added.'.format(len(print_trajs)))
                ids_for_resolve.append(len(print_trajs))
            if partial_process:
                cprint('Only solving transitions between the specified ids, ignoring all others.', 'magenta')
    built_obstacles = []
    trans_traj = {}
    for seq_id in range(len(print_trajs)+1):
        if seq_id not in ids_for_resolve:
            # TODO: even if the seq_id is not solved, its printed element is added as a collision object
            if seq_id < len(print_trajs): #and not partial_process:
                try:
                    built_obstacles.append(element_bodies[tuple(print_trajs[seq_id][0].element)])
                except:
                    warnings.warn('element attribute not specified in print traj: {}'.format(print_trajs[seq_id][0]))
                    for pt in print_trajs[seq_id]:
                        cprint(pt, 'yellow')
            continue
        if seq_id < len(print_trajs):
            if not partial_process:
                if seq_id > 0:
                    last_seq_id = seq_id - 1
                    tr_start_conf = print_trajs[last_seq_id][-1].traj_path[-1]
                else:
                    tr_start_conf = initial_conf
            else:
                if ids_for_resolve.index(seq_id) > 0:
                    last_seq_id = ids_for_resolve[ids_for_resolve.index(seq_id) - 1]
                    tr_start_conf = print_trajs[last_seq_id][-1].traj_path[-1]
                else:
                    tr_start_conf = initial_conf
            tr_end_conf = print_trajs[seq_id][0].traj_path[0]
            tag_msg = 'Tr2 E#{}'.format(tuple(print_trajs[seq_id][0].element))
        elif return2idle:
            if partial_process:
                last_id = ids_for_resolve[-2]
                if last_id > len(print_trajs) - 1: last_id = len(print_trajs) - 1
            else:
                last_id = len(print_trajs) - 1
            tr_start_conf = print_trajs[last_id][-1].traj_path[-1]
            tr_end_conf = initial_conf
            tag_msg = 'return2idle'
        else:
            break
        print('='*10)
        print('transition seq #{}/{}: {}'.format(seq_id, len(print_trajs)-1, tag_msg))
        # TODO: can use robot, joints from the trajectory class itself as well
        set_joint_positions(robot, ik_joints, tr_start_conf)
        tr_path = plan_joint_motion(robot, ik_joints, tr_end_conf,
                                    obstacles=obstacles + built_obstacles,
                                    self_collisions=self_collisions, disabled_collisions=disabled_collisions,
                                    extra_disabled_collisions=extra_disabled_collisions,
                                    **kwargs)
        if not tr_path:
            cprint('seq #{}:{} cannot find transition path'.format(seq_id, tag_msg),
                'green', 'on_red')
            print('Diagnosis...')
            from pybullet_planning import MAX_DISTANCE
            max_distance = kwargs['max_distance'] if 'max_distance' in kwargs else MAX_DISTANCE
            print('max_distance: ', max_distance)
            cfn = get_collision_fn(robot, ik_joints, obstacles=obstacles + built_obstacles, attachments=[],
                                   self_collisions=True, disabled_collisions=disabled_collisions,
                                   extra_disabled_collisions=extra_disabled_collisions,
                                   max_distance=max_distance)

            if has_gui():
                for b in obstacles + built_obstacles:
                    set_color(b, (0,0,1,0.3))
            print('start pose:')
            cfn(tr_start_conf, diagnosis=True)

            print('end pose:')
            cfn(tr_end_conf, diagnosis=True)
        else:
            cprint('Trajectory found!', 'green')
        trans_traj[seq_id] = MotionTrajectory(robot, ik_joints, tr_path, element_id=seq_id, tag=tag_msg, planner_parameters=kwargs)
        # add printed element to the built obstacles
        if seq_id < len(print_trajs):
            built_obstacles.append(element_bodies[tuple(print_trajs[seq_id][0].element)])
    return trans_traj
