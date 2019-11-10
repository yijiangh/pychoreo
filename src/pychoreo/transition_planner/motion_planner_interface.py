
from pybullet_planning import set_joint_positions
from pybullet_planning import plan_joint_motion

def solve_transition_between_cartesian_processes():
    pass

    static_obstacles = list(obstacle_from_name.values())
    # reset brick poses
    for e_id in element_seq.values():
        for e_body in brick_from_index[e_id].body:
            set_pose(e_body, brick_from_index[e_id].initial_pose)

    for seq_id, e_id in element_seq.items():
        print('transition seq#{}'.format(seq_id))
        picknplace_unit = picknplace_cart_plans[seq_id]
        # brick = brick_from_index[e_id]

        if seq_id != 0:
            tr_start_conf = picknplace_cart_plans[seq_id-1]['place_retreat'][-1]
        else:
            tr_start_conf = initial_conf
        set_joint_positions(robot, movable_joints, tr_start_conf)

        cur_mo_list = []
        for mo_id, mo in brick_from_index.items():
            if mo_id in element_seq.values():
                cur_mo_list.extend(mo.body)

        place2pick_path = plan_joint_motion(robot, movable_joints,
                            picknplace_cart_plans[seq_id]['pick_approach'][0],
                            obstacles=static_obstacles + cur_mo_list, self_collisions=SELF_COLLISIONS)
        if not place2pick_path:
            print('seq #{} cannot find place2pick transition'.format(seq_id))
            print('Diagnosis...')

            cfn = get_collision_fn_diagnosis(robot, movable_joints, obstacles=static_obstacles + cur_mo_list, attachments=[], self_collisions=SELF_COLLISIONS)

            print('start pose:')
            cfn(tr_start_conf)

            end_conf = picknplace_cart_plans[seq_id]['pick_approach'][0]
            print('end pose:')
            cfn(end_conf)

        # create attachement without needing to keep track of grasp...
        set_joint_positions(robot, movable_joints, picknplace_cart_plans[seq_id]['pick_retreat'][0])
        # attachs = [Attachment(robot, tool_link, invert(grasp.attach), e_body) for e_body in brick.body]
        attachs = [create_attachment(robot, end_effector_link, e_body) for e_body in brick_from_index[e_id].body]

        cur_mo_list = []
        for mo_id, mo in brick_from_index.items():
            if mo_id != e_id and mo_id in element_seq.values():
                cur_mo_list.extend(mo.body)

        set_joint_positions(robot, movable_joints, picknplace_cart_plans[seq_id]['pick_retreat'][-1])
        pick2place_path = plan_joint_motion(robot, movable_joints, picknplace_cart_plans[seq_id]['place_approach'][0], obstacles=static_obstacles + cur_mo_list, attachments=attachs, self_collisions=SELF_COLLISIONS, )
        resolutions = TRANSITION_JT_RESOLUTION*np.ones(len(movable_joints))
        # if not pick2place_path:
        #     print('seq #{} cannot find pick2place transition'.format(seq_id))
        #     print('Diagnosis...')

        #     cfn = get_collision_fn_diagnosis(robot, movable_joints, obstacles=static_obstacles + cur_mo_list, attachments=attachs, self_collisions=SELF_COLLISIONS)

        #     print('start pose:')
        #     cfn(picknplace_cart_plans[seq_id]['pick_retreat'][-1])

        #     end_conf = picknplace_cart_plans[seq_id]['place_approach'][0]
        #     print('end pose:')
        #     cfn(end_conf)

        picknplace_cart_plans[seq_id]['place2pick'] = place2pick_path
        picknplace_cart_plans[seq_id]['pick2place'] = pick2place_path

        # set e_id element to goal pose
        for mo in brick_from_index[e_id].body:
            set_pose(mo, brick_from_index[e_id].goal_pose)
