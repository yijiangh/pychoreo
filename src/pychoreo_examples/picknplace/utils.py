from compas_fab.backends.pybullet import pb_pose_from_Frame
from pybullet_planning import wait_for_user, has_gui, WorldSaver
from pybullet_planning import get_pose, set_pose, set_color
from pybullet_planning import pairwise_collision, pairwise_collision

def flatten_dict_entries(in_dict, keys):
    out_list = []
    for k in keys:
        out_list.extend(in_dict[k])
    return out_list


def sanity_check_collisions(elements_from_index, obstacle_from_name):
    """check if elements are colliding with the static obstalces in its assembled state.
    # TODO: check if elements are colliding with each other.

    Parameters
    ----------
    elements_from_index : [type]
        [description]
    obstacle_from_name : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    in_collision = False
    init_pose = None
    with WorldSaver():
        for brick in elements_from_index.values():
            for e_body in brick.pybullet_bodies:
                if not init_pose:
                    init_pose = get_pose(e_body)
                for so_id, so in obstacle_from_name.items():
                    set_pose(e_body, pb_pose_from_Frame(brick.initial_frame))
                    if pairwise_collision(e_body, so):
                        set_color(e_body, (1, 0, 0, 0.6))
                        set_color(so, (0, 0, 1, 0.6))

                        in_collision = True
                        print('collision detected between brick #{} and static #{} in its pick pose'.format(brick.name, so_id))
                        if has_gui():
                            wait_for_user()
                        else:
                            input()

                    set_pose(e_body, pb_pose_from_Frame(brick.goal_frame))
                    if pairwise_collision(e_body, so):
                        in_collision = True
                        print('collision detected between brick #{} and static #{} in its place pose'.format(brick.name, so_id))
                        if has_gui():
                            wait_for_user()
                        else:
                            input()

        # # reset their poses for visual...
        for brick in elements_from_index.values():
            for e_body in brick.pybullet_bodies:
                set_pose(e_body, init_pose)

    return in_collision
