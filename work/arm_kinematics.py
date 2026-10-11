"""Visual two-link arms with a fixed grip and selectable elbow branch."""
import numpy as np

UPPER_ARM = 0.32
FOREARM_TO_GRIP = 0.34
GRIP_HEIGHT = 0.08
SHOULDER_Y = 0.20
GRIP_Y = 0.24


def rotate_xz(v, angle):
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = v
    return np.array([c*x-s*z, y, s*x+c*z])


def arm_pose(theta, phi, radius, pars, side, bend_rule='clockwise'):
    pivot = np.array([0.0, 0.0, pars['pivot_height_m']])
    hip = np.array([-pars['a_m'], side*SHOULDER_Y, -radius])
    shoulder_local = hip + rotate_xz([0, 0, pars['torso_length_m']*.58], phi)
    hand_local = np.array([0.0, side*GRIP_Y, -radius+GRIP_HEIGHT])
    shoulder = pivot + rotate_xz(shoulder_local, theta)
    hand = pivot + rotate_xz(hand_local, theta)
    delta = hand-shoulder
    distance = np.linalg.norm(delta)
    if not abs(UPPER_ARM-FOREARM_TO_GRIP) < distance < UPPER_ARM+FOREARM_TO_GRIP:
        raise ValueError(f'Unreachable grip: shoulder-to-hand distance {distance:.6f} m')
    direction = delta/distance
    along = (UPPER_ARM**2-FOREARM_TO_GRIP**2+distance**2)/(2*distance)
    circle_center = shoulder+along*direction
    circle_radius = np.sqrt(max(0.0, UPPER_ARM**2-along**2))
    if bend_rule in ('clockwise', 'counterclockwise'):
        # The GIF camera is on the +y side: screen right is world -x.
        # Hand -> elbow -> shoulder must have a negative signed area
        # in (-x,z). Keep this branch through the upside-down pose.
        bend_direction = np.array([-direction[2], 0.0, direction[0]])
        if bend_rule == 'counterclockwise':
            bend_direction = -bend_direction
    elif bend_rule == 'world-down':
        down = np.array([0.0, 0.0, -1.0])
        bend_direction = down-np.dot(down, direction)*direction
    else:
        raise ValueError(f'Unknown elbow rule: {bend_rule}')
    norm = np.linalg.norm(bend_direction)
    if norm < 1e-10:
        raise ValueError('Elbow direction is undefined in the side view')
    elbow = circle_center+circle_radius*bend_direction/norm
    elbow_angle = np.degrees(np.arccos(np.clip(
        (UPPER_ARM**2+FOREARM_TO_GRIP**2-distance**2)/(2*UPPER_ARM*FOREARM_TO_GRIP), -1, 1)))
    return shoulder, elbow, hand, float(elbow_angle)


def place_visual_arms(model, theta, phi, radius, pars, bend_rule='clockwise'):
    """Move massless world geoms, without changing any simulated coordinates."""
    import mujoco
    poses = []
    for side, label in [(-1, 'left'), (1, 'right')]:
        shoulder, elbow, hand, angle = arm_pose(theta, phi, radius, pars, side, bend_rule)
        for segment, start, end in [('upper', shoulder, elbow), ('fore', elbow, hand)]:
            gid = model.geom(f'arm_{label}_{segment}').id
            model.geom_pos[gid] = (start+end)*.5
            model.geom_size[gid, 1] = np.linalg.norm(end-start)*.5
            # The compiler optimized the placeholder's identity rotation.
            # Dynamic orientations require the general world-transform path.
            model.geom_sameframe[gid] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE
            mujoco.mju_quatZ2Vec(model.geom_quat[gid], end-start)
        model.geom_pos[model.geom(f'arm_{label}_elbow').id] = elbow
        model.geom_pos[model.geom(f'arm_{label}_hand').id] = hand
        poses.append((shoulder, elbow, hand, angle))
    return poses
