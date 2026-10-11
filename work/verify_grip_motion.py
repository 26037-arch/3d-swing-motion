from pathlib import Path
import json
import argparse
import numpy as np
import mujoco
from arm_kinematics import arm_pose, place_visual_arms, rotate_xz, UPPER_ARM, FOREARM_TO_GRIP, GRIP_HEIGHT, GRIP_Y
from render_motion_3d import create_visual_model

root = Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser()
parser.add_argument('--bend-rule',choices=['clockwise','counterclockwise'],default='clockwise')
args=parser.parse_args()
sign=-1 if args.bend_rule=='clockwise' else 1
cases = json.loads((root/'work/motion_3d/motion-data.json').read_text(encoding='utf-8'))
report = {}
for name, case in cases.items():
    pars = case['parameters']
    length_errors, grip_errors, signed_areas, angles, distances = [], [], [], [], []
    for t, theta, phi, psi, radius, speed in case['frames']:
        for side in [-1, 1]:
            shoulder, elbow, hand, angle = arm_pose(np.radians(theta), np.radians(phi), radius, pars, side,args.bend_rule)
            length_errors += [abs(np.linalg.norm(elbow-shoulder)-UPPER_ARM), abs(np.linalg.norm(hand-elbow)-FOREARM_TO_GRIP)]
            expected_hand = np.array([0,0,pars['pivot_height_m']])+rotate_xz([0,side*GRIP_Y,-radius+GRIP_HEIGHT], np.radians(theta))
            grip_errors.append(np.linalg.norm(hand-expected_hand))
            # Measure the ordered triangle in the fixed side view.
            he, hs = elbow-hand, shoulder-hand
            signed_areas.append(.5*(-he[0]*hs[2]+he[2]*hs[0]))
            d = np.linalg.norm(hand-shoulder)
            angles.append(angle)
            distances.append(d)
    assert max(length_errors) < 1e-10
    assert max(grip_errors) < 1e-10
    assert min(sign*area for area in signed_areas)>1e-6
    report[name] = dict(frames=len(case['frames']), max_arm_length_error_m=max(length_errors),
                        max_hand_grip_error_m=max(grip_errors), side_view_signed_area_m2=[min(signed_areas),max(signed_areas)],
                        elbow_interior_angle_deg=[min(angles),max(angles)],
                        shoulder_hand_distance_m=[min(distances),max(distances)])

# Inspect the actual rendered geometry, after MuJoCo computes its world poses.
model, pars = create_visual_model()
state = mujoco.MjData(model)
geometry_errors=[]
for row in cases['rod-ideal']['frames'][::3]:
    state.qpos[:] = np.radians(row[1:4])
    poses = place_visual_arms(model, *state.qpos[:2], row[4], pars,args.bend_rule)
    mujoco.mj_forward(model, state)
    for label, (s,e,h,angle) in zip(['left','right'], poses):
        for segment, start, end in [('upper',s,e),('fore',e,h)]:
            gid = model.geom(f'arm_{label}_{segment}').id
            direction=state.geom_xmat[gid].reshape(3,3)[:,2]
            midpoint=state.geom_xpos[gid]
            half=model.geom_size[gid,1]
            geometry_errors += [np.linalg.norm(midpoint-half*direction-start),np.linalg.norm(midpoint+half*direction-end)]
        geometry_errors.append(np.linalg.norm(state.geom_xpos[model.geom(f'arm_{label}_hand').id]-h))
assert max(geometry_errors) < 1e-10
report['rendered_geometry_max_error_m']=max(geometry_errors)
report['arm_lengths_m']=[UPPER_ARM,FOREARM_TO_GRIP]
(root/f'work/motion_3d/{args.bend_rule}-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
