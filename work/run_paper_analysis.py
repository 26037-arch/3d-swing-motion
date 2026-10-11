from pathlib import Path
import json
import csv
import time
from simulate_paper import simulate,make_model
import mujoco
import numpy as np

base=Path(__file__).resolve().parent
folder=base/'paper_analysis'
folder.mkdir(exist_ok=True)
cases=[
    ('ideal_coarse',{'dt':.001}),
    ('ideal_medium',{'dt':.0005}),
    ('ideal',{'dt':.00025}),
    ('air_medium',{'dt':.0005,'drag':.4,'dry':.4*7/3}),
    ('air',{'dt':.00025,'drag':.4,'dry':.4*7/3}),
    ('rope',{'dt':.0005,'rope':True,'seconds':120}),
    ('short_person',{'dt':.0005,'height':1.6}),
    ('tall_person',{'dt':.0005,'height':1.9}),
    ('higher_pivot',{'dt':.0005,'pivot_height':2.55}),
    ('lower_pivot',{'dt':.0005,'pivot_height':2.45}),
    ('smaller_lean',{'dt':.0005,'amp':45}),
    ('less_torque',{'dt':.0005,'hip_max':400.,'knee_max':200.}),
]
summary={}
for label,params in cases:
    if (folder/f'{label}.json').exists():
        summary[label]=json.loads((folder/f'{label}.json').read_text(encoding='utf-8'))
        continue
    started=time.time()
    options=dict(amp=60.,leg_center=75.,leg_amp=10.,seconds=120)
    options.update(params)
    r,rows=simulate(**options)
    summary[label]=r
    (folder/f'{label}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    with (folder/f'{label}.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(case=label,target=r['target_reached'],time=r['time_s'],max_angle=r['max_angle_deg'],failure=r['failure_reason'],elapsed=time.time()-started)),flush=True)

# Independent energy conservation check with all posture coordinates free and
# no controller, damping, ground or contacts. The physical model is unchanged.
m,theta0,_,_=make_model(ground=False,dt=.00025)
d=mujoco.MjData(m)
d.qpos[:]=[theta0,.1,.15]
d.qvel[:]=[.2,-.1,.1]
mujoco.mj_forward(m,d)
E0=float(d.energy.sum());largest=0.
for i in range(40000):
    d.ctrl[:]=0
    mujoco.mj_step(m,d)
    if i%10==0:
        mujoco.mj_forward(m,d)
        largest=max(largest,abs(float(d.energy.sum())-E0))
verification={'passive_10s_max_energy_error_J':largest,
    'ideal_step_difference_s':abs(summary['ideal']['time_s']-summary['ideal_medium']['time_s']),
    'air_step_difference_s':abs(summary['air']['time_s']-summary['air_medium']['time_s'])}
(folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(folder/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print(json.dumps(verification),flush=True)
