from simulate_paper import make_model
import mujoco
import json
from pathlib import Path
m,theta0,_,_=make_model(ground=False,dt=.00025)
m.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
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
p=Path(__file__).resolve().parent/'paper_analysis'/'verification.json'
r=json.loads(p.read_text())
r['passive_10s_RK4_max_energy_error_J']=largest
r['passive_10s_initial_energy_J']=E0
p.write_text(json.dumps(r,indent=2),encoding='utf-8')
print(json.dumps(r,indent=2))
