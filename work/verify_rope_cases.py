from pathlib import Path
import json
import csv
import time
from simulate_paper import simulate

folder=Path(__file__).resolve().parent/'paper_analysis'
cases=[('rope_fine',0.00025,0.0,0.0),('rope_air',0.00025,0.4,0.4*7/3)]
for name,dt,drag,dry in cases:
    if (folder/f'{name}.json').exists():
        continue
    start=time.time()
    r,rows=simulate(amp=60.,leg_center=75.,leg_amp=10.,seconds=120.,
                    dt=dt,rope=True,drag=drag,dry=dry)
    (folder/f'{name}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    with (folder/f'{name}.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'case':name,'target_reached':r['target_reached'],'time_s':r['time_s'],
                      'max_angle_deg':r['max_angle_deg'],'failure':r['failure_reason'],
                      'max_tension_N':r['max_rope_tension_N'],'elapsed_s':time.time()-start}),flush=True)
