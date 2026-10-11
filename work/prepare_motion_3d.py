from pathlib import Path
import json
import csv
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
folder=ROOT/'work'/'motion_3d'
folder.mkdir(exist_ok=True)
summary=json.loads((ROOT/'outputs'/'swing-180-results.json').read_text(encoding='utf-8'))
names={'rod-ideal':'봉 · 저항 없음','rod-with-resistance':'봉 · 저항 포함',
       'rope-ideal':'줄 · 외부 저항 없음','rope-with-resistance':'줄 · 외부 저항 포함'}
cases={}
for name,label in names.items():
    r=summary['main_cases'][name]
    d=np.genfromtxt(ROOT/'outputs'/f'{name}-data.csv',delimiter=',',names=True,encoding='utf-8-sig')
    end=r['time_s']
    times=np.r_[np.arange(0,end,1/20),end]
    initial={'theta_deg':r['parameters']['initial_theta_deg'],'hip_deg':0,'knee_deg':0,
             'seat_radius_m':2.2,'com_speed_m_s':0}
    cols=[times]
    for key in ['theta_deg','hip_deg','knee_deg','seat_radius_m','com_speed_m_s']:
        cols.append(np.interp(times,np.r_[0,d['time_s']],np.r_[initial[key],d[key]]))
    if r['target_reached']:cols[1][-1]=180
    table=[[round(float(x),precision) for x,precision in zip(row,[4,3,3,3,4,3])]
           for row in np.column_stack(cols)]
    cases[name]={'label':label,'end':round(end,5),'reached':r['target_reached'],
                 'maximum':round(min(180,r['max_angle_deg']),1),'parameters':r['parameters'],
                 'frames':table}
(folder/'motion-data.json').write_text(json.dumps(cases,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
print(json.dumps({'frames':{n:len(c['frames']) for n,c in cases.items()},'bytes':(folder/'motion-data.json').stat().st_size}))
