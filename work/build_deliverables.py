from pathlib import Path
import json
import csv
import shutil
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent.parent
SRC=ROOT/'work'/'paper_analysis'
OUT=ROOT/'outputs'
OUT.mkdir(exist_ok=True)
font_manager.fontManager.addfont(r'C:\Windows\Fonts\malgun.ttf')
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,
    'font.size':12,'axes.titlesize':16,'axes.labelsize':12,'legend.fontsize':11,
    'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
    'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False})
colors={'ideal':'#1864AB','air':'#C55222','rope_fine':'#087F5B','rope_air':'#7B4BB7'}
labels={'ideal':'봉 · 저항 없음','air':'봉 · 공기저항·축마찰 포함',
        'rope_fine':'줄·체인 등가모델 · 외부 저항 없음','rope_air':'줄·체인 등가모델 · 외부 저항 포함'}
names=list(labels)
results={n:json.loads((SRC/f'{n}.json').read_text(encoding='utf-8')) for n in names}
data={n:np.genfromtxt(SRC/f'{n}.csv',delimiter=',',names=True,encoding='utf-8-sig') for n in names}
# Existing runs logged a local foot-tip site. Correct it to the sphere's
# vertical clearance; this is a diagnostic correction, not a dynamics change.
for n in names:
    d=data[n];radius=results[n]['parameters']['foot_radius_m']
    d['foot_height_m']+=radius*(np.cos(np.deg2rad(d['theta_deg']+d['knee_deg']))-1)
    results[n]['lowest_foot_m']=float(np.min(d['foot_height_m']))
summary=json.loads((SRC/'summary.json').read_text(encoding='utf-8'))
verification=json.loads((SRC/'verification.json').read_text(encoding='utf-8'))
verification['rope_max_angle_step_difference_deg']=abs(results['rope_fine']['max_angle_deg']-summary['rope']['max_angle_deg'])
verification['rope_peak_tension_step_difference_fraction']=abs(results['rope_fine']['max_rope_tension_N']-summary['rope']['max_rope_tension_N'])/results['rope_fine']['max_rope_tension_N']

def save(fig,stem):
    fig.savefig(OUT/f'{stem}.png',dpi=220)
    fig.savefig(OUT/f'{stem}.svg')
    plt.close(fig)

def grid(ax):
    ax.grid(True,axis='y',color='#DBE1E7',linewidth=.7)
    ax.set_axisbelow(True)

# Scientific figures, sized for a 16:9 PowerPoint slide.
fig=plt.figure(figsize=(14.4,8.1))
gs=fig.add_gridspec(2,1,left=.09,right=.95,bottom=.13,top=.83,hspace=.29,height_ratios=[1.5,1.])
ax=fig.add_subplot(gs[0]);ae=fig.add_subplot(gs[1],sharex=ax)
for n in ['ideal','air']:
    d=data[n];r=results[n]
    ax.plot(d['time_s'],d['theta_deg'],color=colors[n],lw=1.6,label=labels[n])
    ae.plot(d['time_s'],(d['energy_J']-d['energy_J'][0])/1000,color=colors[n],lw=1.5)
    t=r['time_s'];ax.scatter([t],[180],s=42,color=colors[n],zorder=5)
    ax.annotate(f"{t:.1f}초",(t,180),xytext=(t-1,211),ha='center',fontsize=14,
                color=colors[n],weight='bold',arrowprops={'arrowstyle':'-','color':colors[n]})
ax.axhline(180,color='#283848',ls='--',lw=1.1)
ax.axhline(0,color='#AAB3BC',lw=.7)
ax.set_ylim(-175,231);ax.set_yticks([-135,-90,0,90,180]);ax.set_ylabel('그네 각도 θ (°)')
ax.legend(loc='upper left',frameon=False,ncol=2,bbox_to_anchor=(0,1.13))
ae.set_xlim(0,42);ae.set_ylim(-.3,3.1);ae.set_ylabel('역학적 에너지 증가 (kJ)');ae.set_xlabel('정지 출발 후 시간 (s)')
for a in [ax,ae]:grid(a)
fig.suptitle('단단한 봉: θ=180°에 처음 도달하는 시간',x=.09,ha='left',y=.96,fontsize=22,weight='bold')
fig.text(.09,.90,'키 1.75 m · 질량 70 kg · 축 높이 2.50 m · 봉 길이 2.20 m · 초기 θ=-28.41°, 모든 속도 0',fontsize=12,color='#526171')
fig.text(.09,.045,'논문 기반 3분절 인체의 초인 능력 가정 시뮬레이션. 실제 측정값 또는 전역 최소시간을 뜻하지 않음.',fontsize=10,color='#526171')
save(fig,'swing-180-time')

fig=plt.figure(figsize=(14.4,8.1))
gs=fig.add_gridspec(2,1,left=.09,right=.95,bottom=.14,top=.81,hspace=.31,height_ratios=[1.3,1])
ax=fig.add_subplot(gs[0]);at=fig.add_subplot(gs[1],sharex=ax)
for n in names:
    d=data[n]
    angle=np.minimum(180,np.maximum.accumulate(np.abs(d['theta_deg'])))
    ax.plot(d['time_s'],angle,label=labels[n],color=colors[n],lw=2 if n=='ideal' else 1.6)
for n in ['rope_fine','rope_air']:
    d=data[n];at.plot(d['time_s'],d['rope_tension_N']/1000,color=colors[n],lw=.8,alpha=.85)
ax.axhline(180,color='#283848',ls='--',lw=1)
ax.text(118,184,'목표 180°',ha='right',fontsize=11,color='#283848')
ax.text(118,121,f"120초간 미도달\n저항 없음: {results['rope_fine']['max_angle_deg']:.1f}°",ha='right',va='bottom',color=colors['rope_fine'],fontsize=12)
ax.text(118,99,f"저항 포함: {results['rope_air']['max_angle_deg']:.1f}°",ha='right',va='top',color=colors['rope_air'],fontsize=12)
ax.set_ylim(0,203);ax.set_yticks([0,45,90,135,180]);ax.set_ylabel('그때까지 최대 |θ| (°)')
ax.legend(loc='upper left',frameon=False,ncol=2,bbox_to_anchor=(0,1.24),columnspacing=1.8)
at.set_xlim(0,122);at.set_ylim(0,18);at.set_yticks([0,5,10,15]);at.set_ylabel('줄의 장력 (kN)');at.set_xlabel('정지 출발 후 시간 (s)')
for a in [ax,at]:grid(a)
fig.suptitle('봉과 줄·체인의 차이: 압축 지지와 느슨해짐',x=.09,ha='left',y=.96,fontsize=22,weight='bold')
fig.text(.09,.90,'줄·체인은 장력만 전달하는 탄성 등가모델: 강성 200 kN/m, 감쇠 500 N·s/m',fontsize=12,color='#526171')
fig.text(.09,.045,'줄이 느슨해질 때 좌석의 독립 회전·체인 링크 충돌은 생략. 미도달은 이 동작·모델의 120초 결과이며 불가능성 증명이 아님.',fontsize=10,color='#526171')
save(fig,'swing-rope-vs-rod')

comparison=[('기본 가정','ideal'),('사람 키 1.60 m','short_person'),('사람 키 1.90 m','tall_person'),
            ('축 높이 2.45 m','lower_pivot'),('축 높이 2.55 m','higher_pivot'),
            ('상체 목표 진폭 45°','smaller_lean'),('공기저항·축마찰 포함','air')]
fig,ax=plt.subplots(figsize=(14.4,8.1))
fig.subplots_adjust(left=.28,right=.91,bottom=.2,top=.80)
vals=[summary[n]['time_s'] for _,n in comparison]
y=np.arange(len(vals))
barcolors=['#1864AB' if n=='ideal' else '#C55222' if n=='air' else '#6E899E' for _,n in comparison]
ax.barh(y,vals,height=.58,color=barcolors)
ax.set_yticks(y,[x[0] for x in comparison]);ax.invert_yaxis();ax.set_xlim(0,47);ax.set_xlabel('θ=180° 최초 도달 시간 (s)')
for i,v in enumerate(vals):ax.text(v+.6,i,f'{v:.1f}초',va='center',fontsize=13)
ax.axvline(summary['ideal']['time_s'],color='#1864AB',ls=':',lw=1.1)
ax.grid(True,axis='x',color='#DBE1E7',linewidth=.7);ax.set_axisbelow(True)
fig.suptitle('가정을 바꾸면 도달 시간도 달라진다',x=.08,ha='left',y=.95,fontsize=22,weight='bold')
fig.text(.08,.875,'봉 모델에서 한 조건씩 변경 · 키·축 높이 변경 시 발의 지면 접촉 조건으로 초기 각도 재계산',fontsize=12,color='#526171')
fig.text(.08,.105,f"토크 한계를 절반(400 / 200 N·m)으로 줄이면: 120초간 미도달, 최대 {summary['less_torque']['max_angle_deg']:.1f}°",fontsize=13,color='#C55222')
fig.text(.08,.045,'모든 경우 같은 동작 제어 규칙 사용. 각 조건에 맞춘 최적 동작을 별도로 탐색한 결과는 아님.',fontsize=10,color='#526171')
save(fig,'swing-assumption-comparison')

fig=plt.figure(figsize=(14.4,8.1))
gs=fig.add_gridspec(2,2,left=.09,right=.95,bottom=.13,top=.82,hspace=.34,wspace=.29)
axes=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
d=data['ideal']
axes[0].plot(d['time_s'],d['hip_deg'],color='#1864AB',lw=1.4,label='상체 φ')
axes[0].plot(d['time_s'],d['knee_deg'],color='#C55222',lw=1.4,label='종아리 ψ')
axes[0].set_ylabel('좌석에 대한 각도 (°)');axes[0].legend(frameon=False,ncol=2)
axes[1].plot(d['time_s'],d['hip_torque_Nm'],color='#1864AB',lw=1,label='상체')
axes[1].plot(d['time_s'],d['knee_torque_Nm'],color='#C55222',lw=1,label='무릎')
axes[1].set_ylabel('내부 관절 토크 (N·m)')
axes[2].plot(d['time_s'],d['com_length_m'],color='#087F5B',lw=1.3)
axes[2].set_ylabel('축–전체 질량중심 거리 (m)')
axes[3].plot(d['time_s'],d['com_speed_m_s'],color='#7B4BB7',lw=1.3)
axes[3].set_ylabel('전체 질량중심 속력 (m/s)')
for a in axes:a.set_xlim(0,28);a.set_xlabel('시간 (s)');grid(a)
fig.suptitle('27.1초를 만드는 몸동작과 실제 필요 능력',x=.09,ha='left',y=.96,fontsize=22,weight='bold')
fig.text(.09,.895,'봉 · 외부 저항 없음 | 상체 최대 토크 800 N·m, 무릎 약 330 N·m, 전체 질량중심 최대 속력 8.47 m/s',fontsize=12,color='#526171')
fig.text(.09,.045,'몸동작으로 질량중심 거리와 관성모멘트가 함께 변함. 관절 토크의 반작용을 포함하며, 축에는 구동 토크를 가하지 않음.',fontsize=10,color='#526171')
save(fig,'swing-body-motion')

fig,ax=plt.subplots(figsize=(14.4,8.1));ax.axis('off')
fig.subplots_adjust(left=.06,right=.94,bottom=.13,top=.80)
table_rows=[
    ['사람 / 설치 치수','키 1.75 m · 사람 70 kg · 축 높이 2.50 m · 연결부 2.20 m'],
    ['출발 조건','초기 θ=-28.41° · 모든 속도 0 · 기준 앉은 자세에서 발이 지면에 접촉'],
    ['상체 / 무릎 토크 한계','800 / 400 N·m (양측 합산)'],
    ['힘으로 환산한 능력','상체 16 kN (모멘트팔 5 cm) · 무릎 10 kN (모멘트팔 4 cm)'],
    ['동작 속도 / 출력 상한','각 관절 12 rad/s · 두 관절 합산 출력 상한 14.4 kW'],
    ['피로 / 에너지 공급 여력','120 s 동안 능력 유지 · 기계적 공급 여력 100 kJ · 이상 효율 100%'],
    ['봉 / 줄 지지부 하중 여력','봉 10 kN · 줄 20 kN 이상 가정 (재료 파괴 모델 생략)'],
    ['줄의 유연성','강성 200 kN/m · 감쇠 500 N·s/m · 압축력 0 N'],
    ['지면 / 기타 이상화','접선 마찰 μ=0 · 축 구동 토크 0 · 평면 강체 운동'],
]
tab=ax.table(cellText=table_rows,colLabels=['가정 항목','계산에 사용한 값'],cellLoc='left',
             colLoc='left',colWidths=[.27,.73],bbox=[0,0,1,1])
tab.auto_set_font_size(False);tab.set_fontsize(12)
for (row,col),cell in tab.get_celld().items():
    cell.set_edgecolor('#D8E1EA');cell.set_linewidth(.7)
    if row==0:
        cell.set_facecolor('#1864AB');cell.set_text_props(color='white',weight='bold',fontsize=13)
    else:
        cell.set_facecolor('#F0F5F9' if row%2 else '#FFFFFF')
        if col==0:cell.set_text_props(weight='bold',color='#24394C')
fig.suptitle('초인 가정을 힘·속도·출력의 크기로 표현',x=.06,ha='left',y=.95,fontsize=22,weight='bold')
fig.text(.06,.865,'가정한 능력의 한계와 실제 계산에서 사용한 최대값은 다름. 아래 수치는 인체 측정값이 아님.',fontsize=12,color='#526171')
fig.text(.06,.055,'상체 목표 진폭 60° · 동작 위상은 진폭에 따라 조절 · 줄 모델은 좌석 방향을 고정한 등가모델이며 실제 체인 운동의 모든 자유도는 생략.',fontsize=10,color='#526171')
save(fig,'swing-assumptions')

public_names={'ideal':'rod-ideal','air':'rod-with-resistance','rope_fine':'rope-ideal','rope_air':'rope-with-resistance'}
for source,dest in public_names.items():
    d=data[source]
    with (OUT/f'{dest}-data.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(d.dtype.names)
        w.writerows(tuple(float(row[k]) for k in d.dtype.names) for row in d)
public_summary={'definition':'First |theta|=180 degrees from rest; NOT a full 360-degree return',
    'main_cases':{public_names[n]:r for n,r in results.items()},
    'sensitivity_cases':{n:summary[n] for _,n in comparison if n not in ['ideal','air']},
    'less_torque':summary['less_torque'],'numerical_verification':verification}
(OUT/'swing-180-results.json').write_text(json.dumps(public_summary,ensure_ascii=False,indent=2),encoding='utf-8')

metrics=[]
for n in names:
    r=results[n];dd=data[n]
    metrics.append({'case':public_names[n],'reached_180':r['target_reached'],
        'first_arrival_s':r['time_s'] if r['target_reached'] else '',
        'simulated_s':r['simulation_end_time_s'],'max_abs_theta_deg':min(180,r['max_angle_deg']),
        'max_com_speed_m_s':r['max_com_speed_m_s'],'max_hip_torque_Nm':r['max_hip_torque_Nm'],
        'max_knee_torque_Nm':r['max_knee_torque_Nm'],'max_hip_speed_rad_s':r['max_hip_speed_rad_s'],
        'max_knee_speed_rad_s':r['max_knee_speed_rad_s'],'max_support_force_N':r['max_pivot_reaction_N'],
        'max_tension_N':r['max_rope_tension_N'],'max_abs_power_W':r['max_abs_power_W'],
        'net_actuator_work_J':r['work_J'],'positive_actuator_work_J':r['positive_work_J'],
        'dissipation_J':r['loss_J'],'contact_work_J':r['constraint_work_J'],
        'max_energy_residual_J':r['max_energy_residual_J'],
        'hip_min_deg':float(np.min(dd['hip_deg'])),'hip_max_deg':float(np.max(dd['hip_deg'])),
        'knee_min_deg':float(np.min(dd['knee_deg'])),'knee_max_deg':float(np.max(dd['knee_deg']))})
with (OUT/'swing-180-summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
print(json.dumps({'figures':4,'main_cases':[{k:r[k] for k in ['target_reached','time_s','max_angle_deg']} for r in results.values()],
                  'verification':verification},ensure_ascii=False,indent=2))
