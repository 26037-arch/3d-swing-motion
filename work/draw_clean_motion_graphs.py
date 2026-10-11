from pathlib import Path
import json
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs'
data=np.genfromtxt(OUT/'rod-ideal-data.csv',delimiter=',',names=True,encoding='utf-8-sig')
result=json.loads((OUT/'swing-180-results.json').read_text(encoding='utf-8'))['main_cases']['rod-ideal']
end=result['time_s']
initial_energy=float(data['energy_J'][0]-data['energy_residual_J'][0]-data['work_J'][0]
                     +data['loss_J'][0]-data['constraint_work_J'][0])
inside=data['time_s']<end
t=np.r_[0.,data['time_s'][inside],end]
theta=np.r_[result['parameters']['initial_theta_deg'],data['theta_deg'][inside],180.]
energy=np.r_[initial_energy,data['energy_J'][inside],np.interp(end,data['time_s'],data['energy_J'])]
assert np.max(np.abs(data['energy_J']-data['kinetic_J']-data['potential_J']))<1e-7

font_manager.fontManager.addfont(r'C:\Windows\Fonts\malgun.ttf')
plt.rcParams.update({'font.family':'Malgun Gothic','font.size':18,'axes.labelsize':20,
    'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
    'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
    'svg.fonttype':'path','axes.linewidth':1.,'xtick.labelsize':16,'ytick.labelsize':16})

def style(ax,kind):
    ax.set_xlim(0,28.3)
    ax.set_xticks([0,5,10,15,20,25,end],['0','5','10','15','20','25','27.1'])
    ax.set_xlabel('시간 t (s)',labelpad=12)
    ax.tick_params(direction='out',length=5,pad=7)
    ax.grid(True,color='#D9DEE4',linewidth=.7)
    ax.set_axisbelow(True)
    if kind=='angle':
        ax.set_ylim(-160,200);ax.set_yticks([-135,-90,-45,0,45,90,135,180])
        ax.set_ylabel('각도 θ (°)',labelpad=14)
        ax.axhline(0,color='#9AA7B3',linewidth=.9)
        ax.plot(t,theta,color='#1864AB',linewidth=2.2)
        ax.scatter([end],[180],color='#1864AB',s=35,zorder=3)
    else:
        ax.set_ylim(0,3600);ax.set_yticks([0,500,1000,1500,2000,2500,3000,3500])
        ax.set_ylabel('역학적 에너지 E (J)',labelpad=14)
        ax.plot(t,energy,color='#1864AB',linewidth=2.2)
        ax.scatter([end],[energy[-1]],color='#1864AB',s=35,zorder=3)

files=[]
def save(fig,name):
    for ext in ['png','svg']:
        path=OUT/f'{name}.{ext}'
        fig.savefig(path,dpi=220)
        files.append(path)
    plt.close(fig)

for kind,name in [('angle','swing-angle-clean'),('energy','swing-energy-clean')]:
    fig,ax=plt.subplots(figsize=(12.8,7.2))
    fig.subplots_adjust(left=.14,right=.96,bottom=.17,top=.96)
    style(ax,kind);save(fig,name)

with plt.rc_context({'font.size':14,'axes.labelsize':16,'xtick.labelsize':13,'ytick.labelsize':13}):
    fig,axes=plt.subplots(2,1,figsize=(12.8,8.6))
    fig.subplots_adjust(left=.15,right=.97,bottom=.10,top=.97,hspace=.36)
    style(axes[0],'angle');style(axes[1],'energy');save(fig,'swing-angle-energy-clean')

with zipfile.ZipFile(OUT/'swing-clean-graphs.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
    for path in files:archive.write(path,path.name)
print(json.dumps({'graphs':[p.name for p in files],'time_s':end,
                  'initial_energy_J':initial_energy,'final_energy_J':float(energy[-1])},ensure_ascii=False,indent=2))
