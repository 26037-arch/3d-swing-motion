import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)


def simulate(duration=.1, short=1.8, step=.0002, mass=70., angle=20., save=False):
    g, long = 9.81, 2.
    t, theta, omega = 0., -math.radians(angle), 0.
    rows, events = [], []
    curve = None
    Lfixed = long
    total_work = 0.
    last_row = None

    def radial(time):
        if curve is None:
            return Lfixed, 0., 0.
        start, before, after = curve
        u = max(0., min(1., (time-start)/duration))
        s = 10*u**3 - 15*u**4 + 6*u**5
        ds = 30*u**2*(1-u)**2
        dds = 60*u*(1-u)*(1-2*u)
        return before+(after-before)*s, (after-before)*ds/duration, (after-before)*dds/duration**2

    def rate(time, a, w):
        L, dL, _ = radial(time)
        return w, -2*dL/L*w-g/L*math.sin(a)

    def rk(time, a, w, h):
        a1,w1=rate(time,a,w)
        a2,w2=rate(time+h/2,a+h*a1/2,w+h*w1/2)
        a3,w3=rate(time+h/2,a+h*a2/2,w+h*w2/2)
        a4,w4=rate(time+h,a+h*a3,w+h*w3)
        return a+h*(a1+2*a2+2*a3+a4)/6, w+h*(w1+2*w2+2*w3+w4)/6

    def record():
        nonlocal total_work, last_row
        L,dL,ddL=radial(t)
        force=mass*(ddL-L*omega**2-g*math.cos(theta))
        power=force*dL
        energy=.5*mass*(dL**2+(L*omega)**2)+mass*g*(long-L*math.cos(theta))
        row=dict(time_s=t,angle_deg=math.degrees(theta),omega_rad_s=omega,
                 length_m=L,radial_speed_m_s=dL,radial_acceleration_m_s2=ddL,
                 tangential_speed_m_s=L*omega,total_speed_m_s=math.hypot(dL,L*omega),
                 radial_force_N=force,power_W=power,energy_J=energy)
        if last_row is not None:
            total_work += .5*(last_row['power_W']+power)*(t-last_row['time_s'])
        row['net_work_J']=total_work
        row['energy_residual_J']=energy-(rows[0]['energy_J'] if rows else energy)-total_work
        rows.append(row)
        last_row=row

    def advance_until(event=None, finish=None):
        nonlocal t, theta, omega
        start=t
        while t-start < 60:
            h=step if finish is None else min(step,finish-t)
            if h <= 1e-12:
                return
            a,w=rk(t,theta,omega,h)
            hit = (event=='bottom' and theta*a <= 0 and abs(theta)>1e-10) or (event=='peak' and omega*w <= 0 and abs(omega)>1e-10)
            if event=='complete':
                hit=rotation_direction*(a-rotation_target)>=0
            if hit:
                lo,hi=0.,h
                for _ in range(35):
                    mid=(lo+hi)/2
                    am,wm=rk(t,theta,omega,mid)
                    crossed=(theta*am <= 0) if event=='bottom' else ((omega*wm <= 0) if event=='peak' else rotation_direction*(am-rotation_target)>=0)
                    if crossed: hi=mid
                    else: lo=mid
                h=(lo+hi)/2
                a,w=rk(t,theta,omega,h)
            t,theta,omega=t+h,a,w
            record()
            if hit: return
            if finish is not None and t >= finish-1e-12: return
        raise RuntimeError('Event not found')

    def change(after, kind):
        nonlocal curve,Lfixed
        before=Lfixed
        curve=(t,before,after)
        events.append(dict(time_s=t,kind=kind,angle_deg=math.degrees(theta),before_m=before,after_m=after))
        advance_until(finish=t+duration)
        curve=None
        Lfixed=after

    record()
    final_bottom=None
    for n in range(1,100):
        advance_until(event='bottom')
        bottom_time=t
        direction=1 if omega>0 else -1
        change(short,'shorten')
        energy_over_bottom=.5*mass*short**2*omega**2+mass*g*short*(1-math.cos(theta))
        if energy_over_bottom > 2*mass*g*short:
            final_bottom=bottom_time
            rotation_direction=direction
            rotation_target=direction*2*math.pi
            advance_until(event='complete')
            break
        advance_until(event='peak')
        events.append(dict(time_s=t,kind='peak',angle_deg=math.degrees(theta)))
        change(long,'lengthen')
    if final_bottom is None:
        raise RuntimeError('No rotation')
    maximum=lambda key:max(abs(r[key]) for r in rows)
    end_energy=rows[-1]['energy_J']
    r=dict(mass_kg=mass,long_length_m=long,short_length_m=short,initial_angle_deg=angle,
           change_duration_s=duration,step_s=step,pumping_time_s=final_bottom,
           first_rotation_time_s=t-final_bottom,total_time_s=t,
           contractions=n,extensions=n-1,max_radial_force_N=maximum('radial_force_N'),
           min_radial_force_N=min(x['radial_force_N'] for x in rows),
           max_outward_radial_force_N=max(x['radial_force_N'] for x in rows),
           max_abs_power_W=maximum('power_W'),
           max_radial_speed_m_s=maximum('radial_speed_m_s'),
           max_radial_acceleration_m_s2=maximum('radial_acceleration_m_s2'),
           max_tangential_speed_m_s=maximum('tangential_speed_m_s'),
           max_total_speed_m_s=maximum('total_speed_m_s'),
           max_abs_omega_rad_s=maximum('omega_rad_s'),
           peak_energy_residual_J=maximum('energy_residual_J'),
           final_energy_J=end_energy,net_work_J=total_work,
           top_speed_m_s=math.sqrt(max(0.,(short*omega)**2-4*g*short)),events=events)
    if save:
        with (OUT/'swing_motion.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (OUT/'swing_calculation.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    return r,rows


if __name__=='__main__':
    r,rows=simulate(save=True)
    print(json.dumps(r,ensure_ascii=False,indent=2))
    fine,_=simulate(step=.0001)
    comparisons=[]
    for short,duration in [(1.8,.05),(1.8,.2),(1.7,.1),(1.6,.1)]:
        x,_=simulate(short=short,duration=duration)
        comparisons.append(x)
    check={'refinement_total_time_difference_s':abs(fine['total_time_s']-r['total_time_s']),
           'refinement_loop_difference_s':abs(fine['first_rotation_time_s']-r['first_rotation_time_s']),
           'comparison_scenarios':comparisons}
    (OUT/'swing_verification.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(check,ensure_ascii=False,indent=2))
