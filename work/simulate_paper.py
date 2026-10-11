from pathlib import Path
import math
import json
import csv
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.optimize import brentq
from scipy.special import ellipk

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs'
OUT.mkdir(exist_ok=True)

def make_model(height=1.75,mass=70.,pivot_height=2.5,length=2.2,dt=.001,ground=True,rope=False):
    scale=height/1.58
    l1,l2,l3=np.array([.792,.393,.395])*scale
    m1,m2,m3=np.array([31.3,12.3,6.4])*mass/50
    a=(m2/2+m3)*l2/mass
    b=(m1+m2/2)*l2/mass
    floor='<geom name="ground" type="plane" size="10 10 .1" contype="1" conaffinity="1" friction="0 0 0"/>' if ground else ''
    contact='1' if ground else '0'
    xml=f'''<mujoco model="Hirata three-segment swing">
      <compiler angle="radian" inertiafromgeom="auto"/>
      <option timestep="{dt}" integrator="implicitfast" gravity="0 0 -9.81" iterations="50" tolerance="1e-12"><flag energy="enable"/></option>
      <default><joint axis="0 -1 0" damping="0" armature="0"/><geom contype="0" conaffinity="0" rgba=".5 .6 .7 1"/></default>
      <worldbody>{floor}
        <body name="swing" pos="0 0 {pivot_height}">
          <joint name="theta" type="hinge" limited="false"/>
          <geom name="rod" type="capsule" fromto="0 0 0 0 0 {-length}" size=".01" mass="1"/>
          <geom name="seat" type="box" pos="0 0 {-length}" size=".22 .3 .025" mass="3"/>
          <geom name="thigh" type="capsule" fromto="{-a} 0 {-length} {b} 0 {-length}" size=".045" mass="{m2}"/>
          <body name="torso" pos="{-a} 0 {-length}">
            <joint name="hip" type="hinge" limited="false"/>
            <inertial pos="0 0 {l1/2}" mass="{m1}" diaginertia="{m1*l1*l1/12} {m1*l1*l1/12} {.001*m1}"/>
            <geom type="capsule" fromto="0 0 0 0 0 {l1}" size=".07" mass="0"/>
          </body>
          <body name="lowerleg" pos="{b} 0 {-length}">
            <joint name="knee" type="hinge" limited="false"/>
            <inertial pos="0 0 {-l3/2}" mass="{m3}" diaginertia="{m3*l3*l3/12} {m3*l3*l3/12} {.001*m3}"/>
            <geom type="capsule" fromto="0 0 0 0 0 {-l3}" size=".025" mass="0"/>
            <geom name="foot" type="sphere" pos="0 0 {-l3}" size=".025" mass="0" contype="{contact}" conaffinity="{contact}" friction="0 0 0" solref=".004 1"/>
            <site name="foot_tip" pos="0 0 {-l3-.025}"/>
          </body>
        </body>
      </worldbody>
      <actuator><motor name="hip_motor" joint="hip" ctrllimited="true" ctrlrange="-800 800"/><motor name="knee_motor" joint="knee" ctrllimited="true" ctrlrange="-400 400"/></actuator>
    </mujoco>'''
    if rope:
        root=ET.fromstring(xml)
        swing=root.find('.//body[@name="swing"]')
        payload=ET.Element('body',{'name':'payload'})
        ET.SubElement(payload,'joint',{'name':'rope_radius','type':'slide','axis':'0 0 -1',
           'limited':'true','range':'.05 20','solreflimit':'.002 1','solimplimit':'.9999 .99999 .001'})
        for child in list(swing):
            if child.tag=='joint':
                continue
            swing.remove(child)
            if child.get('name')=='rod':
                continue
            if child.get('pos'):
                vals=list(map(float,child.get('pos').split()))
                vals[2]+=length
                child.set('pos',' '.join(map(str,vals)))
            if child.get('fromto'):
                vals=list(map(float,child.get('fromto').split()))
                vals[2]+=length;vals[5]+=length
                child.set('fromto',' '.join(map(str,vals)))
            payload.append(child)
        ET.SubElement(swing,'inertial',{'pos':'0 0 0','mass':'0.000001','diaginertia':'.000001 .000001 .000001'})
        swing.append(payload)
        xml=ET.tostring(root,encoding='unicode')
    model=mujoco.MjModel.from_xml_string(xml)
    foot_radius=.025
    R=math.hypot(length+l3,b)
    if pivot_height-foot_radius>R:
        raise ValueError('The foot cannot reach the floor in the specified initial pose')
    theta0=-math.atan2(b,length+l3)-math.acos((pivot_height-foot_radius)/R)
    pars=dict(height_m=height,human_mass_kg=mass,pivot_height_m=pivot_height,length_m=length,
              torso_length_m=l1,thigh_length_m=l2,shin_length_m=l3,
              torso_mass_kg=m1,thigh_mass_kg=m2,shin_mass_kg=m3,a_m=a,b_m=b,
              initial_theta_deg=math.degrees(theta0),foot_radius_m=foot_radius,
              suspension_mass_kg=0. if rope else 1.,seat_mass_kg=3.,ground=ground,timestep_s=dt,rope=rope)
    return model,theta0,pars,xml


def simulate(amp=60.,center=15.,leg_amp=40.,leg_center=25.,phase_slope=.5,drag=.0,dry=.0,
             seconds=240,dt=.001,save=False,ground=True,pivot_height=2.5,length=2.2,
             hip_max=800.,knee_max=400.,vmax=(12.,12.),height=1.75,mass=70.,foot_guard=True,rope=False):
    m,theta0,pars,xml=make_model(dt=dt,ground=ground,pivot_height=pivot_height,length=length,height=height,mass=mass,rope=rope)
    d=mujoco.MjData(m)
    state_indices=np.array([0,2,3]) if rope else np.array([0,1,2])
    act_indices=state_indices[1:]
    d.qpos[state_indices]=[theta0,0.,0.]
    if rope:d.qpos[1]=length
    mujoco.mj_forward(m,d)
    initial_energy=float(d.energy.sum())
    bodies=np.arange(1,m.nbody)
    masses=m.body_mass[bodies]
    com=d.subtree_com[1]
    gravity_torque=m.body_subtreemass[1]*9.81*np.linalg.norm(com-m.body_pos[1])
    fullM=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,fullM)
    natural=math.sqrt(gravity_torque/fullM[0,0])
    amplitude=abs(theta0)
    frequency=natural*math.pi/(2*ellipk(math.sin(amplitude/2)**2))
    amp,center,leg_amp,leg_center=map(math.radians,[amp,center,leg_amp,leg_center])
    work=loss=work_contact=0.
    max_theta=abs(theta0)
    rows=[];peaks=[]
    last_v=float(d.qvel[0])
    max_hip_torque=max_knee_torque=0.
    max_hip_speed=max_knee_speed=max_com_speed=max_reaction=0.
    lowest_foot=100.
    max_power=0.
    min_seat_radius=length
    max_seat_radius=length
    prev_energy=initial_energy
    first_slack=None
    rope_slack_time=None
    rope_stiffness=200000.
    rope_damping=500.
    max_rope_tension=0.
    clearance_corrections=0
    jacp=np.zeros((3,m.nv));jacr=np.zeros_like(jacp)
    jacpd=np.zeros_like(jacp);jacrd=np.zeros_like(jacp)
    peak_work_positive=0.
    failure_reason=''
    event_time=None
    for step in range(round(seconds/dt)):
        mujoco.mj_step1(m,d)
        th,phi,psi=d.qpos[state_indices]
        current_length=d.qpos[1] if rope else length
        min_seat_radius=min(min_seat_radius,float(current_length))
        max_seat_radius=max(max_seat_radius,float(current_length))
        v=d.qvel.copy()
        if step>10 and last_v*v[0]<0 and abs(v[0])<.1 and (not peaks or d.time-peaks[-1][0]>.7):
            peaks.append((float(d.time),float(th)))
            amplitude=max(math.radians(5),abs(th))
            frequency=natural*math.pi/(2*ellipk(math.sin(min(amplitude,math.radians(175))/2)**2))
        last_v=float(v[0])
        phase=math.atan2(v[0]/frequency,-th)
        alpha=max(.1,math.pi/2-phase_slope*amplitude)
        target=np.array([center+amp*math.cos(phase-alpha),leg_center+leg_amp*math.cos(phase-alpha)])
        if ground and foot_guard:
            knee_height=pivot_height-current_length*math.cos(th)+pars['b_m']*math.sin(th)
            ratio=(knee_height-pars['foot_radius_m']-.04)/pars['shin_length_m']
            if ratio<1:
                needed=math.acos(max(-1.,ratio))-th
                if needed>target[1]:
                    target[1]=min(math.radians(100),needed)
                    clearance_corrections+=1
        # The initial velocities are zero. Finite torques move the joints to
        # the paper-inspired targets; qpos is never overwritten during motion.
        blend=min(1.,d.time/.4)
        target*=blend
        error=target-d.qpos[act_indices]
        gains=np.array([1400.,550.])
        damping=np.array([180.,55.])
        tau=np.clip(gains*error-damping*v[act_indices],[-hip_max,-knee_max],[hip_max,knee_max])
        for j,limit in enumerate(vmax):
            if tau[j]*v[act_indices[j]]>0:
                tau[j]*=max(0.,min(1.,(limit-abs(v[act_indices[j]]))/(.2*limit)))
        d.ctrl[:]=tau
        friction=-drag*length**3*v[0]*abs(v[0])-dry*math.tanh(v[0]/.02)
        d.qfrc_applied[0]=friction
        stretch=max(0.,current_length-length) if rope else 0.
        tension=max(0.,rope_stiffness*stretch+rope_damping*v[1]) if rope and stretch>0 else 0.
        if rope:
            d.qfrc_applied[1]=-tension
            max_rope_tension=max(max_rope_tension,tension)
            if rope_slack_time is None and current_length<length-.001:
                rope_slack_time=float(d.time)
        old_theta=float(d.qpos[0]);old_time=float(d.time)
        mujoco.mj_step2(m,d)
        # Midpoint-like work estimates are refined with the timestep check.
        vmid=(v+d.qvel)/2
        power=float(tau@vmid[act_indices])
        work+=power*dt
        peak_work_positive+=max(0.,power)*dt
        loss-=friction*vmid[0]*dt
        if rope:
            loss+=(tension-rope_stiffness*stretch)*vmid[1]*dt
        work_contact+=float(d.qfrc_constraint@vmid)*dt
        max_hip_torque=max(max_hip_torque,abs(tau[0]))
        max_knee_torque=max(max_knee_torque,abs(tau[1]))
        max_hip_speed=max(max_hip_speed,abs(d.qvel[act_indices[0]]))
        max_knee_speed=max(max_knee_speed,abs(d.qvel[act_indices[1]]))
        max_power=max(max_power,abs(power))
        max_theta=max(max_theta,abs(d.qpos[0]))
        if step%10==0 or abs(d.qpos[0])>=math.pi:
            mujoco.mj_forward(m,d)
            mujoco.mj_subtreeVel(m,d)
            mujoco.mj_rnePostConstraint(m,d)
            # Generalized force at an unactuated pivot is zero. Its spatial
            # reaction comes from complete inverse Newton-Euler dynamics.
            reaction=np.linalg.norm(d.cfrc_int[1,3:])
            radial_axis=np.array([-math.sin(d.qpos[0]),0.,math.cos(d.qpos[0])])
            signed_radial=float(d.cfrc_int[1,3:]@radial_axis)
            if first_slack is None and signed_radial < -1.:
                first_slack=dict(time_s=float(d.time),theta_deg=float(np.degrees(d.qpos[0])),radial_reaction_N=signed_radial)
            max_reaction=max(max_reaction,reaction)
            com=d.subtree_com[1].copy()
            velocity=d.subtree_linvel[1].copy()
            max_com_speed=max(max_com_speed,np.linalg.norm(velocity))
            ell=np.linalg.norm(com-m.body_pos[1])
            # A sphere's lowest point is vertically below its center, regardless
            # of the shin angle. The local foot-tip site is not floor clearance.
            feet=float(d.geom_xpos[m.geom('foot').id,2]-pars['foot_radius_m'])
            lowest_foot=min(lowest_foot,feet)
            spring_energy=.5*rope_stiffness*max(0.,float(d.qpos[1])-length)**2 if rope else 0.
            E=float(d.energy.sum())+spring_energy
            rows.append(dict(time_s=float(d.time),theta_deg=float(np.degrees(d.qpos[0])),
              omega_rad_s=float(d.qvel[0]),hip_deg=float(np.degrees(d.qpos[act_indices[0]])),knee_deg=float(np.degrees(d.qpos[act_indices[1]])),
              hip_speed_rad_s=float(d.qvel[act_indices[0]]),knee_speed_rad_s=float(d.qvel[act_indices[1]]),
              com_length_m=float(ell),com_speed_m_s=float(np.linalg.norm(velocity)),
              energy_J=E,kinetic_J=float(d.energy[1]),potential_J=float(d.energy[0]),
              work_J=work,loss_J=loss,constraint_work_J=work_contact,
              energy_residual_J=E-initial_energy-work+loss-work_contact,
              hip_torque_Nm=float(tau[0]),knee_torque_Nm=float(tau[1]),
              power_W=power,pivot_reaction_N=float(reaction),signed_radial_N=signed_radial,foot_height_m=feet,
              seat_radius_m=float(current_length),rope_tension_N=float(tension),spring_energy_J=spring_energy))
        if abs(d.qpos[0])>=math.pi:
            event_time=old_time+dt*(math.pi-abs(old_theta))/(abs(float(d.qpos[0]))-abs(old_theta))
            break
        if np.any(np.abs(d.qvel[act_indices])>np.array(vmax)*1.001):
            failure_reason='joint_speed_limit_exceeded'
            break
        if not np.isfinite(d.qpos).all(): raise RuntimeError('Nonfinite state')
    result=dict(parameters=pars,target_reached=bool(abs(d.qpos[0])>=math.pi),time_s=event_time if event_time is not None else float(d.time),
        failure_reason=failure_reason,simulation_end_time_s=float(d.time),
        max_angle_deg=math.degrees(max_theta),max_hip_torque_Nm=max_hip_torque,
        max_knee_torque_Nm=max_knee_torque,max_hip_speed_rad_s=max_hip_speed,
        max_knee_speed_rad_s=max_knee_speed,max_com_speed_m_s=float(max_com_speed),
        max_pivot_reaction_N=float(max_reaction),max_abs_power_W=max_power,
        lowest_foot_m=lowest_foot,work_J=work,positive_work_J=peak_work_positive,
        loss_J=loss,constraint_work_J=work_contact,
        max_energy_residual_J=max(abs(r['energy_residual_J']) for r in rows),
        foot_clearance_controller_fraction=clearance_corrections/(step+1),
        rope_first_slack_s=rope_slack_time,max_rope_tension_N=max_rope_tension,
        rope_stiffness_N_m=rope_stiffness if rope else None,rope_damping_Ns_m=rope_damping if rope else None,
        min_seat_radius_m=min_seat_radius,max_seat_radius_m=max_seat_radius,
        first_compression=first_slack,peaks=peaks,
        controller=dict(torso_amplitude_deg=math.degrees(amp),torso_center_deg=math.degrees(center),
          leg_amplitude_deg=math.degrees(leg_amp),leg_center_deg=math.degrees(leg_center),
          phase_slope=phase_slope,drag_k=drag,dry_friction_Nm=dry,hip_max_Nm=hip_max,knee_max_Nm=knee_max,vmax=list(vmax)))
    if save:
        with (OUT/'paper_swing_motion.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (OUT/'paper_swing_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        (OUT/'paper_swing_model.xml').write_text(xml,encoding='utf-8')
    return result,rows

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--amp',type=float,default=60)
    p.add_argument('--drag',type=float,default=0)
    p.add_argument('--phase',type=float,default=.5)
    p.add_argument('--leg-center',type=float,default=25)
    p.add_argument('--leg-amp',type=float,default=40)
    p.add_argument('--dry',type=float,default=0)
    p.add_argument('--seconds',type=float,default=180)
    p.add_argument('--dt',type=float,default=.001)
    p.add_argument('--save',action='store_true')
    p.add_argument('--rope',action='store_true')
    x=p.parse_args()
    r,_=simulate(amp=x.amp,leg_center=x.leg_center,leg_amp=x.leg_amp,drag=x.drag,dry=x.dry,phase_slope=x.phase,seconds=x.seconds,dt=x.dt,save=x.save,rope=x.rope)
    print(json.dumps({k:v for k,v in r.items() if k!='peaks'},indent=2))
    print('Last peaks:',r['peaks'][-10:])
