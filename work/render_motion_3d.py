from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from simulate_paper import make_model
from arm_kinematics import place_visual_arms, UPPER_ARM, FOREARM_TO_GRIP, GRIP_HEIGHT
import numpy as np
import mujoco
import xml.etree.ElementTree as ET
import json
import argparse
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

ROOT=Path(__file__).resolve().parent.parent
WORK=ROOT/'work'/'motion_3d'
OUT=ROOT/'outputs'
WORK.mkdir(exist_ok=True)

def add_geom(body,**kwargs):
    attrs={'mass':'0','contype':'0','conaffinity':'0'}
    attrs.update({k:str(v) for k,v in kwargs.items()})
    return ET.SubElement(body,'geom',attrs)

def create_visual_model():
    base,_,pars,xml=make_model()
    root=ET.fromstring(xml)
    visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',{'offwidth':'1280','offheight':'720'})
    ET.SubElement(visual,'quality',{'offsamples':'4','shadowsize':'2048'})
    ET.SubElement(visual,'headlight',{'ambient':'.65 .65 .65','diffuse':'.7 .7 .7','specular':'.1 .1 .1'})
    ET.SubElement(visual,'map',{'znear':'.02','zfar':'30','fogstart':'20','fogend':'30'})
    ET.SubElement(visual,'rgba',{'haze':'1 1 1 1'})
    asset=ET.SubElement(root,'asset')
    ET.SubElement(asset,'texture',{'name':'sky','type':'skybox','builtin':'flat','rgb1':'.96 .975 .99','rgb2':'.96 .975 .99','width':'256','height':'256'})
    w=root.find('worldbody')
    ground=w.find("geom[@name='ground']")
    ground.set('rgba','.88 .91 .94 1');ground.set('size','20 20 .1')
    ET.SubElement(w,'light',{'pos':'-3 -4 8','dir':'.2 .3 -1','diffuse':'.8 .8 .8','specular':'.2 .2 .2','castshadow':'true'})
    ET.SubElement(w,'light',{'pos':'3 3 6','dir':'-.3 -.3 -1','diffuse':'.4 .4 .4','castshadow':'false'})
    # Preserve the original masses and inertia tensors exactly. All added
    # head/arm/twin-leg/frame geometry is visual only and introduces no DOF.
    for body in root.findall('.//body'):
        i=base.body(body.get('name')).id
        inertial=body.find('inertial')
        if inertial is None:inertial=ET.SubElement(body,'inertial')
        for key,arr in [('pos',base.body_ipos[i]),('quat',base.body_iquat[i]),('diaginertia',base.body_inertia[i])]:
            inertial.set(key,' '.join(map(str,arr)))
        inertial.set('mass',str(base.body_mass[i]))
        for geom in body.findall('geom'):geom.set('mass','0')
    swing=root.find(".//body[@name='swing']")
    swing.find("geom[@name='rod']").set('rgba','0 0 0 0')
    swing.find("geom[@name='thigh']").set('rgba','0 0 0 0')
    swing.find("geom[@name='seat']").set('rgba','.18 .25 .32 1')
    for y in [-.24,.24]:
        add_geom(swing,type='capsule',fromto=f'0 {y} 0 0 {y} -2.2',size='.018',rgba='.38 .44 .51 1')
    a,b,l1,l3=pars['a_m'],pars['b_m'],pars['torso_length_m'],pars['shin_length_m']
    for y in [-.12,.12]:
        add_geom(swing,type='capsule',fromto=f'{-a} {y} -2.2 {b} {y} -2.2',size='.06',rgba='.87 .39 .16 1')
    torso=root.find(".//body[@name='torso']")
    torso.find('geom').set('rgba','0 0 0 0')
    add_geom(torso,type='capsule',fromto=f'0 0 .03 0 0 {l1*.63}',size='.12',rgba='.13 .43 .72 1')
    add_geom(torso,type='sphere',pos=f'0 0 {l1*.84}',size=str(l1*.16),rgba='.78 .68 .58 1')
    add_geom(torso,type='capsule',fromto=f'0 -.2 {l1*.58} 0 .2 {l1*.58}',size='.04',rgba='.13 .43 .72 1')
    add_geom(swing,name='waist_grip_bar',type='capsule',fromto=f'0 -.31 {-2.2+GRIP_HEIGHT} 0 .31 {-2.2+GRIP_HEIGHT}',size='.018',rgba='.25 .32 .4 1')
    for label in ['left','right']:
        add_geom(w,name=f'arm_{label}_upper',type='capsule',pos='0 0 1',size=f'.038 {UPPER_ARM/2}',rgba='.13 .43 .72 1')
        add_geom(w,name=f'arm_{label}_fore',type='capsule',pos='0 0 1',size=f'.032 {FOREARM_TO_GRIP/2}',rgba='.78 .68 .58 1')
        add_geom(w,name=f'arm_{label}_elbow',type='sphere',pos='0 0 1',size='.041',rgba='.78 .68 .58 1')
        add_geom(w,name=f'arm_{label}_hand',type='sphere',pos='0 0 1',size='.035',rgba='.78 .68 .58 1')
    leg=root.find(".//body[@name='lowerleg']")
    for geom in leg.findall('geom'):geom.set('rgba','0 0 0 0')
    for y in [-.12,.12]:
        add_geom(leg,type='capsule',fromto=f'0 {y} 0 0 {y} {-l3}',size='.05',rgba='.87 .39 .16 1')
        add_geom(leg,type='box',pos=f'.055 {y} {-l3}',size='.085 .06 .028',rgba='.16 .22 .29 1')
    for y in [-1.,1.]:
        for x in [-1.35,1.35]:
            add_geom(w,type='capsule',fromto=f'{x} {y} 0 0 {y} 2.5',size='.045',rgba='.53 .6 .66 1',group='4')
    add_geom(w,type='capsule',fromto='0 -1.14 2.5 0 1.14 2.5',size='.055',rgba='.36 .43 .5 1',group='4')
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    assert np.allclose(model.body_mass,base.body_mass)
    assert np.allclose(model.body_inertia,base.body_inertia)
    assert model.nq==3
    return model,pars

def screen_signed_areas(poses,scene,bend_rule='clockwise'):
    """Check elbow order after the actual perspective camera transform."""
    areas=[]
    for camera in scene.camera:
        position=np.asarray(camera.pos,dtype=float)
        forward=np.asarray(camera.forward,dtype=float)
        up=np.asarray(camera.up,dtype=float)
        right=np.cross(forward,up)
        for shoulder,elbow,hand,angle in poses:
            points=[]
            for point in [hand,elbow,shoulder]:
                relative=point-position
                depth=np.dot(relative,forward)
                assert depth>0
                points.append(np.array([np.dot(relative,right),np.dot(relative,up)])/depth)
            first,second=points[1]-points[0],points[2]-points[0]
            areas.append(.5*(first[0]*second[1]-first[1]*second[0]))
    sign=-1 if bend_rule=='clockwise' else 1
    if min(sign*area for area in areas)<=1e-8:
        raise ValueError(f'Arm order is not {bend_rule} in the rendered camera: {areas}')
    return areas

def render_video(preview_only=False,output_name='swing-3d-clockwise-motion',bend_rule='clockwise'):
    summary=json.loads((OUT/'swing-180-results.json').read_text(encoding='utf-8'))
    result=summary['main_cases']['rod-ideal'];end=result['time_s']
    data=np.genfromtxt(OUT/'rod-ideal-data.csv',delimiter=',',names=True,encoding='utf-8-sig')
    model,pars=create_visual_model();state=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,height=720,width=1280)
    arm_renderer=mujoco.Renderer(model,height=264,width=352)
    camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,2.55];camera.distance=9.3
    camera.azimuth=-75;camera.elevation=-9
    arm_camera=mujoco.MjvCamera();arm_camera.distance=2.5
    arm_camera.azimuth=-75;arm_camera.elevation=-6
    main_option=mujoco.MjvOption();main_option.geomgroup[4]=1
    arm_option=mujoco.MjvOption();arm_option.geomgroup[4]=0
    screen_areas=[]
    direction_label='시계 방향' if bend_rule=='clockwise' else '반시계 방향'
    font=ImageFont.truetype(r'C:\Windows\Fonts\malgun.ttf',27)
    small=ImageFont.truetype(r'C:\Windows\Fonts\malgun.ttf',20)
    bold=ImageFont.truetype(r'C:\Windows\Fonts\malgunbd.ttf',30)
    def frame(t):
        initial=[pars['initial_theta_deg'],0,0]
        vals=[np.interp(min(t,end),np.r_[0,data['time_s']],np.r_[value,data[name]]) for name,value in zip(['theta_deg','hip_deg','knee_deg'],initial)]
        if t>=end:vals[0]=180
        state.qpos[:]=np.deg2rad(vals)
        poses=place_visual_arms(model,*state.qpos[:2],pars['length_m'],pars,bend_rule)
        mujoco.mj_forward(model,state)
        renderer.update_scene(state,camera=camera,scene_option=main_option)
        screen_areas.extend(screen_signed_areas(poses,renderer.scene,bend_rule))
        pixels=renderer.render()
        im=Image.fromarray(pixels);draw=ImageDraw.Draw(im)
        arm_camera.lookat[:]=np.mean([pose[index] for pose in poses for index in [0,2]],axis=0)
        arm_renderer.update_scene(state,camera=arm_camera,scene_option=arm_option)
        screen_areas.extend(screen_signed_areas(poses,arm_renderer.scene,bend_rule))
        closeup=Image.fromarray(arm_renderer.render())
        draw.rectangle([906,114,1266,386],fill=(248,250,252))
        im.paste(closeup,(910,118))
        draw.text((924,127),f'손 → 팔꿈치 → 어깨 · {direction_label}',font=small,fill=(33,53,74))
        draw.rectangle([0,0,1280,102],fill=(248,250,252))
        draw.text((36,15),'정지 출발 → θ=180° | 단단한 봉, 공기·축 저항 없음',font=bold,fill=(33,53,74))
        v=np.interp(min(t,end),np.r_[0,data['time_s']],np.r_[0,data['com_speed_m_s']])
        draw.text((36,59),f'시간 {min(t,end):5.2f} s     θ {vals[0]:6.1f}°     질량중심 속력 {v:4.2f} m/s',font=font,fill=(40,65,91))
        draw.rectangle([0,666,1280,720],fill=(248,250,252))
        draw.text((36,681),f'손은 허리 높이의 봉에 고정 · 손 → 팔꿈치 → 어깨의 {direction_label} 유지',font=small,fill=(73,90,107))
        if t>=end:
            draw.rounded_rectangle([922,406,1245,471],radius=12,fill=(24,100,171))
            draw.text((940,421),f'θ=180° · {end:.1f}초',font=bold,fill=(255,255,255))
        return np.asarray(im)
    for t,label in [(0,'start'),(end*.65,'swing'),(end,'top')]:
        Image.fromarray(frame(t)).save(WORK/f'{output_name}-preview-{label}.png')
    if preview_only:
        renderer.close();arm_renderer.close();print('3D previews complete',flush=True);return
    fps=30;times=np.r_[np.arange(0,end,1/fps),np.full(45,end)]
    path=OUT/f'{output_name}.mp4'
    with imageio.get_writer(path,fps=fps,codec='libx264',quality=8,macro_block_size=1,pixelformat='yuv420p',ffmpeg_log_level='error') as writer:
        for i,t in enumerate(times):
            writer.append_data(frame(t))
            if i%150==0:print(f'Frames {i}/{len(times)}',flush=True)
    renderer.close();arm_renderer.close()
    Image.open(WORK/f'{output_name}-preview-top.png').save(OUT/f'{output_name}-preview.png')
    report={'file':str(path),'frames':len(times),'duration_s':len(times)/fps,'arrival_s':end,
            'bytes':path.stat().st_size,'screen_signed_area_range':[min(screen_areas),max(screen_areas)],
            'checked_arm_camera_poses':len(screen_areas),'bend_rule':bend_rule,'screen_order':'hand -> elbow -> shoulder'}
    (WORK/f'{output_name}-screen-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--name',default='swing-3d-clockwise-motion')
    parser.add_argument('--bend-rule',choices=['clockwise','counterclockwise'],default='clockwise')
    args=parser.parse_args()
    render_video(args.preview,args.name,args.bend_rule)
