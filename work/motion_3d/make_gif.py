from pathlib import Path
import subprocess
import json
import argparse
import imageio_ffmpeg
from PIL import Image

root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--name',default='swing-3d-motion')
args=parser.parse_args()
source=root/'outputs'/f'{args.name}.mp4'
target=root/'outputs'/f'{args.name}.gif'
palette=Path(__file__).resolve().parent/f'{args.name}-palette.png'
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
common=[ffmpeg,'-hide_banner','-loglevel','error','-y']
scale='fps=15,scale=960:-1:flags=lanczos'
subprocess.run(common+['-i',str(source),'-vf',scale+',palettegen=stats_mode=diff',
                      '-frames:v','1',str(palette)],check=True)
subprocess.run(common+['-i',str(source),'-i',str(palette),'-filter_complex',
                      f'[0:v]{scale}[video];[video][1:v]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle',
                      '-loop','0',str(target)],check=True)
with Image.open(target) as im:
    width,height=im.size
    total=0
    frames=im.n_frames
    loop=im.info.get('loop')
    for i in range(frames):
        im.seek(i);im.load();total+=im.info.get('duration',0)
    im.seek(0);im.convert('RGB').save(palette.parent/f'{args.name}-first.png')
    im.seek(frames-1);im.convert('RGB').save(palette.parent/f'{args.name}-last.png')
assert (width,height)==(960,540)
assert frames>350 and 28.3<total/1000<28.9 and loop==0
report={'file':str(target),'size_px':[width,height],'frames':frames,
        'duration_s':total/1000,'loop':loop,'file_bytes':target.stat().st_size,
        'source_arrival_s':27.113402542984215,
        'note':'Source timing retained; final pose held about 1.5 s; GIF loops.'}
(palette.parent/f'{args.name}-gif-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
