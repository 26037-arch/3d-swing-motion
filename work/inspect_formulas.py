from PIL import Image, ImageDraw
from pathlib import Path
p=Path(__file__).resolve().parent/'ppt_source'
names=['image5.png','image9.png','image10.png','image11.png','image12.png','image13.png','image15.png']
y=0
canvas=Image.new('RGB',(1800,2400),'white')
d=ImageDraw.Draw(canvas)
for n in names:
    im=Image.open(p/n).convert('RGBA')
    scale=min(1.,1700/im.width)
    im=im.resize((int(im.width*scale),int(im.height*scale)))
    d.text((20,y+5),n,fill='black')
    canvas.paste(im,(20,y+25),im)
    y+=im.height+50
canvas.crop((0,0,1800,y)).save(p/'formulas.png')
