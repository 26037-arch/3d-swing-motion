from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
import json
import re

base=Path(__file__).resolve().parent/'ppt_source'
source=base/'source.pptx'
base.mkdir(exist_ok=True)
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main','r':'http://schemas.openxmlformats.org/package/2006/relationships'}
with zipfile.ZipFile(source) as z:
    slides=sorted([x for x in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',x)],key=lambda x:int(re.search(r'(\d+)\.xml',x).group(1)))
    out=[]
    for name in slides:
        root=ET.fromstring(z.read(name))
        text=[''.join(p.itertext()) for p in root.findall('.//a:t',ns)]
        relationships='ppt/slides/_rels/'+Path(name).name+'.rels'
        rels=[]
        if relationships in z.namelist():
            rels=[e.attrib for e in ET.fromstring(z.read(relationships))]
        out.append({'slide':int(re.search(r'(\d+)\.xml',name).group(1)),'text':text,'relationships':rels})
    for name in z.namelist():
        if name.startswith('ppt/media/') and not name.endswith('/'):
            (base/Path(name).name).write_bytes(z.read(name))
    (base/'contents.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
