from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
work=ROOT/'work'/'motion_3d'
dest=Path(r'C:\Users\junwoo\.codex\visualizations\2026\10\10\01a124ee-13f9-76c0-8380-0e112c1382ae\swing-motion-3d.html')
dest.parent.mkdir(exist_ok=True,parents=True)
fragment=(work/'swing-motion-fragment.html').read_text(encoding='utf-8')
license_text=(work/'three-license.txt').read_text(encoding='utf-8')
library=(work/'three.min.js').read_text(encoding='utf-8')
data=(work/'motion-data.json').read_text(encoding='utf-8')
fragment=fragment.replace('__THREE_LIBRARY__',library.replace('</script','<\\/script')).replace('__MOTION_DATA__',data)
fragment='<!-- Three.js license\n'+license_text+'\n-->\n'+fragment
assert len(fragment.encode('utf-8'))<1000000
assert '__MOTION_DATA__' not in fragment
dest.write_text(fragment,encoding='utf-8')
sys.path.insert(0,r'C:\Users\junwoo\.codex\plugins\cache\openai-bundled\visualize\1.0.47\skills\visualize\scripts')
from render import render
(work/'preview.html').write_text(render(dest,title='그네 몸동작 3D'),encoding='utf-8')
print(f'Fragment: {dest}, {dest.stat().st_size} bytes',flush=True)
