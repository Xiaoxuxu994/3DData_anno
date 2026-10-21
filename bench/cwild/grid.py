"""Grid of sheet.jpg for clips matching a substring: python grid.py OUT.jpg SUBSTR [SUBSTR...]"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw
C = Path("/data/home/z00533370/data/eval3d/C_wild/clips")
ds = [d for d in sorted(C.iterdir()) if (d / "sheet.jpg").exists() and any(s in d.name for s in sys.argv[2:])]
ims = []
for d in ds:
    im = Image.open(d / "sheet.jpg"); im = im.resize((660, int(660 * im.height / im.width)))
    c = Image.new("RGB", (660, im.height + 20), "white"); c.paste(im, (0, 20)); ImageDraw.Draw(c).text((4, 3), d.name, fill="black"); ims.append(c)
cols = 2
rows = [ims[i:i + cols] for i in range(0, len(ims), cols)]
H = sum(max(x.height for x in r) for r in rows)
g = Image.new("RGB", (660 * cols, H), "white"); y = 0
for r in rows:
    for j, im in enumerate(r): g.paste(im, (660 * j, y))
    y += max(x.height for x in r)
g.save(sys.argv[1], quality=75); print(len(ims))
