import sys
from pathlib import Path
from PIL import Image, ImageDraw
SRC = Path("/data/home/z00533370/data/youtube_samples/clips")
ds = [d for d in sorted(SRC.iterdir()) if d.is_dir() and not d.name.startswith("_") and (d / "sheet.jpg").exists()]
ims = []
for d in ds:
    im = Image.open(d / "sheet.jpg"); im = im.resize((720, int(720 * im.height / im.width)))
    c = Image.new("RGB", (720, im.height + 22), "white"); c.paste(im, (0, 22)); ImageDraw.Draw(c).text((4, 4), d.name, fill="black"); ims.append(c)
W = 2
H = sum(max(ims[i].height for i in range(r, min(r + W, len(ims)))) for r in range(0, len(ims), W))
g = Image.new("RGB", (720 * W, H), "white"); y = 0
for r in range(0, len(ims), W):
    row = ims[r:r + W]
    for j, im in enumerate(row): g.paste(im, (720 * j, y))
    y += max(im.height for im in row)
g.save(sys.argv[1], quality=80); print(len(ims))
