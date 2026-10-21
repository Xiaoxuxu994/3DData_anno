import sys
from pathlib import Path
from PIL import Image, ImageDraw
SRC = Path("/data/home/z00533370/data/youtube_samples/clips")
names = sys.argv[2:]
ims = []
for n in names:
    im = Image.open(SRC / n / "sheet.jpg"); im = im.resize((720, int(720 * im.height / im.width)))
    c = Image.new("RGB", (720, im.height + 22), "white"); c.paste(im, (0, 22)); ImageDraw.Draw(c).text((4, 4), n, fill="black"); ims.append(c)
g = Image.new("RGB", (1440, sum(i.height for i in ims[::2])), "white"); y = 0
for r in range(0, len(ims), 2):
    for j, im in enumerate(ims[r:r + 2]): g.paste(im, (720 * j, y))
    y += ims[r].height
g.save(sys.argv[1], quality=80)
