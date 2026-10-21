"""Search YouTube (Creative Commons filter) per scene category and list candidate metadata."""
import json
import sys
import urllib.parse

import yt_dlp

QUERIES = {
    "indoor_static": ["apartment walkthrough 4k", "house tour walkthrough 4k no talking", "museum walking tour 4k"],
    "indoor_dynamic": ["shopping mall walking tour 4k", "indoor market walk 4k", "airport terminal walk 4k"],
    "outdoor_static": ["hiking trail pov 4k", "drone flight landscape 4k", "forest walk pov 4k"],
    "outdoor_dynamic": ["city walking tour 4k", "busy street walk 4k", "night market walk 4k"],
    "driving": ["dashcam city driving 4k", "driving pov downtown 4k", "highway drive dashcam 4k"],
    "embodied": ["robot arm manipulation first person", "gopro head mounted cooking pov", "humanoid robot household tasks"],
}
CC = "EgIwAQ%3D%3D"  # YouTube search filter: Creative Commons
out = {}
flat = yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True, "playlistend": 12})
full = yt_dlp.YoutubeDL({"quiet": True, "skip_download": True})
for cat, qs in QUERIES.items():
    seen, rows = set(), []
    for q in qs:
        url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(q)}&sp={CC}"
        try:
            res = flat.extract_info(url, download=False)
        except Exception as e:  # noqa: BLE001
            print("search fail", q, e, file=sys.stderr)
            continue
        for e in (res.get("entries") or [])[:8]:
            vid = e.get("id")
            if not vid or vid in seen:
                continue
            seen.add(vid)
            try:
                i = full.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
            except Exception as ex:  # noqa: BLE001
                continue
            rows.append({"id": vid, "title": i.get("title"), "channel": i.get("channel"), "license": i.get("license"),
                         "duration": i.get("duration"), "height": i.get("height"), "fps": i.get("fps"), "query": q})
    out[cat] = rows
    for r in rows:
        print(f"{cat:16s} {r['id']} {str(r['license'])[:28]:28s} {r['height']}p {r['fps']}fps {r['duration']}s | {str(r['title'])[:70]}", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
