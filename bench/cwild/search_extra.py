import json, sys, urllib.parse, time
import yt_dlp
Q = {"night_drive": ["night driving dashcam 4k city", "night drive pov 4k"],
     "zoom": ["zoom in stock footage 4k free", "camera zoom city stock footage"],
     "cinematic": ["cinematic travel video 4k", "travel montage cinematic 4k"],
     "indoor_walk": ["apartment walkthrough tour 4k gimbal", "hotel room tour walkthrough 4k"]}
CC = "EgIwAQ%3D%3D"
flat = yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True, "playlistend": 10})
full = yt_dlp.YoutubeDL({"quiet": True, "skip_download": True})
out = {}
for k, qs in Q.items():
    rows, seen = [], set()
    for q in qs:
        res = flat.extract_info(f"https://www.youtube.com/results?search_query={urllib.parse.quote(q)}&sp={CC}", download=False)
        for e in (res.get("entries") or [])[:6]:
            if not e.get("id") or e["id"] in seen:
                continue
            seen.add(e["id"])
            try:
                i = full.extract_info(f"https://www.youtube.com/watch?v={e['id']}", download=False)
            except Exception:
                continue
            time.sleep(1.5)
            rows.append({"id": e["id"], "title": i.get("title"), "lic": i.get("license"), "h": i.get("height"), "dur": i.get("duration")})
            print(k, e["id"], str(i.get("license"))[:20], i.get("height"), i.get("duration"), (i.get("title") or "")[:60], flush=True)
    out[k] = rows
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
