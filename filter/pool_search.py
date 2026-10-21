"""Search YouTube (Creative Commons filter) for the filter candidate pool.

Usage: python pool_search.py OUT_JSON
Writes a list of candidate videos with metadata (license, height, duration, channel, query, category hint).
Prefilter: CC-BY license, >=1080p, 120-3600 s, not Shorts. Resumable: existing OUT_JSON entries are kept.
"""
import json
import os
import random
import sys
import time
import urllib.parse

import yt_dlp

Q = {
 "indoor_static": ["apartment walkthrough 4k", "house tour walkthrough no talking", "museum walk 4k", "library walk 4k",
                   "hotel room tour 4k", "office tour walkthrough", "church interior walk 4k", "empty house tour 4k"],
 "indoor_dynamic": ["shopping mall walking tour 4k", "airport terminal walk 4k", "indoor market walk", "train station walk 4k",
                    "subway station walk 4k", "supermarket walk 4k", "food court walk 4k"],
 "outdoor_static": ["hiking trail pov 4k", "forest walk 4k", "mountain hike pov", "beach walk 4k no people",
                    "drone flight landscape 4k", "fpv drone nature", "canyon hike pov", "garden walk 4k"],
 "outdoor_dynamic": ["city walking tour 4k", "busy street walk 4k", "night market walk", "street food market walk 4k",
                     "old town walking tour 4k", "festival walk 4k", "rainy city walk 4k"],
 "driving": ["dashcam city driving 4k", "driving pov downtown 4k", "highway drive 4k", "night driving 4k",
             "rain driving pov", "mountain road drive 4k", "driving tour city 4k", "snow driving pov"],
 "embodied": ["gopro head mount pov", "cooking pov gopro", "woodworking pov", "cycling pov 4k", "mountain biking pov",
              "skiing pov gopro", "walking pov first person", "gardening pov"],
 "hard_static_cam": ["timelapse city 4k", "static camera street", "livestream camera intersection", "tripod nature video 4k"],
 "hard_far_aerial": ["drone flyover city high altitude", "aerial footage 4k high altitude", "airplane window view"],
 "hard_zoom": ["zoom in 4k camera test", "telephoto zoom wildlife"],
 "hard_edited": ["travel vlog cinematic 4k", "real estate cinematic video", "music video travel"],
 "hard_cg_game": ["game walkthrough 4k no commentary", "unreal engine 5 walkthrough", "minecraft walkthrough",
                  "3d animation architecture walkthrough"],
 "hard_lowlight": ["night walk dark 4k", "candle light video", "cave exploration pov"],
}
CC = "EgIwAQ%3D%3D"
out_path = sys.argv[1]
have = json.load(open(out_path)) if os.path.exists(out_path) else []
seen = {r["id"] for r in have}
flat = yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True, "playlistend": 25, "noprogress": True})
full = yt_dlp.YoutubeDL({"quiet": True, "skip_download": True, "noprogress": True})
done_q = {r["query"] for r in have}
for cat, qs in Q.items():
    for q in qs:
        if q in done_q:
            continue
        url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(q)}&sp={CC}"
        try:
            res = flat.extract_info(url, download=False)
        except Exception as e:  # noqa: BLE001
            print("search fail", q, str(e)[:80], flush=True)
            continue
        n_ok = 0
        for e in (res.get("entries") or [])[:20]:
            vid = e.get("id")
            if not vid or vid in seen:
                continue
            seen.add(vid)
            time.sleep(random.uniform(0.5, 1.5))
            try:
                i = full.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
            except Exception as ex:  # noqa: BLE001
                continue
            r = {"id": vid, "title": i.get("title"), "channel": i.get("channel"), "channel_id": i.get("channel_id"),
                 "license": i.get("license"), "duration": i.get("duration"), "height": i.get("height"),
                 "fps": i.get("fps"), "query": q, "cat_hint": cat, "url": i.get("webpage_url")}
            r["ok"] = bool(r["license"] and "Creative Commons" in r["license"] and (r["height"] or 0) >= 1080
                           and 120 <= (r["duration"] or 0) <= 3600 and "/shorts/" not in (r["url"] or ""))
            have.append(r)
            n_ok += r["ok"]
            if n_ok >= 10:
                break
        json.dump(have, open(out_path, "w"), indent=1, ensure_ascii=False)
        print(f"{cat:16s} {q:40s} ok={n_ok} total_ok={sum(x['ok'] for x in have)}", flush=True)
print("SEARCH_DONE", sum(x["ok"] for x in have), len(have), flush=True)
