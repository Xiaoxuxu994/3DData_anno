import numpy as np, pandas as pd
for k in ["A", "B", "scannetpp", "wild"]:
    d = pd.read_csv(f"{k}_per_clip.csv")
    for dep in ["pi3x", "da3"]:
        b = d[d.method == f"H:vipe+{dep}+geo"].set_index("clip"); a = d[d.method == f"H:vipe+{dep}+med"].set_index("clip")
        r = np.log(b[f"scale_{dep}"] / a[f"scale_{dep}"])
        print(k, dep, "fallback", round(b[f"geo_fallback_{dep}"].mean(), 2), "ratio_med", round(float(np.exp(r.median())), 3),
              "frac|dlog|>0.1", round(float((r.abs() > 0.1).mean()), 2))
        if k == "wild":
            print("  fallback by cat", b.groupby("cat")[f"geo_fallback_{dep}"].mean().round(2).to_dict())
