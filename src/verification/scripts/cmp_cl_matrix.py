import json, sys

diag = sys.argv[1]
a = json.load(open(diag + r"\results\tables\cl_matrix.json"))
b = json.load(open(diag + r"\results_cuda_fix\tables\cl_matrix.json"))


def find_R(obj):
    if isinstance(obj, dict):
        for k in ("R", "matrix", "values", "r_matrix"):
            if k in obj:
                v = find_R(obj[k])
                if v is not None:
                    return v
        for v in obj.values():
            r = find_R(v)
            if r is not None:
                return r
    if isinstance(obj, list) and obj and isinstance(obj[0], list) and all(
        isinstance(x, (int, float)) or x is None for x in obj[0]
    ):
        return obj
    return None


ra, rb = find_R(a), find_R(b)
print("shapes", None if ra is None else (len(ra), len(ra[0])), None if rb is None else (len(rb), len(rb[0])))
mx = 0.0
n = 0
for i in range(len(ra)):
    for j in range(len(ra[i])):
        va, vb = ra[i][j], rb[i][j]
        if va is None or vb is None:
            continue
        mx = max(mx, abs(va - vb))
        n += 1
print("cells compared", n, "max_abs_diff", mx)
print("keys A", list(a.keys()))
if "metrics" in a:
    print("A final", a["metrics"].get("final_average_macro_f1"))
if "metrics" in b:
    print("B final", b["metrics"].get("final_average_macro_f1"))
