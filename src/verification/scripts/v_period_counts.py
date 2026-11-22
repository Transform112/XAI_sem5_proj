import csv, json
from datetime import datetime, timezone
p=r"data\raw\NF-UNSW-NB15-v3\NF-UNSW-NB15-v3.csv"
b=[1421927376907,1421945819780,1421965347772,1424229622862,1424246259998,1424262564927]
print("boundaries ISO:", [datetime.fromtimestamp(x/1000,tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") for x in b])
import bisect
counts=[{} for _ in range(5)]
with open(p,newline="",encoding="utf-8",errors="replace") as f:
    r=csv.reader(f); hdr=next(r); ti=hdr.index("FLOW_START_MILLISECONDS"); ai=hdr.index("Attack")
    n=0
    for row in r:
        n+=1
        t=int(row[ti]); a=row[ai]
        k=min(max(bisect.bisect_left(b,t)-1,0),4)
        counts[k][a]=counts[k].get(a,0)+1
print("rows",n)
for i,c in enumerate(counts):
    print(f"P{i+1} edges={sum(c.values())} classes={len(c)}", dict(sorted(c.items(), key=lambda kv:-kv[1])[:5]))
