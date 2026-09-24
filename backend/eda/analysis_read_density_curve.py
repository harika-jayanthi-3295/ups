"""Transformer accuracy vs read density, by subsampling the existing read stream.

Measures the curve Phase B would move the system along. Reports the two things
that move together when reads increase -- per-slot RSSI precision and antenna
coverage -- separately, because coverage saturates at 32 and precision does not.
"""
import sys, json, math, random
import numpy as np, torch
sys.path.insert(0,'/home/ups/backend/ml'); sys.path.insert(0,'/home/ups/backend/ml/model_1')
from common import MANIFEST_JSON, RUNS_DIR, run_power, usable_runs, stratified_run_split
from labels import load_antennas, tag_labels
from features import FeatureScaler, tag_dynamic_matrix, xyz_norm_stats
from dataset import Sample
import nn_common, transformer_model

SW, SS = 3.80, 0.76   # measured per-read and per-session sigmas (dB)

def thin(tag, frac, rng):
    out = {}
    for k, r in (tag.get("reads") or {}).items():
        pc = [p for p in (r.get("pc") or []) if rng.random() < frac]
        if not pc: continue
        rs = [p[2] for p in pc if p[2] is not None]
        if not rs: continue
        out[k] = {"count": len(pc), "avgRssi": sum(rs)/len(rs),
                  "minRssi": min(rs), "maxRssi": max(rs), "pc": pc}
    return {**tag, "reads": out}

ants = load_antennas(); order = [a.key for a in ants]
manifest = json.loads(MANIFEST_JSON.read_text())
train_ids, _ = stratified_run_split(manifest); train_set = set(train_ids)
runs = [(r["testId"], str(r["layout"]), run_power(r)) for r in usable_runs(manifest)]
raw = {tid: json.loads((RUNS_DIR/f"{tid}.json").read_text()) for tid,_,_ in runs}

def build(frac, seed):
    rng = random.Random(seed); tr=[]; te=[]
    for tid, lay, pw in runs:
        for epc, tag in raw[tid]["tags"].items():
            lab = tag_labels(tag, ants)
            if lab is None: continue
            t = tag if frac >= 1.0 else thin(tag, frac, rng)
            dyn, pres = tag_dynamic_matrix(t, order, pw)
            s = Sample(epc=epc, test_id=tid, layout=lay, box=str(tag.get("box","?")),
                       tag_num=int(tag.get("tagNum",0)), dyn=dyn, present=pres,
                       labels=lab, total_reads=int(pres.sum()), power_dbm=pw)
            s.nreads = sum(r["count"] for r in t["reads"].values())
            (tr if tid in train_set else te).append(s)
    return tr, te

def feats(samples, sc):
    return torch.from_numpy(np.stack([
        np.concatenate([sc.transform(s.dyn, s.present), s.present[:,None]],axis=1)
        for s in samples]).astype(np.float32))

xyz = np.array([[a.x,a.y,a.z] for a in ants],dtype=np.float32)
m,sd = xyz_norm_stats(xyz); xyz_n=(xyz-m)/sd

FRACS=(0.10,0.20,0.35,0.55,0.75,1.00)
print(f"{'frac':>5} {'rds/tag':>8} {'ants/tag':>9} {'rds/slot':>9} {'slotErr':>8} "
      f"{'EXACT':>7} {'>=24dBm':>8} {'<=21dBm':>8} {'ants>=24':>9} {'ants<=21':>9}")
res=[]
for frac in FRACS:
    nn_common.set_seed(13)
    tr, te = build(frac, seed=1000+int(frac*100))
    sc = FeatureScaler.fit(np.stack([s.dyn for s in tr]), np.stack([s.present for s in tr]))
    Xtr, Xte = feats(tr,sc), feats(te,sc)
    ytr = torch.tensor([s.labels["joint"] for s in tr],dtype=torch.long)
    model = transformer_model.TagTransformer(xyz_n, d_model=64, nhead=4, layers=2)
    w = nn_common.class_weights(tr)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    N=len(tr)
    for ep in range(200):
        model.train(); perm=torch.randperm(N)
        for i in range(0,N,64):
            idx=perm[i:i+64]; opt.zero_grad()
            nn_common.multi_head_loss(model(Xtr[idx]), {"joint":ytr[idx]}, w).backward(); opt.step()
    model.eval(); P=[]
    with torch.no_grad():
        for i in range(0,len(te),256): P.append(model(Xte[i:i+256])["joint"].argmax(1).numpy())
    P=np.concatenate(P); T=np.array([s.labels["joint"] for s in te])
    pw=np.array([s.power_dbm or 0 for s in te]); hi=pw>=24; lo=pw<=21
    apt=np.array([s.total_reads for s in te]); rpt=np.array([s.nreads for s in te])
    rps=rpt.sum()/max(apt.sum(),1)
    err=math.sqrt(SW**2/max(rps,1e-9)+SS**2)
    acc,ahi,alo=(P==T).mean(),(P[hi]==T[hi]).mean(),(P[lo]==T[lo]).mean()
    print(f"{frac:5.2f} {rpt.mean():8.1f} {apt.mean():9.2f} {rps:9.2f} {err:8.2f} "
          f"{acc*100:6.1f}% {ahi*100:7.1f}% {alo*100:7.1f}% {apt[hi].mean():9.2f} {apt[lo].mean():9.2f}")
    res.append((rps,err,acc,ahi,alo,apt[hi].mean()))

print("\n--- extrapolation to the noise floor (sigma_s = 0.76 dB) ---")
e=np.array([r[1] for r in res]); a=np.array([r[3] for r in res])   # >=24 dBm, coverage ~saturated
ok=a>0.05
for label,x in (("linear in log(slotErr)",np.log(e[ok])),("linear in 1/slotErr",1/e[ok])):
    A=np.vstack([x,np.ones(len(x))]).T
    c,_,_,_=np.linalg.lstsq(A,a[ok],rcond=None)
    xf=np.log(0.76) if "log" in label else 1/0.76
    pred=c[0]*xf+c[1]
    fit=A@c; r2=1-((a[ok]-fit)**2).sum()/((a[ok]-a[ok].mean())**2).sum()
    print(f"  {label:26} R^2={r2:.3f}   accuracy at floor -> {min(pred,1.0)*100:5.1f}%")
print("  (>=24 dBm only: coverage is already ~29/32 there, so the curve is precision-driven)")
