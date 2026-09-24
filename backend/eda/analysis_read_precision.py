"""Isolate the PRECISION axis: does more reads-per-slot help with coverage and
sample count held fixed?

Layout 1 has 3 repeat runs at each of the 6 powers, same 250 tag positions.
Pool all 3 per power -> one high-density set per power. Then compare that set at
full density against the SAME set thinned back to 1/3, so the only thing that
differs is reads per slot. Split by power (train 30/24/18, test 27/21/15)
because pooling consumes the repeat sessions.
"""
import sys, json, math, random, collections
import numpy as np, torch
sys.path.insert(0,'/home/ups/backend/ml'); sys.path.insert(0,'/home/ups/backend/ml/model_1')
from common import MANIFEST_JSON, RUNS_DIR, run_power
from labels import load_antennas, tag_labels
from features import FeatureScaler, tag_dynamic_matrix, xyz_norm_stats
from dataset import Sample
import nn_common, transformer_model

SW, SS = 3.80, 0.76
ants = load_antennas(); order=[a.key for a in ants]
manifest = json.loads(MANIFEST_JSON.read_text())
L1 = [r for r in manifest if str(r["layout"])=="1"]
by_pw = collections.defaultdict(list)
for r in L1: by_pw[run_power(r)].append(r["testId"])
print("layout-1 runs per power:", {p:len(v) for p,v in sorted(by_pw.items(),reverse=True)})

def pool(tids):
    """Merge the per-(tag,antenna) read lists of several runs of the same layout."""
    acc=collections.defaultdict(lambda: collections.defaultdict(list)); meta={}
    for tid in tids:
        d=json.loads((RUNS_DIR/f"{tid}.json").read_text())
        for epc,tag in d["tags"].items():
            meta[epc]=tag
            for k,r in (tag.get("reads") or {}).items():
                acc[epc][k].extend(r.get("pc") or [])
    return acc, meta

def make(acc, meta, pw, frac, rng):
    out=[]
    for epc, slots in acc.items():
        tag=meta[epc]
        lab=tag_labels(tag, ants)
        if lab is None: continue
        reads={}
        for k,pc in slots.items():
            keep=pc if frac>=1.0 else [p for p in pc if rng.random()<frac]
            rs=[p[2] for p in keep if p[2] is not None]
            if not rs: continue
            reads[k]={"count":len(keep),"avgRssi":sum(rs)/len(rs),
                      "minRssi":min(rs),"maxRssi":max(rs),"pc":keep}
        t={**tag,"reads":reads}
        dyn,pres=tag_dynamic_matrix(t, order, pw)
        s=Sample(epc=epc,test_id=f"L1_{pw:g}",layout="1",box=str(tag.get("box","?")),
                 tag_num=int(tag.get("tagNum",0)),dyn=dyn,present=pres,labels=lab,
                 total_reads=int(pres.sum()),power_dbm=pw)
        s.nreads=sum(r["count"] for r in reads.values())
        out.append(s)
    return out

POOLED={p: pool(by_pw[p]) for p in by_pw}
TRAIN_PW=[30.0,24.0,18.0]; TEST_PW=[27.0,21.0,15.0]
xyz=np.array([[a.x,a.y,a.z] for a in ants],dtype=np.float32)
m,sd=xyz_norm_stats(xyz); xyz_n=(xyz-m)/sd

def feats(ss,sc):
    return torch.from_numpy(np.stack([
        np.concatenate([sc.transform(s.dyn,s.present),s.present[:,None]],axis=1)
        for s in ss]).astype(np.float32))

print(f"\n{'condition':16} {'rds/tag':>8} {'ants/tag':>9} {'rds/slot':>9} {'slotErr':>8} "
      f"{'n_tr':>5} {'n_te':>5} {'EXACT':>7}")
for label, frac in (("1x (thinned)",1/3), ("3x (pooled)",1.0)):
    rng=random.Random(7); nn_common.set_seed(13)
    tr=[s for p in TRAIN_PW for s in make(*POOLED[p], p, frac, rng)]
    te=[s for p in TEST_PW  for s in make(*POOLED[p], p, frac, rng)]
    sc=FeatureScaler.fit(np.stack([s.dyn for s in tr]),np.stack([s.present for s in tr]))
    Xtr,Xte=feats(tr,sc),feats(te,sc)
    ytr=torch.tensor([s.labels["joint"] for s in tr],dtype=torch.long)
    model=transformer_model.TagTransformer(xyz_n,d_model=64,nhead=4,layers=2)
    w=nn_common.class_weights(tr)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    N=len(tr)
    for ep in range(200):
        model.train(); perm=torch.randperm(N)
        for i in range(0,N,64):
            idx=perm[i:i+64]; opt.zero_grad()
            nn_common.multi_head_loss(model(Xtr[idx]),{"joint":ytr[idx]},w).backward(); opt.step()
    model.eval(); P=[]
    with torch.no_grad():
        for i in range(0,len(te),256): P.append(model(Xte[i:i+256])["joint"].argmax(1).numpy())
    P=np.concatenate(P); T=np.array([s.labels["joint"] for s in te])
    apt=np.array([s.total_reads for s in te]); rpt=np.array([s.nreads for s in te])
    rps=rpt.sum()/max(apt.sum(),1); err=math.sqrt(SW**2/max(rps,1e-9)+SS**2)
    print(f"{label:16} {rpt.mean():8.1f} {apt.mean():9.2f} {rps:9.2f} {err:8.2f} "
          f"{len(tr):5d} {len(te):5d} {(P==T).mean()*100:6.1f}%")
