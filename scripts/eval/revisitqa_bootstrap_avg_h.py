"""Episode-clustered bootstrap (1,000 resamples, seed 20260817) of the seven-subset mean H for all 22 models.

Resamples episodes within each subset, recomputes KA, AR and H per subset, averages over subsets, and prints the 95% interval
plus paired differences among the top four. Point values are checked against result.json. Writes outputs/revisitqa/released/avgH_ci.json.
Run from any directory on a node with outputs/ (B or C): python scripts/eval/revisitqa_bootstrap_avg_h.py
"""
import json, numpy as np, os
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
R="outputs/revisitqa/released"
S=["cook","bike","healthy","music","sports","thor","vhome"]
OFF="qwen25_vl_3b qwen25_vl_7b qwen3_vl_4b qwen3_vl_8b qwen3_vl_32b qwen35_4b qwen35_9b internvl35_8b molmo2_8b minicpm_v45 internvideo25_8b eagle25_8b videollama3_7b llava_next_video_7b glm41v_9b".split()
STR="flash_vstream_7b livecc_7b timechat_online_7b rekv_llava_ov_7b moviechat_7b streamchat_7b dispider_7b".split()
models=[("online",k) for k in OFF]+[("streaming_native",k) for k in STR]
def load(p,k,s):
    rows=[json.loads(l) for l in open(f"{R}/{s}/models/{p}/{k}/pred.jsonl") if l.strip()]
    res=json.load(open(f"{R}/{s}/models/{p}/{k}/result.json"))
    return rows,res
B=1000; rng=np.random.default_rng(20260817)
eps_idx={}  # per subset: episode list and resample matrix
data={}
for s in S:
    rows,res=load("online","qwen3_vl_8b",s)
    eps=sorted({r["episode_id"] for r in rows}); eps_idx[s]=(eps,rng.integers(0,len(eps),(B,len(eps))))
for p,k in models:
    Hs=np.zeros((B,len(S))); point=[]
    for si,s in enumerate(S):
        rows,res=load(p,k,s); eps,idx=eps_idx[s]; pos={e:i for i,e in enumerate(eps)}
        kn=np.zeros(len(eps)); kc=np.zeros(len(eps)); un=np.zeros(len(eps)); uc=np.zeros(len(eps))
        for r in rows:
            i=pos[r["episode_id"]]
            if r["answer"]==3: un[i]+=1; uc[i]+=bool(r["correct"])
            else: kn[i]+=1; kc[i]+=bool(r["correct"])
        assert int(un.sum())==res["uncertain_count"] and int(kn.sum())==res["known_count"],(p,k,s)
        def h(kn_,kc_,un_,uc_):
            ka=kc_.sum(-1)/np.maximum(kn_.sum(-1),1); ar=uc_.sum(-1)/np.maximum(un_.sum(-1),1)
            return np.where(ka+ar>0,2*ka*ar/np.maximum(ka+ar,1e-9),0)
        Hs[:,si]=100*h(kn[idx],kc[idx],un[idx],uc[idx])
        point.append(100*h(kn[None],kc[None],un[None],uc[None])[0])
        assert abs(point[-1]-100*res["known_abstention_hmean"])<0.05,(p,k,s,point[-1],res["known_abstention_hmean"])
    data[k]=(np.mean(point),Hs.mean(1))
out={k:{"mean":float(v[0]),"lo":float(np.percentile(v[1],2.5)),"hi":float(np.percentile(v[1],97.5))} for k,v in data.items()}
json.dump(out,open("outputs/revisitqa/released/avgH_ci.json","w"),indent=1)
for k,v in sorted(out.items(),key=lambda x:-x[1]["mean"])[:8]: print(f"{k:18s} {v['mean']:.1f} [{v['lo']:.1f}, {v['hi']:.1f}] halfwidth {(v['hi']-v['lo'])/2:.1f}")
top=["glm41v_9b","videollama3_7b","molmo2_8b","qwen35_4b"]
for a in top:
    for b in top:
        if a<b:
            d=data[a][1]-data[b][1]; print(f"{a} - {b}: {data[a][0]-data[b][0]:+.2f} CI [{np.percentile(d,2.5):+.1f}, {np.percentile(d,97.5):+.1f}]")
