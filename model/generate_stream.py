"""Seam repair by CONTINUATION sampling instead of post-hoc bridging.

Today the whole-record trace is a concatenation of independently generated 42 s windows: each window
starts from its own noise, so its phase is arbitrary and every join is a step (11.6x a normal sample step
before bridging, 1.2x after bridge windows + raised-cosine, belt 1.04x). Bridging only hides the step.

Here consecutive windows OVERLAP by K chunks (K*6 s; stride 42-6K s) and the overlap is clamped to the
already generated stream during sampling (RePaint-style: at every DDIM step the first K chunks of x are
replaced by the forward-noised known tail, so the remaining chunks are sampled conditioned on a fixed
prefix through the DiT's cross-chunk attention). The stream is continuous by construction and the new
part's phase has to continue the old one. Same fold weights, same 84 s adaptation and calibration
guidance as the assembled system (combo arm), same rows for the conditioning radar.

Arms per subject (evaluated from 84 s on, first window at local row 14):
    indep   current system: disjoint windows (stride 42 s), plain join
    cont    continuation chain (stride 42-6K s), overlap clamped
    belt    the belt assembled the same way as cont (reference for the seam ratio)
Scores: seam ratio at joins, whole-record metrics, whole-record spectral bpm (which the seams used to
corrupt), and per-42 s-window metrics on the cont stream vs the independent windows at the same starts.
Env: SEED, K (overlap chunks, default 1), GUIDE (0.4).
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['SRC']='envdphi'
import numpy as np,torch
from metrics import zn
import diffusion_model as D
import network as V1
import augment as RA
import fewshot as FH
import rate_guidance as SG
import metrics_windowed as T
import tables as MTAB
from determinism import set_det,start_noise
from seam import seam_ratio
from metrics_timing import ev as ev_
PRE=V1.PRE;GTAG=V1.GTAG;dev=D.dev
NC,CH,FS,W=D.NC,D.CH,D.FS,V1.W
S,G,ZONE=D.S,D.G,D.ZONE;N=V1.N
EXCL=tuple(int(v) for v in os.environ.get('EXCL','2,10,13,21,22').split(','))
SEED=int(os.environ.get('SEED','0'));GUIDE=float(os.environ.get('GUIDE','0.4'));K=int(os.environ.get('K','1'))   # overlap in 6 s units
assert 1<=K<=3,'K = overlap in 6 s hop units, 1..3'
MODE=os.environ.get('MODE','hard')          # hard: clamp x and x0 every step, output prefix = known exactly
                                            # soft: clamp x only (RePaint), release the last RFREE steps, raised-cosine crossfade over the overlap
RFREE=int(os.environ.get('RFREE','10'));assert MODE in ('hard','soft')
assert not (int(os.environ.get('SIGNVOTE','0')) and MODE=='hard'),'SIGNVOTE needs the released overlap (MODE=soft)'
ANCHOR=int(os.environ.get('ANCHOR','0'))   # 1: the chain starts 6K s BEFORE 84 s with its overlap clamped to the calibration BELT (known by premise); stream still begins at 84 s
SIGNVOTE=int(os.environ.get('SIGNVOTE','0'))   # 1 (soft only): flip a generated window's sign if its own (released) overlap anti-correlates with the known piece it continues — GT-free, uses only generated/premise samples
KC=2*K                                                     # overlap in 3 s chunks (CH=51 samples = 3 s)
HOPROWS=int(round(6.0/getattr(D,'HOP_S',6.0)))            # rows per 6 s chunk (1 on the 6 s grid)
CALS=float(os.environ.get('CALS','84'))          # calibration block length (s): rate from the first CALS s, few-shot on windows inside it, stream starts at CALS
assert CALS>=6 and abs(CALS/6.0-round(CALS/6.0))<1e-9,'CALS must be a multiple of 6 s (the dataset row grid) and at least 6 s'
FIRST=int(round(CALS/6.0))*HOPROWS                  # local row of the first window starting at CALS s (84 -> 14, 42 -> 7)
NCAL=max(0,int((CALS-42.0)//6)+1)                   # few-shot rows fully inside the calibration block (84 -> 8, 42 -> 1, shorter -> 0:
                                                    # a few-shot row IS a 42 s window, so below 42 s none fits and adapting on one
                                                    # would show the model belt from the scored region. Short blocks give the rate
                                                    # anchor and the first window's 6 s clamp only - no weight update at all.
DONOR=int(os.environ.get('DONOR','0'))   # 1: few-shot on a DIFFERENT held-out subject's calibration block -> is the gain this person, or just 100 more steps?
NCAL=min(NCAL,int(os.environ['NCALX'])) if os.environ.get('NCALX','') else NCAL   # cap the few-shot rows: NCALX=0 with CALS=84 is
                                                    # 'long block, anchor only', which separates what the anchor buys from what few-shot buys
STRIDE_I=7*HOPROWS;STRIDE_C=(7-K)*HOPROWS                  # rows between windows: independent / continuation
WTAG=os.environ.get('WTAG','')                 # weight tag, e.g. _rc = random-crop (current deployed) weights
PERM=int(os.environ.get('PERM','0'))
RADAR=os.environ.get('RADAR','both')            # com | tv: zero the other radar's four envdphi channels at inference (matches single-radar training)
CALEST=os.environ.get('CALEST','spectral')      # calibration-rate estimator: spectral peak (current) | count = breath count over the calibration belt
PERMR=os.environ.get('PERMR','')                # '' full permutation (both radars displaced, current control) | com | tv:
PERMSL=slice(0,4) if PERMR=='com' else (slice(4,8) if PERMR=='tv' else None)   # displace ONLY that radar's four envdphi
assert PERMR in ('','com','tv'),'PERMR = com | tv'                            # channels; context, range profile and the
if PERMR: assert PERM,'PERMR needs PERM=1'                                    # other radar stay on the aligned window
ROLLRATE=float(os.environ.get('ROLLRATE','0'))   # >0: the guidance rate is (1-w)*calibration anchor + w*(breath count of the
ROLLS=float(os.environ.get('ROLLS','30'))        # last ROLLS seconds already generated). ROLLSRC=belt uses the belt instead
ROLLSRC=os.environ.get('ROLLSRC','self')         # (the oracle ceiling for a rolling estimator).
PROTO=os.environ.get('PROTO','')                 # (N,714) prototype per dataset row; injected in gen_cont like the event oracle
PROTO_GE=float(os.environ.get('PROTO_GE','0.3'));PROTO_ORDER=os.environ.get('PROTO_ORDER','guide_first');PROTO_WARP=int(os.environ.get('PROTO_WARP','0'))
PROTOARM=os.environ.get('PROTOARM','aligned')    # aligned: this row's prototype | perm: the partner row's prototype (prototype-permutation control)
PROTOW=np.load(f'{PRE}/{PROTO}').astype(np.float32) if PROTO else None
SHRINK=float(os.environ.get('SHRINK','0'))      # rate anchor = calib + SHRINK*(mean of the OTHER subjects' calibration rates - calib), GT-free empirical-Bayes          # 1: also run the chain with the conditioning radar taken from a window >=72 s away (same subject) -> stream_perm
agg=MTAB.agg;MTAG=('' if MODE=='hard' else f'_{MODE}{RFREE}')+('_anchor' if ANCHOR else '')+('_sv' if SIGNVOTE else '')+WTAG+('_perm'+PERMR if PERM else '')+(f'_shr{SHRINK:g}' if SHRINK>0 else '')+(f'_roll{ROLLRATE:g}x{ROLLS:g}{"B" if ROLLSRC=="belt" else ""}' if ROLLRATE>0 else '')+(f'_proto{PROTO_GE:g}{"gl" if PROTO_ORDER=="guide_last" else ""}{"w" if PROTO_WARP else ""}{"P" if PROTOARM=="perm" else ""}' if PROTO else '')+(f'_cal{CALS:g}' if CALS!=84 else '')+('_cnt' if CALEST=='count' else '')+(f'_{RADAR}' if RADAR!='both' else '')+(f'_in{os.environ.get("INCACHE")}' if os.environ.get('INCACHE') else '')+os.environ.get('OUTSUF','')

@torch.no_grad()
def _proto_template(row,rate):
    """the prototype for this row, z-scored, optionally time-warped so its breath count matches the anchor rate"""
    if PROTOW is None: return None
    w=PROTOW[row];w=(w-w.mean())/(w.std()+1e-9)
    if PROTO_WARP:
        pk=ev_(w,1);r0=len(pk)/(W/FS)*60 if len(pk)>=2 else rate
        f=float(np.clip(rate/max(r0,1e-3),0.6,1.6));tt=np.arange(W)*f;tt=(tt-tt.mean())+(W-1)/2
        w=np.interp(tt,np.arange(W),w,left=w[0],right=w[-1]);w=(w-w.mean())/(w.std()+1e-9)
    return torch.tensor(w.astype(np.float32),device=dev).reshape(1,-1)
def gen_cont(net,tok,tku,row,rate,g,known,seed=7,steps=50,proto_row=None):
    """SG.gen_guided for ONE window with the first K chunks clamped to `known` (K,CH) at every step"""
    n=1;x=start_noise([row],1,seed);tm=_proto_template(proto_row if proto_row is not None else row,rate)
    kn=torch.tensor(known,device=dev,dtype=x.dtype).unsqueeze(0);assert kn.shape[1]==KC
    masks=torch.stack([SG.narrow_mask(float(rate))])
    ts=torch.linspace(D.T_DIFF-1,0,steps).long().to(dev)
    gen=torch.Generator(device='cpu');gen.manual_seed(1000003*seed+int(row)+7919)
    for i in range(steps):
        a=D.ab[ts[i]]
        clamp=(MODE=='hard') or (i<steps-RFREE)
        if clamp:                                          # RePaint clamp: forward-noise the known prefix
            ek=torch.randn(kn.shape,generator=gen).to(dev)
            x=x.clone();x[:,:KC]=a.sqrt()*kn+(1-a).sqrt()*ek
        t=ts[i].repeat(n);ec=net(x,t,tok);eu=net(x,t,tku);e=eu+V1.GS*(ec-eu)
        x0=V1.band_any((x-(1-a).sqrt()*e)/a.sqrt(),1)
        x0c=V1.band_any((x-(1-a).sqrt()*ec)/a.sqrt(),1)
        r_=(x0c.reshape(n,-1).std(1)/(x0.reshape(n,-1).std(1)+1e-9)).reshape(n,1,1)
        x0=V1.PHI*(x0*r_)+(1-V1.PHI)*x0
        v=x0.reshape(n,-1);gi=SG.gstep(g,i,steps)
        def _guide(v):
            Fv=torch.fft.rfft(v,dim=1);nb=torch.fft.irfft(Fv*masks,n=W,dim=1);return (1-gi)*v+gi*nb
        def _blend(v):
            sd=v.std(1,keepdim=True)+1e-9;return (1-PROTO_GE)*v+PROTO_GE*sd*tm
        if tm is not None and PROTO_ORDER=='guide_last':
            v=_blend(v)
            if gi>0: v=_guide(v)
        else:
            if gi>0: v=_guide(v)
            if tm is not None: v=_blend(v)
        x0=v.reshape(n,NC,CH)
        if MODE=='hard': x0=x0.clone();x0[:,:KC]=kn
        e=(x-a.sqrt()*x0)/(1-a).sqrt()
        if i<steps-1: a2=D.ab[ts[i+1]];x=a2.sqrt()*x0+(1-a2).sqrt()*e
        else: x=x0
    return x[0].cpu().numpy()                              # (NC,CH)

@torch.no_grad()
def gen_one(net,row,XIQ,Cn,Pn,rate,g,swap_row=None,swap_ch=None):
    """swap_ch = the channel slice of ONE radar taken from swap_row instead of row (per-radar permutation control);
    context and range profile always stay on `row`, so only that radar's waveform evidence is displaced."""
    ci=torch.tensor(Cn[[row]],device=dev);pi=torch.tensor(Pn[[row]],device=dev);x=XIQ[[row]]
    if swap_ch is not None:
        x=x.copy();x[:,swap_ch]=XIQ[[swap_row]][:,swap_ch]
    xi=torch.tensor(x,device=dev)
    tok,_,_=net.cond(xi,ci,pi);tku,_,_=net.cond(torch.zeros_like(xi),ci,pi)
    return tok,tku

if __name__=='__main__':
    set_det(1)
    subs=np.array([u for u in np.unique(S) if u not in EXCL])
    P,folds=FH.plan(subs)                                      # folds are planned on ALL subjects so held-out weights stay correct
    if os.environ.get('ONLYSUB'):                              # then only the listed subjects are generated
        keep={int(v) for v in os.environ['ONLYSUB'].split(',')};folds=[np.array([u for u in te if int(u) in keep]) for te in folds]
    INCACHE=os.environ.get('INCACHE','')
    if INCACHE: XIQ=np.load(f'{PRE}/_{INCACHE}{GTAG}.npy').astype(np.float32);XIQ=XIQ.reshape(len(XIQ),-1,NC,CH);V1.NCH=XIQ.shape[1];V1.NCAND=XIQ.shape[1]//2;print(f'  input cache {INCACHE} {XIQ.shape}',flush=True)
    else: XIQ=RA.envdphi_cache()
    if RADAR=='com': XIQ=XIQ.copy();XIQ[:,4:]=0.0
    elif RADAR=='tv': XIQ=XIQ.copy();XIQ[:,:4]=0.0
    Gt2=G.reshape(N,NC,CH).astype(np.float32)
    calib={}
    for u in subs:
        g_=np.load(f'{PRE}/sub{u:02d}/data_fc729.npz',allow_pickle=True)['gt_aligned'].astype(float)[:int(CALS*FS)]
        gg=g_[np.isfinite(g_)];calib[int(u)]=(len(ev_(zn(gg),1))/(len(gg)/FS)*60) if CALEST=='count' else T.bpm(zn(gg))
    OUT={}
    print(f'  mode {MODE} (free steps {RFREE})  seed {SEED}  overlap {6*K} s = {KC} chunks, continuation stride {42-6*K} s ({STRIDE_C} rows), guide g={GUIDE}',flush=True)
    for fi,te in enumerate(folds):
        trm=~np.isin(S,te)&~np.isin(S,EXCL);Cn,Pn=D.fold_norm(trm)
        if RADAR in ('com','tv') and os.environ.get('GEOSPLIT','0')=='1':      # strict single radar: also drop the other radar's geometry context and profile half (as trained with GEOSPLIT=1)
            Cn=Cn.copy();Pn=Pn.copy();half=Pn.shape[-1]//2;keep={'com':[0,7,10],'tv':[1,8,11]}[RADAR];drop=[k for k in range(12) if k not in keep];Cn[...,drop]=0.0
            if RADAR=='com': Pn[...,half:]=0.0
            else: Pn[...,:half]=0.0
        ema=V1.Net().to(dev);ema.load_state_dict(torch.load(f'{PRE}/foldw/_envdphi_s2{WTAG}_s{SEED}_f{fi}.pt',map_location=dev))
        ema.eval();[p.requires_grad_(False) for p in ema.parameters()]
        for u in te:
            u=int(u);d=P[u];rows=d['rows'];L=len(rows);rate=calib[u]
            if SHRINK>0: rate=rate+SHRINK*(np.mean([calib[int(v)] for v in subs if int(v)!=u])-rate)
            cal_rows=P[d['partner']]['cal_head'] if DONOR else d['cal_head']   # DONOR: another HELD-OUT subject's block - same
            tg=('o' if DONOR else 'h')+f'{u}s{SEED}c{CALS:g}'                   # belt, same 100 steps, none of it this person
            nh=FH.adapt(ema,cal_rows[:NCAL],XIQ,Cn,Pn,Gt2,tg) if NCAL else ema   # few-shot on the calibration block only; no 42 s window fits below CALS=42
            # independent (current): disjoint windows from 84 s
            li=list(range(FIRST,L,STRIDE_I));ri=[int(rows[i]) for i in li]
            wi=[];bi=[]
            for r in ri:
                tok,tku=gen_one(nh,r,XIQ,Cn,Pn,rate,GUIDE)
                wi.append(SG.gen_guided(nh,tok,tku,np.array([r]),[rate],GUIDE).cpu().numpy().reshape(NC,CH));bi.append(Gt2[r])
            # continuation chain
            c0=FIRST-K*HOPROWS if ANCHOR else FIRST
            lc=list(range(c0,L,STRIDE_C));rc=[int(rows[i]) for i in lc]
            stream=[];bstream=[];wc_full=[];joins_c=[];nflip=0
            for j,r in enumerate(rc):
                rate_j=rate
                if ROLLRATE>0 and j>0:                    # rolling anchor: re-read the rate from what has already been produced
                    src=np.concatenate(bstream) if ROLLSRC=='belt' else np.concatenate(stream)
                    tail=src.reshape(-1)[-int(ROLLS*FS):]
                    if len(tail)>int(10*FS) and np.std(tail)>1e-9:
                        rr=len(ev_(zn(tail),1))/(len(tail)/FS)*60
                        if 8<=rr<=35: rate_j=(1-ROLLRATE)*rate+ROLLRATE*rr
                tok,tku=gen_one(nh,r,XIQ,Cn,Pn,rate_j,GUIDE)
                if j==0 and ANCHOR:                       # overlap = belt 78..84 s of THIS window (calibration block, known by premise)
                    w=gen_cont(nh,tok,tku,r,rate_j,GUIDE,Gt2[r][:KC],proto_row=(rc[(j+max(2,len(rc)//2))%len(rc)] if PROTOARM=='perm' else r))
                    if SIGNVOTE and np.corrcoef(w[:KC].reshape(-1),Gt2[r][:KC].reshape(-1))[0,1]<0: w=-w;nflip+=1
                    stream.append(w[KC:]);bstream.append(Gt2[r][KC:])   # the stream starts at 84 s; the belt prefix is never output
                elif j==0:
                    w=SG.gen_guided(nh,tok,tku,np.array([r]),[rate_j],GUIDE).cpu().numpy().reshape(NC,CH)
                    stream.append(w);bstream.append(Gt2[r])
                else:
                    known=np.concatenate(stream)[-KC:]
                    w=gen_cont(nh,tok,tku,r,rate_j,GUIDE,known,proto_row=(rc[(j+max(2,len(rc)//2))%len(rc)] if PROTOARM=='perm' else r))
                    if SIGNVOTE and np.corrcoef(w[:KC].reshape(-1),known.reshape(-1))[0,1]<0: w=-w;nflip+=1
                    joins_c.append(sum(len(s) for s in stream)*CH)
                    if MODE=='soft':                      # replace the stream's tail by a raised-cosine blend known -> generated
                        ov=KC*CH;fw=0.5*(1-np.cos(np.linspace(0,np.pi,ov)))     # 0 at the start of the overlap, 1 at its end
                        tail=np.concatenate(stream).reshape(-1)[-ov:];newov=w[:KC].reshape(-1)
                        blend=((1-fw)*tail+fw*newov).astype(np.float32)
                        last=stream[-1].reshape(-1);last[-ov:]=blend;stream[-1]=last.reshape(-1,CH)
                    stream.append(w[KC:])
                    # belt: each G row is z-scored on its own window; map the new row onto the stream's scale
                    # through the overlap so the belt reference has no artificial scale steps at joins
                    bt=np.concatenate(bstream)[-KC:].reshape(-1);bo=Gt2[r][:KC].reshape(-1)
                    sc_,of_=np.polyfit(bo,bt,1);bstream.append((sc_*Gt2[r][KC:]+of_).astype(np.float32))
                wc_full.append(w)
            stream_p=[]
            if PERM and len(rc)>=4:                       # permutation control: same chain, same start noise, same rate, radar from a window >= 72 s away
                Lc=len(rc);sh=max(2,Lc//2)
                for j,r in enumerate(rc):
                    rp=rc[(j+sh)%Lc]
                    tok,tku=(gen_one(nh,r,XIQ,Cn,Pn,rate,GUIDE,swap_row=rp,swap_ch=PERMSL) if PERMR
                             else gen_one(nh,rp,XIQ,Cn,Pn,rate,GUIDE))
                    if j==0 and ANCHOR: w=gen_cont(nh,tok,tku,r,rate,GUIDE,Gt2[r][:KC]);stream_p.append(w[KC:])
                    elif j==0: w=SG.gen_guided(nh,tok,tku,np.array([r]),[rate],GUIDE).cpu().numpy().reshape(NC,CH);stream_p.append(w)
                    else:
                        known=np.concatenate(stream_p)[-KC:];w=gen_cont(nh,tok,tku,r,rate,GUIDE,known)
                        if MODE=='soft':
                            ov=KC*CH;fw=0.5*(1-np.cos(np.linspace(0,np.pi,ov)));tail=np.concatenate(stream_p).reshape(-1)[-ov:];newov=w[:KC].reshape(-1)
                            last=stream_p[-1].reshape(-1);last[-ov:]=((1-fw)*tail+fw*newov).astype(np.float32);stream_p[-1]=last.reshape(-1,CH)
                        stream_p.append(w[KC:])
            del nh;torch.cuda.empty_cache()
            OUT[u]=dict(rows_i=np.array(ri),wi=np.array(wi),bi=np.array(bi),rows_c=np.array(rc),
                        stream=np.concatenate(stream).reshape(-1),bstream=np.concatenate(bstream).reshape(-1),
                        wc_full=np.array(wc_full),joins_c=np.array(joins_c),nflip=nflip,stream_perm=(np.concatenate(stream_p).reshape(-1) if stream_p else np.zeros(0,np.float32)))
        del ema;torch.cuda.empty_cache();print(f'  fold{fi} done'+(f'  (sign flips so far {sum(OUT[k]["nflip"] for k in OUT)})' if SIGNVOTE else ''),flush=True)
    # save the generations first so a scoring error cannot lose them
    np.savez_compressed(f'{PRE}/_seam_cont{GTAG}_K{K}{MTAG}_s{SEED}_raw.npz',seed=SEED,K=K,subs=subs,
                        **{f'sub{u:02d}_{k}':OUT[int(u)][k] for u in subs if int(u) in OUT for k in ('rows_i','wi','bi','rows_c','stream','bstream','wc_full','joins_c','stream_perm')})
    # ---- scores ----
    half=int(2*FS);KEYS=['Corr','ACF','DTW','CSp','F1','dtp','dtv','mp']
    R={a:{k:[] for k in KEYS+['seam','bpm_spec','bpm_cnt','dRR_spec']} for a in ('indep','cont','belt','asine')}
    FW={a:[] for a in ('indep','cont','asine')}     # first 42 s window (84..126 s) corr per subject
    PW={'indep':[],'cont':[]}
    from metrics_timing import ev
    cnt=lambda x:len(ev(x,1))/(len(x)/FS)*60
    subs=np.array([u for u in subs if int(u) in OUT])          # ONLYSUB: summarise only what was generated
    for u in subs:
        o=OUT[int(u)]
        si=np.concatenate([zn(w.reshape(-1)) for w in o['wi']]);gi=np.concatenate([zn(b.reshape(-1)) for b in o['bi']])
        sc=o['stream'];gc=o['bstream'];Lm=min(len(si),len(sc));si,gi,sc,gc=si[:Lm],gi[:Lm],sc[:Lm],gc[:Lm]
        ji=[k*W for k in range(1,len(o['wi'])) if k*W<Lm];jc=[int(j) for j in o['joins_c'] if j<Lm]
        # anchored sine: calibration rate, phase fitted on the belt's last 6 s before 84 s, then free-running (the rate-only competitor of ANCHOR)
        r0=int(P[int(u)]['rows'][FIRST-K*HOPROWS]);pre=zn(Gt2[r0][:KC].reshape(-1));tp=np.arange(len(pre))/FS-len(pre)/FS
        f=calib[int(u)]/60;ph=np.arctan2(np.dot(pre,np.sin(2*np.pi*f*tp)),np.dot(pre,np.cos(2*np.pi*f*tp)))
        asn=zn(np.cos(2*np.pi*f*np.arange(Lm)/FS-ph))
        for a,rec,gt,jn in (('indep',si,gi,ji),('cont',sc,gc,jc),('belt',gc,gc,jc),('asine',asn,gc,[])):
            m=agg(zn(rec),zn(gt),None)
            for k in KEYS: R[a][k].append(float(np.corrcoef(zn(rec),zn(gt))[0,1]) if k=='Corr' else m[k])
            jn2=sorted(set(jn+[j-KC*CH for j in jn if j-KC*CH>0])) if (MODE=='soft' and a!='indep') else jn
            R[a]['seam'].append(np.mean(seam_ratio(rec,jn2,half)) if jn2 else np.nan)
            R[a]['bpm_spec'].append(T.bpm(zn(rec)));R[a]['bpm_cnt'].append(cnt(zn(rec)))
            R[a]['dRR_spec'].append(abs(T.bpm(zn(rec))-T.bpm(zn(gt))))
            if a in FW: FW[a].append(float(np.corrcoef(zn(rec[:W]),zn(gt[:W]))[0,1]))
        # per-window: the cont stream cut at its own window starts vs an independent window at the same rows
        for j,r in enumerate(o['rows_c']):
            s0=j*(NC-KC)*CH;seg=sc[s0:s0+W]
            if len(seg)<W: continue
            gt=zn(G[r]);mi=agg(zn(seg),gt,W)
            PW['cont'].append([np.corrcoef(zn(seg),gt)[0,1],mi['CSp'],mi['ACF'],mi['F1']])
        for r,w in zip(o['rows_i'],o['wi']):
            mi=agg(zn(w.reshape(-1)),zn(G[r]),W);PW['indep'].append([np.corrcoef(zn(w.reshape(-1)),zn(G[r]))[0,1],mi['CSp'],mi['ACF'],mi['F1']])
    from scipy.stats import ttest_rel
    print(f'\n  whole-record, n={len(subs)} subjects (from 84 s), K={K}')
    print(f'  {"metric":9s} {"indep":>8s} {"cont":>8s} {"belt":>8s} {"a-sine":>8s}   cont-indep         cont-asine')
    for k in ['seam','Corr','CSp','ACF','DTW','F1','dtp','dtv','mp','dRR_spec']:
        a=np.array(R['indep'][k]);b=np.array(R['cont'][k]);c=np.array(R['belt'][k]);d=np.array(R['asine'][k]);g=np.isfinite(a)&np.isfinite(b)
        p=ttest_rel(a[g],b[g])[1] if g.sum()>2 else np.nan;g2=np.isfinite(b)&np.isfinite(d);p2=ttest_rel(b[g2],d[g2])[1] if g2.sum()>2 else np.nan
        print(f'  {k:9s} {np.nanmean(a):8.3f} {np.nanmean(b):8.3f} {np.nanmean(c):8.3f} {np.nanmean(d):8.3f}   {np.nanmean(b-a):+.3f} p={p:.3f}   {np.nanmean(b[g2]-d[g2]):+.3f} p={p2:.3f}')
    fi_,fc_,fs_=(np.array(FW[a]) for a in ('indep','cont','asine'))
    print(f'  first window (84-126 s) corr: indep {fi_.mean():+.3f} | cont {fc_.mean():+.3f} | anchored sine {fs_.mean():+.3f}; negative (flipped) subjects: indep {(fi_<0).sum()} cont {(fc_<0).sum()} sine {(fs_<0).sum()} / {len(fi_)}')
    print('  per-subject first-window corr (indep -> cont):',' '.join(f'{int(u)}:{x:+.2f}->{y:+.2f}' for u,x,y in zip(subs,fi_,fc_)))
    for a in ('indep','cont'):
        q=np.array(PW[a]);print(f'  per-window {a:6s}: n={len(q)} corr {q[:,0].mean():+.4f} CSp {q[:,1].mean():.3f} ACF {q[:,2].mean():.3f} F1 {q[:,3].mean():.3f}')
    np.savez_compressed(f'{PRE}/_seam_cont{GTAG}_K{K}{MTAG}_s{SEED}.npz',seed=SEED,K=K,subs=subs,
                        **{f'{a}_{k}':np.array(v) for a in R for k,v in R[a].items()},
                        **{f'sub{u:02d}_{k}':OUT[int(u)][k] for u in subs if int(u) in OUT for k in ('rows_i','wi','bi','rows_c','stream','bstream','wc_full','joins_c')})
    print(f'  saved {PRE}/_seam_cont{GTAG}_K{K}{MTAG}_s{SEED}.npz')
