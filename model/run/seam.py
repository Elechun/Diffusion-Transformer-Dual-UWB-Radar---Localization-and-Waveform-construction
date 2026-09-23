"""Join consecutive 42 s windows into one continuous trace, so the figures show a stream and not blocks.

Each window is generated from its own noise, so where two meet the trace can jump or double back - measured
at 11.6x a normal sample step against the belt's 1.7x. The repair is to generate one more window CENTRED on
each join and hand the trace over to it across a short cross-fade, so the samples either side of a join come
from a single continuous generation. Measured on 45 subjects and 209 joins, that takes the step to 1.17x.

Nothing is smoothed: every surviving peak keeps its position. The generator has no fixed sign, so a bridge
is flipped to agree with what it is joining before it is blended in.

    stitch(main_out, bridges, hop, W, fade)   ->  one array of length len(main_out)*W

main_out    the disjoint windows in time order, each already generated
bridges     dict mapping the index of the join (1..len(main)-1) to a generated window and the offset, in
            samples, of the join within that window
"""
import _path  # noqa: F401  (see _path.py)
import numpy as np
def _zn(x):
    x=np.asarray(x,float);return (x-x.mean())/(x.std()+1e-9)
def stitch(main_out,bridges,W,fade):
    """main_out (n,W); bridges {join_index: (window, offset_of_join_within_window)}"""
    n=len(main_out);T=n*W
    out=np.concatenate([_zn(m) for m in main_out]).astype(float)
    for jn,(bw,off) in bridges.items():
        c=jn*W                                   # where the join sits in the stitched trace
        a0,a1=c-fade,c+fade
        b0,b1=off-fade,off+fade
        if a0<0 or a1>T or b0<0 or b1>len(bw): continue
        seg=_zn(bw)[b0:b1];tgt=out[a0:a1]
        if seg.std()<1e-9 or tgt.std()<1e-9: continue
        r=float(np.corrcoef(seg,tgt)[0,1])
        if not np.isfinite(r): continue
        if r<0: seg=-seg                          # the generator has no fixed sign
        w=0.5*(1-np.cos(np.linspace(0,2*np.pi,len(seg))))   # zero at the edges, one at the join
        out[a0:a1]=(1-w)*tgt+w*seg
    return out
def bridge_plan(disj_idx,all_idx_of_subject,hop,W):
    """for each join, the dataset window whose centre is nearest to it, and where the join falls inside it.
    Windows are spaced `hop` samples apart, so a join at k*W is nearest the window starting at k*W - W/2."""
    plan={}
    pos={int(v):p for p,v in enumerate(all_idx_of_subject)}
    for jn in range(1,len(disj_idx)):
        want=jn*W-W//2                            # the start that would centre this window on the join
        j=int(round(want/hop))
        if j<0 or j>=len(all_idx_of_subject): continue
        s0=j*hop
        off=jn*W-s0                               # where the join lands inside that window
        if off<W//8 or off>W-W//8: continue       # too near an edge to bridge anything
        plan[jn]=(int(all_idx_of_subject[j]),int(off))
    return plan
def seam_ratio(y,joins,half):
    """the step at each join against a normal step nearby - the number the repair is judged on"""
    d=np.abs(np.diff(np.asarray(y,float)));out=[]
    for c in joins:
        if c-half<1 or c+half>=len(d): continue
        near=np.concatenate([d[c-half:c-2],d[c+2:c+half]])
        if len(near)<5 or np.median(near)<1e-9: continue
        out.append(float(np.max(d[c-1:c+2])/np.median(near)))
    return out
