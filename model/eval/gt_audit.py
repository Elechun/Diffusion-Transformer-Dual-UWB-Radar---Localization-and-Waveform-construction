"""The ground truth a given run was actually trained and scored against.

A run made with RESYNC=1 replaced the belt of every hand-aligned subject before training. Scoring that run
against the pipeline's original belts would be measuring it against a target it never saw, and would read
as a loss no matter how good the alignment was. So every script that opens a result file asks here which
belts belong to it, rather than reaching for diffusion_model.G and hoping.

The substitution rule is the same one the training script uses, and the two are checked against each other
by --verify rather than being trusted to have stayed in step:

    row i of a subject starts at i*HOP, verified at corr 1.0000 on all fourteen
    a window running past the end of the hand-cut belt keeps its original ground truth, which is what
    happens when the belt recording stopped before the radar did
    subjects in the run's exclusion list are skipped, so a belt cut for an excluded subject is inert
"""
import _path  # noqa: F401  (see _path.py)
import os,glob,numpy as np
import diffusion_model as D
PRE='/home/user1/Desktop/UWB_BIOPAC/preprocessed'
MS=f'{PRE}/matlab_sync'
W=D.NC*D.CH;HOP=D.CH*2

def manual_belts(excl=(2,10,13,21,22)):
    """returns (G with hand-aligned belts substituted, changed rows, {sub: (replaced, total, offset_s)})"""
    G2=D.G.copy().astype(np.float32);hit=[];did={}
    for f in sorted(glob.glob(f'{MS}/sub*_manual_sync.npz')):
        z=np.load(f,allow_pickle=True);u=int(z['sub'])
        if u in tuple(excl): continue
        man=np.asarray(z['gt_manual'],float);rows=np.flatnonzero(D.S==u)
        if not len(rows): continue
        ok=0
        for i,r in enumerate(rows):
            s0=i*HOP
            if s0+W>len(man): continue
            seg=man[s0:s0+W]
            if seg.std()<1e-9: continue
            G2[r]=((seg-seg.mean())/(seg.std()+1e-9)).astype(np.float32);hit.append(int(r));ok+=1
        did[u]=(ok,len(rows),float(z['offset_seconds']))
    return G2,np.array(sorted(hit)),did

def decoy_belts(excl=(2,10,13,21,22),seed=0):
    """The same subjects re-cut at a RANDOM offset drawn from the same allowed range.

    This is the control the hand alignment needs. The operator chose each offset while looking at the four
    radar candidates, so the belt was selected to match that subject's radar - and the model sees the same
    radar. If the picks are right, the model should benefit. If the picks are noise-fitting, the operator
    has quietly written a radar-matched target into the ground truth and the model would reproduce it
    either way. A random offset carries the same disturbance - a re-cut belt at a new phase, which alone
    lowers the donor score - but none of the matching. Whatever survives the difference is the alignment.

    The full belt is rebuilt here rather than reusing the stored cut, because a random offset needs belt
    the stored piece does not contain.
    """
    from scipy.io import loadmat
    from scipy.signal import butter,filtfilt
    import signal_processing as RC
    def band(x,fs,lo=0.1,hi=0.5):
        b,a=butter(3,[lo/(fs/2),hi/(fs/2)],btype='band');return filtfilt(b,a,x-np.mean(x))
    rs=np.random.RandomState(20250828+seed)
    G2=D.G.copy().astype(np.float32);hit=[];did={}
    for f in sorted(glob.glob(f'{MS}/sub*_manual_sync.npz')):
        z=np.load(f,allow_pickle=True);u=int(z['sub'])
        if u in tuple(excl): continue
        rows=np.flatnonzero(D.S==u)
        if not len(rows): continue
        E=loadmat(f'{MS}/sub{u:02d}_for_sync.mat')
        fs=float(E['fs'][0,0]);T=int(E['cand'].shape[1])
        R=loadmat(str(E['ref'][0]));lab=[str(x).strip() for x in R['labels'].ravel()]
        ch=next((i for i,l in enumerate(lab) if l.upper().startswith('RSP')),0)
        raw=np.asarray(R['data'][:,ch],float);fsb=1000.0/float(np.array(R['isi']).ravel()[0])
        tb=np.arange(len(raw))/fsb
        belt=band(np.interp(np.arange(0,tb[-1]*fs)/fs,tb,raw),fs)
        # Only offsets that still cover the whole radar. The aligner permitted more, but every offset the
        # operator actually chose was inside this range (7.3 to 29.2 s), and a decoy that truncated half
        # the record would differ from the hand alignment in how much belt it replaces as well as in where
        # it was cut - two changes at once, and the comparison stops meaning anything.
        hi=max(0,len(belt)-T)
        off=int(rs.randint(0,hi+1))
        man=belt[off:off+T];ok=0
        for i,r in enumerate(rows):
            s0=i*HOP
            if s0+W>len(man): continue
            seg=man[s0:s0+W]
            if seg.std()<1e-9: continue
            G2[r]=((seg-seg.mean())/(seg.std()+1e-9)).astype(np.float32);hit.append(int(r));ok+=1
        did[u]=(ok,len(rows),off/fs)
    return G2,np.array(sorted(hit)),did

def for_run(npz):
    """the belts that belong to one result file, read from the flags it stored"""
    z=npz if hasattr(npz,'files') else np.load(npz,allow_pickle=True)
    if 'resync' not in z.files or not int(z['resync']):
        return D.G.astype(np.float32),{}
    excl=tuple(int(v) for v in z['excl']) if 'excl' in z.files else (2,10,13,21,22)
    if int(z['resync'])==2:
        G2,_,did=decoy_belts(excl,seed=int(z['seed']))
    else:
        G2,_,did=manual_belts(excl)
    return G2,did

if __name__=='__main__':
    import sys
    G2,hit,did=manual_belts()
    print(f'  {len(hit)} windows across {len(did)} subjects carry a hand-aligned belt')
    for u in sorted(did):
        ok,tot,off=did[u];print(f'    sub{u:02d}  {off:+6.2f} s   {ok}/{tot} windows')
    if '--verify' in sys.argv:
        os.environ.setdefault('RESYNC','1')
        import br_candidates as BA
        H2,h2,d2=BA.manual_belts()
        same=np.array_equal(G2,H2) and np.array_equal(hit,h2) and did==d2
        print(f'\n  identical to the training script\'s own substitution: {same}')
        if not same: raise SystemExit('the two substitutions have drifted apart')
