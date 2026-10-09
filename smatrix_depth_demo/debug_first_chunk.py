import sys, os, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE); sys.path.insert(0,sys.argv[2]); os.environ["TQDM_DISABLE"]="1"
import numpy.ma, matplotlib
np.float, np.int, np.complex = float, int, complex
import torch
sys.path.insert(0,sys.argv[1])
import GradDS
import GradDS.AmpflowS as A
import smatrix_depth_demo as m
m.MU=60.0
d=np.load(HERE+'/results/layers_and_data.npz'); dps=d['dps']
K=m.K
dc=np.fft.fftshift(dps,axes=(-2,-1)).astype(np.float32).copy()
dims=np.array([20.,20.,K/m.LW,K/m.LW]); idx=np.arange(25)
sc_=np.zeros((3,25,25,2)); sc_[...,0]=(12+idx)[None,:,None]; sc_[...,1]=(12+idx)[None,None,:]
cap={}
class Stop(Exception): pass
_AT=A.AT
def AT(S,illum,pos,cw,*a,**k):
    out=_AT(S,illum,pos,cw,*a,**k)
    if 'Z' not in cap: cap.update(Z=np.array(out),illum=np.array(illum),pos=np.array(pos),S0=np.array(S))
    return out
_AI=A.AinvT
def AinvT(ew,illum,pos,cw,S,w,*a,**k):
    cap.update(resid_q=np.array(ew),w=w,S_before=np.array(S).copy())
    _AI(ew,illum,pos,cw,S,w,*a,**k)
    cap['S_after']=np.array(S).copy()
    raise Stop
A.AT=AT; A.AinvT=AinvT
try:
    GradDS.reconstruct_Smatrix_from_datacube(dc.copy(),dims,m.DEFOCI,20.0,3e5,mu=0.36923,niterations=1,nchunks=20,padding=1.5,scan_coordinates=sc_,stream_datacube=True,report_memory=False)
except Stop: pass
# --- mine, same first chunk
by,bx=m.beam_list(); nb=len(by); kby,kbx=by/m.LW,bx/m.LW; k2=kby**2+kbx**2
c0=48; coords=[(c0+4*a,c0+4*b) for a in range(25) for b in range(25)]
illum_m=np.zeros((20,nb),np.complex64)
for i in range(20):
    chi=-np.pi*m.LAM*k2*m.DEFOCI[0]; sh=-2*np.pi*(kby*coords[i][0]+kbx*coords[i][1])*m.DR
    illum_m[i]=np.exp(1j*(chi+sh))/np.sqrt(nb*K*K)
print("authors' first-chunk shapes: Z",cap['Z'].shape,"illum",cap['illum'].shape)
# map beams: authors' beam order -> mine
Sa=cap['S0']; print("S0 authors |S| mean",np.abs(Sa).mean(),"(expect 1/K=%.5f)"%(1/K))
# compare illumination up to the known factor K and beam ordering
ia=cap['illum']; 
# authors' beam list order (np.nonzero) -> reconstruct via initialize_Smatrix
_,bi,_,M_=A.initialize_Smatrix(20.0,3e5,(25,25,64,64),dims,padding=1.5)
key={(int(a),int(b)):i for i,(a,b) in enumerate(zip(*bi))}
order=[key[(int(3*a),int(3*b))] for a,b in zip(by,bx)]
ia_m=ia[:, order]
print("illum ratio authors/mine (should be const K=%d): mean %.4f std %.2e"%(K,np.mean(np.abs(ia_m)/np.abs(illum_m)),np.std(np.abs(ia_m)/np.abs(illum_m))))
print("illum phase max |diff| (rad):", np.max(np.abs(np.angle(ia_m*np.conj(illum_m)))))
# forward: authors' Z vs mine with S0 mine
S0m=(np.exp(2j*np.pi*kby[:,None]*(np.arange(m.Y)[None,:]*m.DR))[:,:,None]*np.exp(2j*np.pi*kbx[:,None]*(np.arange(m.X)[None,:]*m.DR))[:,None,:]).astype(np.complex64)
Zm=np.empty((20,K,K),np.complex64)
for i in range(20):
    y0,x0=coords[i][0]-K//2,coords[i][1]-K//2
    Zm[i]=np.fft.fft2(np.tensordot(illum_m[i],S0m[:,y0:y0+K,x0:x0+K],axes=(0,0)),norm='ortho')
print("forward Z: max|Z_a - Z_m| =",np.abs(cap['Z']-Zm).max(),"  max|Z|=",np.abs(Zm).max())
# pattern amplitude used
amp_m=np.sqrt(dps.reshape(-1,K,K)[:20]/dps.reshape(-1,K*K).sum(1).max())
print("authors' residual (q-space) norm per pattern vs mine:")
Zn=np.where(np.abs(Zm)>1e-12,Zm/np.maximum(np.abs(Zm),1e-12),0)
res_m=Zm-amp_m*Zn
print("  authors",np.linalg.norm(cap['resid_q'].reshape(20,-1),axis=1)[:5]," mine",np.linalg.norm(res_m.reshape(20,-1),axis=1)[:5])
print("  max |resid_a - resid_m| =",np.abs(cap['resid_q']-res_m).max(),"  max|resid_m|=",np.abs(res_m).max())
print("---- per-pattern diagnosis")
amp_a=np.abs(cap['Z']-cap['resid_q'])     # implied authors' amplitude  (Z - (Z - amp*phase)) = amp*phase
for i in range(5):
    dres=np.abs(cap['resid_q'][i]-res_m[i]); j=np.unravel_index(np.argmax(dres),dres.shape)
    print(f"pattern {i}: max|dresid|={dres.max():.4f} at {j}; |Z|={np.abs(Zm[i][j]):.3e}, amp_mine={amp_m[i][j]:.4e}, amp_authors(implied)={amp_a[i][j]:.4e}; n pixels differing >1e-6: {(dres>1e-6).sum()}")
da=np.abs(amp_a-amp_m); print("implied amplitude diff: max",da.max(),"; frac pixels >1e-5:",(da>1e-5).mean())
print("authors implied amp sum^2 per pattern:", (amp_a**2).sum((1,2))[:5], " mine:", (amp_m**2).sum((1,2))[:5])
print("---- (1) residual on numerically well-defined pixels (|Z|>1e-6)")
good=np.abs(Zm)>1e-6
print("  fraction of pixels well-defined: %.3f ; max|resid_a-resid_m| there: %.2e"%(good.mean(), np.abs(cap['resid_q']-res_m)[good].max()))
bad=~good
print("  numerically-zero pixels: %.3f of all; their share of mine residual energy: %.3f"%(bad.mean(), (np.abs(res_m[bad])**2).sum()/(np.abs(res_m)**2).sum()))
print("---- (2) back-projection with the SAME residual (authors' resid_q) applied by my update rule")
R=np.fft.ifft2(cap['resid_q'],axes=(-2,-1),norm='ortho').astype(np.complex64)
Sx=S0m.copy(); w=float(cap['w'])
for i in range(20):
    y0,x0=coords[i][0]-K//2,coords[i][1]-K//2
    coef=(w*np.conj(illum_m[i])/np.abs(illum_m[i])**2).astype(np.complex64)
    Sx[:,y0:y0+K,x0:x0+K]-=coef[:,None,None]*R[i][None]
Sa_after=cap['S_after'][order]*K
upd_m=Sx-S0m; upd_a=Sa_after-S0m
print("  ||update_a - update_m|| / ||update_m|| = %.2e"%(np.linalg.norm(upd_a-upd_m)/np.linalg.norm(upd_m)))
