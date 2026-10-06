"""Historical figure script retained for provenance, not current reproduction.

Several inputs and labels predate the current corrected analysis. Follow
``docs/reproducibility.md`` and do not use this script alone to support a manuscript claim.
"""
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import willard_osa as OSA

BLUE='#0072B2'; VERM='#D55E00'; GREEN='#009E73'; GRAY='#555555'
plt.rcParams.update({
    'font.family':'serif','mathtext.fontset':'cm',
    'font.size':10,'axes.labelsize':10,'axes.titlesize':10,'xtick.labelsize':9,'ytick.labelsize':9,
    'legend.fontsize':8.5,'axes.linewidth':0.8,'lines.linewidth':1.7,'lines.markersize':5.5,
    'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,
    'savefig.dpi':300,'axes.grid':False,'grid.alpha':0.22,'grid.linewidth':0.5,
})

# ============ Figure 1: two-sided boundary mechanism (fixed layout) ============
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.0,3.0))
sg=np.linspace(0.15,3.0,400)
a1.plot(sg,norm.cdf(0.6/sg),color=GREEN,ls='-.')      # mu_g-g^dagger = -0.6 (feasible side)
a1.plot(sg,0.5*np.ones_like(sg),color=VERM,ls='-')    # on boundary -> flat
a1.plot(sg,norm.cdf(-0.6/sg),color=BLUE,ls='--')      # mu_g-g^dagger = +0.6 (infeasible side)
a1.text(1.6,0.93,r'feasible side ($\mu_g{-}g^\dagger={-}0.6$)',color=GREEN,fontsize=7.5,ha='center')
a1.text(0.12,0.555,r'boundary ($\mu_g{=}g^\dagger$): flat in $\sigma_g$',color=VERM,fontsize=7.5,ha='left')
a1.text(1.6,0.07,r'infeasible side ($\mu_g{-}g^\dagger={+}0.6$)',color=BLUE,fontsize=7.5,ha='center')
a1.set_xlabel(r'posterior toxicity SD  $\sigma_g$'); a1.set_ylabel(r'feasibility weight  $\Phi((g^\dagger-\mu_g)/\sigma_g)$')
a1.set_ylim(-0.02,1.02); a1.set_xlim(0,3); a1.set_title('(a) myopic cEI',loc='left')
# hard-gate KG VoI at the boundary, NORMALISED by the efficacy gap (h-b) so no arbitrary scale enters:
# alpha_KG/(h-b) = Phi(-Phi^{-1}(tau)/kappa) - 1{tau<1/2}, kappa = sigma_g/sigma_yg (sigma_yg=1, obs-noise SD; = Lemma 2 eq:kgboundary) -- exact closed form
def kg_norm(s,g): return norm.cdf(-norm.ppf(g)/s)-(1.0 if g<0.5 else 0.0)
a2.axhline(0,color=GRAY,lw=0.8,ls=':')
for g,c,ls in [(0.9,BLUE,'-'),(0.6,GREEN,'--'),(0.3,VERM,'-.')]:
    a2.plot(sg,[kg_norm(s,g) for s in sg],color=c,ls=ls,label=fr'$\tau={g}$')
a2.text(0.95,0.50,r'strict threshold ($\tau>\frac{1}{2}$): increasing',color=GREEN,fontsize=8,ha='center')
a2.text(1.35,-0.50,r'weak threshold ($\tau<\frac{1}{2}$): decreasing',color=VERM,fontsize=8,ha='center')
a2.legend(loc='center right',fontsize=8,frameon=False,handlelength=1.7)
a2.set_xlabel(r'posterior toxicity SD  $\sigma_g$'); a2.set_ylabel(r'normalized cKG value  $\alpha_{\mathrm{KG}}/(h{-}b)$')
a2.set_xlim(0,3); a2.set_ylim(-0.58,0.58); a2.set_title('(b) one-step cKG',loc='left')
fig.tight_layout(); fig.savefig('fig1_mechanism.pdf',bbox_inches='tight'); plt.close(fig); print('fig1')

# ============ Figure (surfaces): OSA objective geometry ============
xs=np.linspace(0,1,160); Xg,Yg=np.meshgrid(xs,xs)
def surf(fn,z): return np.array([[fn(Xg[i,j],Yg[i,j],z) for j in range(Xg.shape[1])] for i in range(Xg.shape[0])])
fig,axs=plt.subplots(1,2,figsize=(7.2,3.25))
for z,ax,nm in zip((0,1),axs,('mild/moderate','severe')):
    E=-surf(OSA.f_osa,z); G=surf(OSA.g_osa,z); gd=OSA.G_DAGGER[z]; dopt,_=OSA.true_obd_osa(z)
    cf=ax.contourf(Xg,Yg,E,levels=14,cmap='viridis')
    ax.contourf(Xg,Yg,(G>gd).astype(float),levels=[0.5,1.5],colors='none',
                hatches=['////'],alpha=0)                      # hatch infeasible
    ax.contour(Xg,Yg,(G>gd).astype(float),levels=[0.5],colors='white',linewidths=0)
    bd=ax.contour(Xg,Yg,G,levels=[gd],colors='white',linewidths=2.2)
    ax.plot(*dopt,marker='*',color='red',ms=14,markeredgecolor='white',markeredgewidth=0.8,zorder=6)
    cb=fig.colorbar(cf,ax=ax,fraction=0.046,pad=0.03); cb.ax.tick_params(labelsize=7.5)
    cb.set_label(r'efficacy $-f(d,z)$',fontsize=7.5)
    ax.set_xlabel(r'dose $d_1$'); ax.set_title(f'$z={z}$ ({nm}),  $g^\\dagger={gd}$',fontsize=9.5)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.grid(False)
axs[0].set_ylabel(r'dose $d_2$')
# legend proxies
axs[1].legend([Line2D([0],[0],color='white',lw=2.2),
               Patch(facecolor='0.55',edgecolor='white',hatch='////'),
               Line2D([0],[0],marker='*',color='red',lw=0,ms=11)],
              [r'boundary $g=g^\dagger$','infeasible','OBD $d^*$'],
              loc='lower left',fontsize=7.5,facecolor='0.3',edgecolor='0.3',framealpha=0.92,labelcolor='white')
fig.tight_layout(); fig.savefig('fig_surfaces.pdf',bbox_inches='tight'); plt.close(fig); print('fig_surfaces')

# ============ Figure 2: precision crossover (OSA, n=1000) ============
d=json.load(open('results_faithful_cKGfix.json'))
def stat(pol,g,z):
    r=[x['dose_units'] for x in d if x['policy']==pol and abs(x['gamma']-g)<1e-9 and x['stratum']==z]
    return np.mean(r),np.std(r)/np.sqrt(len(r))
G=[0.5,0.7,0.8,0.9]; FLOOR={0:0.44,1:0.53}
fig,axs=plt.subplots(1,2,figsize=(7.0,3.0),sharey=True)
for z,ax in zip((0,1),axs):
    for lab,pol,c,mk,ls in [('cEI','cEI',VERM,'o','-'),('cKG','cKG1fix',BLUE,'s','--')]:
        m=[stat(pol,g,z) for g in G]
        ax.errorbar(G,[x[0] for x in m],yerr=[x[1] for x in m],marker=mk,color=c,ls=ls,label=lab,
                    capsize=2.5,markerfacecolor='white',markeredgewidth=1.3)
    ax.axhline(FLOOR[z],color=GRAY,ls=':',lw=1.0); ax.text(0.515,FLOOR[z]+0.04,'grid floor',color=GRAY,fontsize=7.5)
    ax.set_xlabel(r'gate strictness  $\gamma$'); ax.set_xticks(G); ax.set_xlim(0.46,0.94)
    ax.set_title(f'$z={z}$',fontsize=9.5)
axs[0].set_ylabel('dose-units to OBD  (lower better)'); axs[0].legend(frameon=False,loc='upper left',handlelength=2.2)
fig.tight_layout(); fig.savefig('fig2_precision.pdf',bbox_inches='tight'); plt.close(fig); print('fig2')

# ============ Figure 3 (phase boundary): clean single panel gamma*(kappa) w/ bootstrap CI + closed form ====
P=json.load(open('crossover_phase_fix.json'))
NOISE={'0.25x':0.25,'0.5x':0.5,'1.0x':1.0,'1.5x':1.5,'2.0x':2.0}; GAM=[0.5,0.6,0.7,0.8,0.9]
GRAD=2.436; SG0=1.29; SP=0.25; NSEED=100
def cross(gaps):
    for i in range(len(GAM)-1):
        if gaps[i]>0>=gaps[i+1]: return GAM[i]+gaps[i]/(gaps[i]-gaps[i+1])*(GAM[i+1]-GAM[i])
    return None
rng=np.random.default_rng(0)
ks=[]; gpt=[]; glo=[]; ghi=[]; cens=[]; lo=[]; hi=[]
for nl,k in NOISE.items():
    gp={g:np.array([P[f'{nl}|g{g}|cEI|{s}']-P[f'{nl}|g{g}|cKG1fix|{s}']
                    for s in range(NSEED) if f'{nl}|g{g}|cEI|{s}' in P and f'{nl}|g{g}|cKG1fix|{s}' in P]) for g in GAM}
    mean=[gp[g].mean() for g in GAM]; n=len(gp[GAM[0]])
    bs=[c for c in (cross([gp[g][rng.integers(0,n,n)].mean() for g in GAM]) for _ in range(2000)) if c is not None]
    pt=cross(mean); sg=k*SG0
    dkg=np.mean([P[f'{nl}|g0.5|cKG1fix|{s}'] for s in range(NSEED) if f'{nl}|g0.5|cKG1fix|{s}' in P])*SP
    dcei=np.mean([P[f'{nl}|g0.5|cEI|{s}'] for s in range(NSEED) if f'{nl}|g0.5|cEI|{s}' in P])*SP
    ks.append(k); lo.append(norm.cdf(dkg*GRAD/sg)); hi.append(norm.cdf(dcei*GRAD/sg))
    if pt is None: gpt.append(0.9); cens.append(True); glo.append(0); ghi.append(0)
    else: gpt.append(pt); cens.append(False); glo.append(pt-np.percentile(bs,2.5)); ghi.append(np.percentile(bs,97.5)-pt)
fig,(ax,axb,axc)=plt.subplots(1,3,figsize=(11.6,3.5))
# panel (a): crossover gamma* vs noise (form validation)
ax.fill_between(ks,lo,hi,color=BLUE,alpha=0.16,lw=0,label=r'closed form $\Phi(c\,|\nabla g|/\sigma_g)$')
ko=[k for k,c in zip(ks,cens) if not c]; go=[g for g,c in zip(gpt,cens) if not c]
elo=[e for e,c in zip(glo,cens) if not c]; ehi=[e for e,c in zip(ghi,cens) if not c]
ax.errorbar(ko,go,yerr=[elo,ehi],fmt='o-',color='black',ms=6,capsize=3,lw=1.6,label=r'empirical $\gamma^*$ (95% CI)')
for kk,gg in [(k,g) for k,g,c in zip(ks,gpt,cens) if c]:
    ax.annotate('',xy=(kk,0.965),xytext=(kk,gg),arrowprops=dict(arrowstyle='->',color='black',lw=1.4))
    ax.plot([kk],[gg],'o',mfc='white',mec='black',mew=1.4,ms=7); ax.text(kk+0.04,0.93,r'$\gamma^*>0.9$',fontsize=8)
ax.text(1.45,0.945,r'cEI more precise',fontsize=7.5,color=VERM,ha='center',va='center')
ax.text(1.55,0.555,r'cKG more precise',fontsize=7.5,color=BLUE,ha='center',va='center')
ax.set_xlabel(r'observation noise  $\kappa$  ($\times$ calibrated)'); ax.set_ylabel(r'crossover gate strictness  $\gamma^*$')
ax.set_ylim(0.5,1.0); ax.set_xlim(0.1,2.18); ax.legend(frameon=False,loc='lower left',fontsize=7.5)
ax.set_title(r'(a) crossover $\gamma^*$ vs noise',fontsize=9.5,loc='left')
# panel (b): the scale c = weak-gate GP recommendation error d0, decaying with budget N (~ N^-1/2)
S=json.load(open('dzero_scaling.json')); Ns=[16,32,64,128,256]
d0=np.array([S[str(n)]['d0'] for n in Ns]); d0se=np.array([S[str(n)]['d0_se'] for n in Ns])
sl=S['_fit']['d0_slope']; Nl=np.array([Ns[0],Ns[-1]],float)
axb.plot(Nl,d0[0]*(Nl/Ns[0])**sl,'--',color=VERM,lw=1.7,zorder=1,
         label=fr'fit $d^0\!\propto\!n^{{{sl:.2f}}}\ (\approx\!n^{{-1/2}})$')
axb.errorbar(Ns,d0,yerr=d0se,fmt='o',color='black',ms=6,capsize=3,zorder=2,label=r'measured $d^0=\|\hat d{-}d^*\|$')
axb.axvline(40,color='0.6',ls=':',lw=1.1,zorder=0); axb.text(43,d0.min()*1.18,'trial\n$N{=}40$',fontsize=7,color='0.4')
axb.set_xscale('log',base=2); axb.set_yscale('log')
axb.set_xticks(Ns); axb.set_xticklabels([str(n) for n in Ns])
axb.set_xlabel(r'surrogate fit size  $n$  (offline diagnostic)'); axb.set_ylabel(r'weak-gate error  $d^0$ (the scale $c$)')
axb.set_title(r'(b) the scale $c=d^0$ shrinks with data',fontsize=9.5,loc='left')
axb.legend(frameon=False,loc='upper right',fontsize=8)
# panel (c): boundary-offset geometry -- simple regret is LINEAR in d0 (R^2=0.98), not quadratic (0.70)
SR=np.array([S[str(n)]['SR'] for n in Ns]); SRse=np.array([S[str(n)]['SR_se'] for n in Ns])
def _r2(y,yh): return 1-np.sum((y-yh)**2)/np.sum((y-y.mean())**2)
mlin=np.sum(SR*d0)/np.sum(d0*d0); mq=np.sum(SR*d0**2)/np.sum(d0**4)
xx=np.linspace(0,d0.max()*1.10,100)
axc.plot(xx,mlin*xx,'-',color=VERM,lw=1.9,zorder=1,
         label=fr'linear $|\nabla_\perp e|\,d^0$  ($R^2={_r2(SR,mlin*d0):.2f}$)')
axc.plot(xx,mq*xx**2,'--',color=BLUE,lw=1.6,zorder=1,
         label=fr'quadratic $\propto(d^0)^2$  ($R^2={_r2(SR,mq*d0**2):.2f}$)')
axc.errorbar(d0,SR,xerr=d0se,yerr=SRse,fmt='o',color='black',ms=6,capsize=3,zorder=3,label=r'measured (per fit size $n$)')
axc.set_xlim(0,d0.max()*1.10); axc.set_ylim(0,SR.max()*1.14)
axc.set_xlabel(r'recommendation error  $d^0$'); axc.set_ylabel(r'efficacy shortfall  $e(d^*)\!-\!e(\hat d)$')
axc.set_title(r'(c) efficacy shortfall is linear in $d^0$',fontsize=9.5,loc='left')
axc.legend(frameon=False,loc='upper left',fontsize=7.5)
fig.tight_layout(); fig.savefig('fig3_phase.pdf',bbox_inches='tight'); plt.close(fig); print('fig3')

# ============ Historical Figure 4: gate calibration + final threshold status ============
def gsafe(pol,g,z):
    r=[x for x in d if x['policy']==pol and abs(x['gamma']-g)<1e-9 and x['stratum']==z]
    npass=np.array([x['n_gate_pass'] for x in r]); nsafe=np.array([x['n_gate_pass_safe'] for x in r])
    hp=npass>0; return 100*nsafe[hp].sum()/npass[hp].sum() if hp.any() else np.nan
def runsafe(pol,g,z):
    r=[x['rec_unsafe'] for x in d if x['policy']==pol and abs(x['gamma']-g)<1e-9 and x['stratum']==z]
    return 100*np.mean(r), 100*np.std(r)/np.sqrt(len(r))
fig,(c1,c2)=plt.subplots(1,2,figsize=(7.2,3.4))
for z,zls,zmk in [(0,'-','o'),(1,'--','s')]:
    c1.plot(G,[gsafe('cEI',g,z) for g in G],color=VERM,ls=zls,marker=zmk,mfc='white',mew=1.3)
    c1.plot(G,[gsafe('cKG1fix',g,z) for g in G],color=BLUE,ls=zls,marker=zmk,mfc='white',mew=1.3)
    for pol,col in [('cEI',VERM),('cKG1fix',BLUE)]:
        m=[runsafe(pol,g,z) for g in G]
        c2.errorbar(G,[x[0] for x in m],yerr=[x[1] for x in m],color=col,ls=zls,marker=zmk,
                    mfc='white',mew=1.3,capsize=2)
c1.set_xlabel(r'gate strictness  $\gamma$'); c1.set_ylabel('gate precision %  (higher better)')
c1.set_title('(a) gate calibration',loc='left'); c1.set_xticks(G); c1.set_xlim(0.46,0.94); c1.set_ylim(70,101)
c2.set_xlabel(r'toxicity-rule stringency $\tau$ (larger is stricter)'); c2.set_ylabel('above-threshold final recommendation %  (lower better)')
c2.set_title('(b) final recommendation threshold status',loc='left'); c2.set_xticks(G); c2.set_xlim(0.46,0.94); c2.set_ylim(0,55)
from matplotlib.lines import Line2D
leg=[Line2D([0],[0],color=VERM,lw=2.2,label='cEI'),Line2D([0],[0],color=BLUE,lw=2.2,label='cKG'),
     Line2D([0],[0],color='0.15',lw=1.3,ls='-',marker='o',mfc='white',label='$z{=}0$ (mild)'),
     Line2D([0],[0],color='0.15',lw=1.3,ls='--',marker='s',mfc='white',label='$z{=}1$ (severe)')]
c2.legend(handles=leg,frameon=False,ncol=2,fontsize=7.5,loc='upper right')
fig.tight_layout(); fig.savefig('fig4_safety.pdf',bbox_inches='tight'); plt.close(fig); print('fig4')

# ============ Figure (our surfaces): synthetic + demo objective geometry ============
import ckg_core as CK
def eff_demo(a,b,z):
    if z==0: return (3.2*np.exp(-((a-.3)**2/.03+(b-.7)**2/.1))+1.0*np.sin(2*np.pi*a)*np.cos(3*np.pi*b)
                     +1.4*a*np.exp(-1.2*a**2)+0.9*b*np.exp(-.6*b**2))
    return (2.0*np.exp(-((a-.5)**2+(b-.5)**2)/.05)+1.2*np.sin(3*np.pi*a)*np.sin(3*np.pi*b)
            +1.0*a*np.exp(-1.0*a**2)+0.8*b*np.exp(-.9*b**2))
def tox_demo(a,b,z):
    if z==0: return 3.0*((a-.5)**2+(b-.5)**2)+0.5*np.sin(3*np.pi*a)*np.sin(3*np.pi*b)+1.5*(a**2+b**2)
    return 4.0*((a-.5)**2+(b-.5)**2)+0.7*np.sin(3*np.pi*a)*np.sin(3*np.pi*b)+2.0*(a**2+b**2)
xs=np.linspace(0,1,150); Xg,Yg=np.meshgrid(xs,xs)
def panelsurf(ax,Efn,Gfn,title):
    E=np.array([[Efn(Xg[i,j],Yg[i,j]) for j in range(Xg.shape[1])] for i in range(Xg.shape[0])])
    G=np.array([[Gfn(Xg[i,j],Yg[i,j]) for j in range(Xg.shape[1])] for i in range(Xg.shape[0])])
    gd=float(np.median(G)); idx=np.unravel_index(np.where(G<=gd,E,-1e18).argmax(),E.shape)
    ox,oy=float(Xg[idx]),float(Yg[idx])
    cf=ax.contourf(Xg,Yg,E,levels=14,cmap='viridis')
    tl=[l for l in np.round(np.linspace(G.min(),G.max(),9),2) if abs(l-gd)>0.05*max(1,abs(gd))]
    ct=ax.contour(Xg,Yg,G,levels=tl,colors='#D55E00',linewidths=0.7,linestyles='--',alpha=0.9)
    ax.clabel(ct,inline=True,fontsize=5.5,fmt='%.1f')
    ax.contourf(Xg,Yg,(G>gd).astype(float),levels=[0.5,1.5],colors='none',hatches=['////'])
    ax.contour(Xg,Yg,G,levels=[gd],colors='white',linewidths=2.0)
    ax.plot(ox,oy,marker='*',color='red',ms=13,mec='white',mew=0.8,zorder=6)
    cb=fig.colorbar(cf,ax=ax,fraction=0.046,pad=0.03); cb.ax.tick_params(labelsize=7); cb.set_label('efficacy',fontsize=7)
    ax.set_xlabel(r'dose $d_1$'); ax.set_title(title,fontsize=8.5)
    ax.grid(False); ax.set_xlim(0,1); ax.set_ylim(0,1)
fig,axs=plt.subplots(1,2,figsize=(7.0,3.25))
panelsurf(axs[0],lambda a,b:eff_demo(a,b,0),lambda a,b:tox_demo(a,b,0),r'(a) $z{=}0$')
panelsurf(axs[1],lambda a,b:eff_demo(a,b,1),lambda a,b:tox_demo(a,b,1),r'(b) $z{=}1$')
axs[0].set_ylabel(r'dose $d_2$')
axs[1].legend([Line2D([0],[0],color='white',lw=2),
               Line2D([0],[0],color='#D55E00',lw=0.9,ls='--'),
               Patch(facecolor='0.55',edgecolor='white',hatch='////'),
               Line2D([0],[0],marker='*',color='red',lw=0,ms=10)],
              [r'boundary $g=g^\dagger$',r'toxicity $g$ contours','infeasible','OBD $d^*$'],
              loc='lower left',fontsize=6.3,facecolor='0.3',edgecolor='0.3',framealpha=0.92,labelcolor='white')
fig.tight_layout(); fig.savefig('fig_oursurfaces.pdf',bbox_inches='tight'); plt.close(fig); print('fig_oursurfaces')

# ============ Figure: Willard Eq. 3/4/5 operating characteristics (OSA + simulation study) ============
OSAd=json.load(open('results_faithful_cKGfix.json')); SIMd=json.load(open('results_demo_transfer.json'))
PC={'cEI':VERM,'cKG1fix':BLUE,'GBE':GREEN}; PL={'cEI':'cEI','cKG1fix':'cKG','GBE':'GBE'}
COLS=[('dose_units','dose-units (selection)'),('rpsel','RPSEL (efficacy estimate)'),('toxic','toxic exposure (in-trial)')]
ROWS=[('OSA simulator',OSAd,['cEI','cKG1fix']),('Gaussian-bump simulator',SIMd,['cEI','cKG1fix','GBE'])]
def _m(recs,pol,z,g,key):
    v=[r[key] for r in recs if r['policy']==pol and r['stratum']==z and abs(r['gamma']-g)<1e-9]
    return float(np.mean(v)) if v else np.nan
fig,axs=plt.subplots(2,3,figsize=(7.4,4.7))
for i,(rname,recs,pols) in enumerate(ROWS):
    gams=sorted(set(r['gamma'] for r in recs))
    for j,(key,clab) in enumerate(COLS):
        ax=axs[i,j]
        for pol in pols:
            for z,ls,mk in [(0,'-','o'),(1,'--','s')]:
                ax.plot(gams,[_m(recs,pol,z,g,key) for g in gams],color=PC[pol],ls=ls,
                        marker=mk,ms=3.5,lw=1.5)
        if i==0: ax.set_title(clab,fontsize=10)
        if i==1: ax.set_xlabel(r'gate strictness $\gamma$')
        if j==0: ax.set_ylabel(rname,fontsize=10,fontweight='bold')
        ax.margins(x=0.08)
# one shared legend for all six panels
_leg=[Line2D([],[],color=PC['cEI'],lw=2.6,label='cEI'),
      Line2D([],[],color=PC['cKG1fix'],lw=2.6,label='cKG'),
      Line2D([],[],color=PC['GBE'],lw=2.6,label='GBE'),
      Line2D([],[],color='0.15',ls='-',marker='o',ms=5,label=r'$z{=}0$'),
      Line2D([],[],color='0.15',ls='--',marker='s',ms=5,label=r'$z{=}1$')]
fig.legend(handles=_leg,loc='lower center',bbox_to_anchor=(0.5,0.045),ncol=5,frameon=False,
           fontsize=8.5,columnspacing=1.7,handlelength=1.9)
fig.text(0.5,0.008,r'Colour: policy (GBE on the Gaussian-bump testbed only).   Line style and marker: stratum $z$.   All criteria: lower is better.',
         ha='center',fontsize=7.5,style='italic')
fig.tight_layout(rect=[0,0.10,1,1]); fig.savefig('fig_criteria.pdf',bbox_inches='tight'); plt.close(fig); print('fig_criteria')
print('done')
