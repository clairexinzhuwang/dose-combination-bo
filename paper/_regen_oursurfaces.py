"""Render the Gaussian-bump response-geometry figure used in the Supplement."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':10,'axes.labelsize':10,
    'axes.titlesize':10,'xtick.labelsize':9,'ytick.labelsize':9,'savefig.dpi':300})
def eff_demo(a,b,z):
    if z==0: return (3.2*np.exp(-((a-.3)**2/.03+(b-.7)**2/.1))+1.0*np.sin(2*np.pi*a)*np.cos(3*np.pi*b)
                     +1.4*a*np.exp(-1.2*a**2)+0.9*b*np.exp(-.6*b**2))
    return (2.0*np.exp(-((a-.5)**2+(b-.5)**2)/.05)+1.2*np.sin(3*np.pi*a)*np.sin(3*np.pi*b)
            +1.0*a*np.exp(-1.0*a**2)+0.8*b*np.exp(-.9*b**2))
def tox_demo(a,b,z):
    if z==0: return 3.0*((a-.5)**2+(b-.5)**2)+0.5*np.sin(3*np.pi*a)*np.sin(3*np.pi*b)+1.5*(a**2+b**2)
    return 4.0*((a-.5)**2+(b-.5)**2)+0.7*np.sin(3*np.pi*a)*np.sin(3*np.pi*b)+2.0*(a**2+b**2)
xs=np.linspace(0,1,80); Xg,Yg=np.meshgrid(xs,xs)
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
    cb=fig.colorbar(cf,ax=ax,fraction=0.046,pad=0.03); cb.ax.tick_params(labelsize=7); cb.set_label('True mean efficacy',fontsize=7)
    ax.set_xlabel(r'Standardized dose of agent 1, $d_1$'); ax.set_title(title,fontsize=8.5)
    ax.grid(False); ax.set_xlim(0,1); ax.set_ylim(0,1)
fig,axs=plt.subplots(1,2,figsize=(7.0,3.25))
panelsurf(axs[0],lambda a,b:eff_demo(a,b,0),lambda a,b:tox_demo(a,b,0),r'(a) Stratum $z{=}0$')
panelsurf(axs[1],lambda a,b:eff_demo(a,b,1),lambda a,b:tox_demo(a,b,1),r'(b) Stratum $z{=}1$')
axs[0].set_ylabel(r'Standardized dose of agent 2, $d_2$')
axs[1].legend([Line2D([0],[0],color='white',lw=2),Line2D([0],[0],color='#D55E00',lw=0.9,ls='--'),
               Patch(facecolor='0.55',edgecolor='white',hatch='////'),Line2D([0],[0],marker='*',color='red',lw=0,ms=10)],
              [r'True mean toxicity limit $g=g^\dagger$',r'True mean toxicity contours','True mean toxicity above limit','Continuous-domain efficacy maximum'],
              loc='lower left',fontsize=6.3,facecolor='0.3',edgecolor='0.3',framealpha=0.92,labelcolor='white')
output_path=Path(__file__).resolve().parent/'source'/'fig_oursurfaces.pdf'
fixed_date=datetime(2000,1,1,tzinfo=timezone.utc)
fig.tight_layout(); fig.savefig(output_path,bbox_inches='tight',metadata={
    'Title':'Gaussian-bump response geometry',
    'Author':'dual-combo-BO',
    'Creator':'dual-combo-BO',
    'Producer':'Matplotlib',
    'CreationDate':fixed_date,
    'ModDate':fixed_date,
}); plt.close(fig); print(f'saved {output_path}')
