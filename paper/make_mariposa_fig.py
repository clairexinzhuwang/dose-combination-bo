"""Aggregate-informed synthetic MARIPOSA calibration.

Efficacy (% AHI reduction) is a filled heatmap and the assumed continuous toxicity
score is overlaid with contour lines. Thick white is the synthetic threshold, hatching
marks its infeasible side, and the red star is the synthetic dense-grid target. Its
proximity to the published 2.5/75 mg reference arm is not patient-level validation or
an estimate of a clinical OBD.
"""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import gate_ab_smoke as G
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':10,'axes.labelsize':10,
    'xtick.labelsize':9,'ytick.labelsize':9,'savefig.dpi':300})
S=G.sim_spec('mariposa',0); eff=S['eff']; tox=S['tox']; gd=S['gd']; dopt=np.asarray(S['dopt'])
xs=np.linspace(0,1,220); Xg,Yg=np.meshgrid(xs,xs)
E=np.array([[eff(Xg[i,j],Yg[i,j]) for j in range(220)] for i in range(220)])
T=np.array([[tox(Xg[i,j],Yg[i,j]) for j in range(220)] for i in range(220)])
X,Y=Xg*5,Yg*75
fig,ax=plt.subplots(figsize=(5.0,4.0))
cf=ax.contourf(X,Y,E,levels=14,cmap='viridis')                                   # efficacy heatmap
tl=[l for l in np.round(np.linspace(T.min(),T.max(),9),2) if abs(l-gd)>0.05]     # toxicity contour lines
ct=ax.contour(X,Y,T,levels=tl,colors='#D55E00',linewidths=0.8,linestyles='--',alpha=0.9)  # toxicity gradient
ax.clabel(ct,inline=True,fontsize=6,fmt='%.1f')
ax.contourf(X,Y,(T>gd).astype(float),levels=[0.5,1.5],colors='none',hatches=['////'])
ax.contour(X,Y,T,levels=[gd],colors='white',linewidths=2.4)                       # g=g-dagger boundary
ax.plot(dopt[0]*5,dopt[1]*75,marker='*',color='red',ms=15,mec='white',mew=0.9,zorder=6)
cb=fig.colorbar(cf,ax=ax,fraction=0.046,pad=0.03); cb.set_label('efficacy: AHI reduction (\\%)',fontsize=8); cb.ax.tick_params(labelsize=7)
ax.set_xlabel('aroxybutynin (mg)'); ax.set_ylabel('atomoxetine (mg)')
ax.legend([Line2D([0],[0],color='white',lw=2.4),Line2D([0],[0],color='#D55E00',lw=1.0,ls='--'),
           Patch(facecolor='0.55',edgecolor='white',hatch='////'),Line2D([0],[0],marker='*',color='red',lw=0,ms=11)],
          ['boundary $g{=}g^\\dagger$','toxicity $g$ contours','infeasible','synthetic target'],loc='lower right',fontsize=6.6,
          facecolor='0.3',edgecolor='0.3',framealpha=0.92,labelcolor='white')
fig.tight_layout(); fig.savefig('fig_mariposa.pdf',bbox_inches='tight'); print('saved fig_mariposa.pdf overlay')
