"""Historical diagnostic helper; not a source of current manuscript claims."""
import numpy as np, torch, warnings; warnings.filterwarnings('ignore'); torch.set_default_dtype(torch.double)
import gate_ab_smoke as G
def fp(pol, g=0.7, z=1, N=50):
    fps=[]
    for s in range(N):
        r=G.run_trial(pol, s, z, g, 'osa', 'latent'); fps.append(r['n_gate_pass']-r['n_gate_pass_safe'])
    return float(np.mean(fps)), float(np.std(fps)/np.sqrt(N))
with open('control_panel.log','w') as fo:
    fo.write("CONTROL PANEL false-pass/trial, OSA gamma=0.7 severe stratum z=1, latent, n=50:\n")
    for pol,lab in [('cEI','cEI (gates, samples interior)'),('ustrad','ungated all-boundary sampler'),
                    ('straddle','gated all-boundary sampler'),('GBE','GBE (lookahead-free)'),('cKG1fix','gated cKG')]:
        m,se=fp(pol); fo.write(f"  {lab:34s}: {m:.3f} (SE {se:.3f})\n"); fo.flush()
    fo.write("done\n")
