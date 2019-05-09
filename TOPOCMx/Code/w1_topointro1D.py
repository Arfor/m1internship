import sys
sys.path.append('../code')
from init_mooc_nb import *
init_notebook()

#%output size=150
dims = SimpleNamespace(E_t=holoviews.Dimension(r'$E/t$'),
                       mu_t=holoviews.Dimension(r'$\mu/t$'),
                       lambda_=holoviews.Dimension(r'$\lambda$'),
                       x=holoviews.Dimension(r'$x$'),
                       k=holoviews.Dimension(r'$k$'),
                       amplitude=holoviews.Dimension(r'$|u|^2 + |v|^2$'))

holoviews.core.dimension.title_format = ''

def kitaev_chain(L=None, periodic=False):
    lat = kwant.lattice.chain()

    if L is None:
        sys = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
        L = 1
    else:
        sys = kwant.Builder()

    # transformation to antisymmetric basis
    U = np.array([[1.0, 1.0], [1.j, -1.j]]) / np.sqrt(2)

    def onsite(onsite, p): 
        return - p.mu * U.dot(pauli.sz.dot(U.T.conj()))
    
    for x in range(L):
        sys[lat(x)] = onsite

    def hop(site1, site2, p):
        return U.dot((-p.t * pauli.sz - 1j * p.delta * pauli.sy).dot(U.T.conj()))
    
    sys[kwant.HoppingKind((1,), lat)] = hop

    if periodic:
        def last_hop(site1, site2, p):
            return hop(site1, site2, p) * (1 - 2 * p.lambda_)
        
        sys[lat(0), lat(L - 1)] = last_hop
    return sys


def bandstructure(mu, delta=1, t=1, Dirac_cone="Not show", show_pf=False):
    sys = kitaev_chain(None).finalized()
    p = SimpleNamespace(t=t, delta=delta, mu=mu)
    plot = holoviews.Overlay([plot_bands(sys, p)[-4:4]])
    h_1 = h_k(sys, p, 0)
    h_2 = h_k(sys, p, np.pi)
    pfaffians = [find_pfaffian(h_1), find_pfaffian(h_2)]
    
    if show_pf:
        signs = [('>' if pf > 0 else '<') for pf in pfaffians]
        title = "$\mu = {mu} t$, Pf$(iH_{{k=0}}) {sign1} 0$, Pf$(iH_{{k=\pi}}) {sign2} 0$"
        title = title.format(mu=mu, sign1=signs[0], sign2=signs[1])
        plot *= holoviews.VLine(0) * holoviews.VLine(-np.pi)
    else:
        if pfaffians[0] * pfaffians[1] < 0:
            title = "$\mu = {mu} t$, topological ".format(mu=mu)
        else:
            title = "$\mu = {mu} t$, trivial ".format(mu=mu)
        
    if Dirac_cone == "Show":
        ks = np.linspace(-np.pi, np.pi)
        ec = np.sqrt((mu + 2 * t)**2 + 4.0 * (delta * ks)**2)
        plot *= holoviews.Path((ks, ec), kdims=[dims.k, dims.E_t])(style={'linestyle':'--', 'color':'r'})
        plot *= holoviews.Path((ks, -ec), kdims=[dims.k, dims.E_t])(style={'linestyle':'--', 'color':'r'})
    return plot.relabel(title)

def find_pfaffian(H):
    return np.sign(np.real(pf.pfaffian(1j*H)))
	
#%%
def plot_wf(sys, wf1, wf2, lstyle='-', lcolor='b'):
    L = sys.graph.num_nodes
    xs = np.array([sys.sites[i].pos[0] for i in range(sys.graph.num_nodes)])
    indx = np.argsort(xs)
    wf_sq = np.linalg.norm(wf1.reshape(-1, 2), axis=1)**2 + np.linalg.norm(wf2.reshape(-1, 2), axis=1)**2
    plot = holoviews.Path((xs[indx], wf_sq[indx]), kdims=[dims.x, dims.amplitude])
    return plot(style={'linestyle':lstyle, 'color':lcolor}, plot={'yticks':0, 'xticks':list(range(0, L, 10))})

def plot_pwave(L, t, delta, mu):
    # At mu=0 the first exited state is not well defined due to the massive degeneracy.
    # That is why we add a small offset to mu.
    sys = kitaev_chain(L).finalized()
    p = SimpleNamespace(t=t, delta=delta, mu=mu+1e-4)
    ham = sys.hamiltonian_submatrix(args=[p])
    ev, evec = np.linalg.eigh(ham)
    return (plot_wf(sys, evec[:, L], evec[:, L-1]) *
            plot_wf(sys, evec[:, L+1] , evec[:, L-2], lstyle='--', lcolor='r'))

sys = kitaev_chain(L=25).finalized()
mus = np.arange(0, 4, 0.2)

(spectrum(sys, t=1, delta=1, mu=mus, xticks=[0, 1, 2, 3], yticks=range(-3, 4), 
          xdim=dims.mu_t, ydim=dims.E_t) * 
 holoviews.HoloMap({mu: holoviews.VLine(mu) for mu in mus}, kdims=[dims.mu_t]) +
 holoviews.HoloMap({mu: plot_pwave(25, 1.0, 1.0, mu) for mu in mus}, kdims=[dims.mu_t]))
 
 #%%
 
mus = np.arange(-3, 3, 0.25)

plots = {(mu, Dirac_cone): bandstructure(mu, Dirac_cone=Dirac_cone) 
         for mu in mus 
         for Dirac_cone in ["Show", "Not show"]}

holoviews.HoloMap(plots, kdims=[dims.mu_t, "Dirac cone"])

#%%


mus = np.arange(-3, 3, 0.25)
holoviews.HoloMap({mu: bandstructure(mu, show_pf=True) for mu in mus}, kdims=[dims.mu_t])

#%%

sys = kitaev_chain(L=25, periodic=True).finalized()

kwargs = dict(t=1, delta=1, lambda_=np.linspace(0.0, 1.0), 
              xticks=np.linspace(0, 1, 5), yticks=np.linspace(-4, 4, 5),
              xdim=dims.lambda_, ydim=dims.E_t)

mus = np.arange(0, 4, 0.1)
holoviews.HoloMap({mu: spectrum(sys, mu=mu, **kwargs) for mu in mus}, kdims=[dims.mu_t])