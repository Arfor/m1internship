import numpy as np 
from matplotlib import pyplot as plt
import kwant

dims = dict(kx = 'k_x',ky = 'k_y', mu = 'µ', delta = 'Δ', t = 't', E='ε',mu_t = 'µ/t',E_d='ε/Δ', \
			azi_winding='q', radi_winding= 'p', radius = 'r', R0 = 'R_0', j='j')
#
def rebuild_sys(params, shape= 'disk', plot=False):
	'''
	For now only disk-shape possible
	'''
	
	sys = kwant.Builder() #initialize your system
	sqlat = kwant.lattice.square(norbs=1)

	#define a Boolean function to shape your system

	def disk(position): 
		x,y = position
		return x**2 + y**2 < params['radius']**2

	if shape == 'disk':
		sys[sqlat.shape(disk,(0,0))]= 1

	if plot:
		kwant.plot(sys)
		plt.close()

	return sys.finalized()
#
def plot_spec(eVal, params, save_plot_to = None):
	fig = plt.figure()
	plt.plot(eVal/params['delta'],'.')
	plt.xlim(0,len(eVal))
	plt.suptitle('Spectrum (k={})'.format(len(eVal)))
	plt.ylabel('$\epsilon/\Delta$', fontsize=14)

	pars = params.copy()
	param_text= '\n'

	for key in pars:
		param_text = param_text + ' {} = {} \n'.format(dims[key], pars[key])
		# place text boxes
	plt.text(1.05, 0.5, param_text, fontsize=14, horizontalalignment='left',transform = plt.gca().transAxes,\
		 verticalalignment='center', bbox=dict(facecolor='grey', alpha=0.1))

	if not save_plot_to==None:
		if not isinstance(save_plot_to, str):
			raise Exception('Path has to be a string')
		plt.savefig(save_plot_to + '\spectrum.png', dpi=250, bbox_inches='tight')        
	
	plt.close()
	
	return fig
#
def plot_dos(eVal, params, save_plot_to=None):
	fig = plt.figure()
	plt.hist(eVal/params['delta'],bins=20, density=True)
	plt.suptitle('DoS (k={})'.format(len(eVal)))
	plt.xlabel('$\epsilon/\Delta$', fontsize=14)

	pars = params.copy()
	param_text= '\n'

	for key in pars:
		param_text = param_text + ' {} = {} \n'.format(dims[key], pars[key])
		# place text boxes
	plt.text(1.05, 0.5, param_text, fontsize=14, horizontalalignment='left',transform = plt.gca().transAxes,\
		verticalalignment='center', bbox=dict(facecolor='grey', alpha=0.1))

	if not save_plot_to==None:
		if not isinstance(save_plot_to, str):
			raise Exception('Path has to be a string')
		plt.savefig(save_plot_to + '\DoS.png', dpi=250, bbox_inches='tight')        
	plt.close()	
	return fig
	


import holoviews as hv
hv.extension('bokeh', logo=False)
#
def plot_dos_hv(eVal, params, smooth=False, toolbar=False, save_plot_to=None, n_bins=20):
	plot_opts=dict(width=350, height=350, invert_axes=True, framewise=True, \
	default_tools=['ypan','wheel_zoom','reset', 'save'])
	style_opts=dict(color='tomato',alpha=0.5)

	if not toolbar:
		plot_opts['toolbar']='disable'
		
	fermi_level = hv.VLine(params['mu']).opts(**plot_opts, color='black', line_dash='dashed', line_width=1.5)
	if smooth == True:
		bandwidth =  0.09
		plot_opts['bandwidth'] = bandwidth
		hist = hv.Distribution(eVal/params['delta'], kdims=dims['E_d']).opts(plot=plot_opts, style=style_opts)
	else:
		hist = hv.Histogram(np.histogram(eVal/params['delta'], bins=n_bins, density=True), \
		kdims=dims['E_d'], vdims='Density').opts(plot=plot_opts, style=style_opts)
	plot = (hist*fermi_level)
	
	if not save_plot_to == None:
		if not isinstance(save_plot_to, str):
			raise Exception('Path has to be a string')
		hv.save(plot, save_plot_to + '\DoS_hv.png') 
		
	return plot.opts(title='DoS (k={:d})'.format(len(eVal)))
#
def plot_wf(sys, eigen, params, n=1, save_plot_to = None):

	#sys = Kwant system, eigen=(eVal, eVec), n= number of wavefunction to plot
	eVal, eVec = eigen
	prob_dens = np.abs(eVec)**2

	vmax=np.max(prob_dens[:,n])

	fig, axes = plt.subplots(2,2, sharey=True, sharex=True, figsize=(6,6))

	plt.suptitle('Wavefunction to $\epsilon$={:.3E}'.format(eVal[n]), y=1,fontsize=16)

	axes[0,0].set_title('$c^{\\dagger}_{\\uparrow}$')
	kwant.plotter.density(sys, prob_dens[0::4, n], vmin=0, vmax=vmax, cmap='magma', ax = axes[0,0], background='#000000')
	axes[0,0].text(0.025, 1.05,'|ψ|={:.2f}'.format(np.sum(prob_dens[0::4, n])), \
		 transform = axes[0,0].transAxes, bbox=dict(facecolor='white', alpha=0.5))

	axes[0,1].set_title('$c^{\\dagger}_{\\downarrow}$')
	kwant.plotter.density(sys, prob_dens[1::4, n], vmin=0, vmax=vmax, ax=axes[0,1], cmap='inferno',background='#000000')
	axes[0,1].text(0.025, 1.05,'|ψ|={:.2f}'.format(np.sum(prob_dens[1::4, n])), \
		 transform = axes[0,1].transAxes, bbox=dict(facecolor='white', alpha=0.5))

	axes[1,0].set_title('$c^{}_{\\uparrow}$')
	kwant.plotter.density(sys, prob_dens[3::4, n], vmin=0, vmax=vmax, ax=axes[1,0], cmap='inferno',background='#000000')
	axes[1,0].text(0.025, 1.05,'|ψ|={:.2f}'.format(np.sum(prob_dens[3::4, n])), \
		 transform = axes[1,0].transAxes, bbox=dict(facecolor='white', alpha=0.5))

	axes[1,1].set_title('$c^{}_{\\downarrow}$')
	kwant.plotter.density(sys, prob_dens[2::4, n], vmin=0, vmax=vmax, ax=axes[1,1], cmap='inferno',background='#000000')
	axes[1,1].text(0.025, 1.05,'|ψ|={:.2f}'.format(np.sum(prob_dens[2::4, n])), \
		 transform = axes[1,1].transAxes, bbox=dict(facecolor='white', alpha=0.5))

	pars = params.copy()
	param_text= '\n'

	for key in pars:
		param_text = param_text + ' {} = {} \n'.format(dims[key], pars[key])    
	# place text boxes
	fig.text(0.95, 0.5, param_text, fontsize=14, horizontalalignment='left',\
		 verticalalignment='center', bbox=dict(facecolor='grey', alpha=0.1))

	
	if not save_plot_to==None:
		if not isinstance(save_plot_to, str):
			raise Exception('Path has to be a string')
		plt.savefig(save_plot_to + '\wf{}__e={:.2E}.png'.format(n,eVal[n]), bbox_inches='tight', dpi=50, metadata=params)     
	
	plt.close()
	return fig
	




#





#