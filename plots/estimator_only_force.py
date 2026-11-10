"""Plot current contact reconstruction with prescribed MPM robot and particle motion."""
from current_common import G1, MPM, NERD, np, plt, read, save

folder=G1/'data/estimator-only-20261109'
meta=read(folder/'metadata.json')
data=np.load(folder/'trace.npz')
assert data['forces'].shape==data['reference_forces'].shape==(1000,2,6)
error=data['forces'][...,:3]-data['reference_forces'][...,:3]
assert np.isclose(np.sqrt(np.mean(np.sum(error**2,axis=-1))),meta['force_vector_rmse_N'])
fig,axes=plt.subplots(3,2,figsize=(14,12),sharex=True,sharey='row')
for component in range(3):
    for foot in range(2):
        ax=axes[component,foot]
        ax.plot(data['time'],data['reference_forces'][:,foot,component],color=MPM,lw=1.4,label='Implicit MPM')
        ax.plot(data['time'],data['forces'][:,foot,component],color=NERD,lw=1.2,ls='--',label='NeRD force estimator · prescribed particles')
        ax.set_ylabel(f"$F_{'xyz'[component]}$ (N)")
        ax.set_xlim(0,5)
        ax.grid(alpha=.15)
for ax in axes[-1]:ax.set_xlabel('Elapsed replay time (s)')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
fig.subplots_adjust(top=.93,bottom=.07,left=.08,right=.99,hspace=.25,wspace=.16)
save(fig,'force_estimation','estimator_only_force')
