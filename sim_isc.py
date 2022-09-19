from brainiak.isc import isc, bootstrap_isc
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform
from brainiak.utils.utils import (array_correlation,
                                  phase_randomize,
                                  p_from_null,
                                  _check_timeseries_input)

import scipy

ntp=200
nsubj=20
nsim = 200

amp_sig = 0
amp_noise_mean = 2
amp_noise_std = 0


# Noise
allisc=[]
allp_sw=[]
allp_ew=[]
for simind in range(nsim):
    sig = scipy.signal.detrend(np.cumsum(2 * amp_sig * np.random.randn(ntp))) # Signal for a given experiment
    
    y=np.zeros((ntp, nsubj)) 
    for subjind in range(nsubj):
        subjnoise = amp_noise_mean + np.random.randn(1)*amp_noise_std # add noise for each subject (some better than others)
        y[:,subjind] = np.random.randn(ntp) * subjnoise + sig # signal same for all subjects
    simisc = isc(np.reshape(y, [ntp, 1, nsubj]), pairwise=True)
    if not simind:
        plt.figure()
        plt.imshow(squareform(simisc[:,0]))
        plt.colorbar()
        plt.savefig('exampleisc.jpg')
    
    observed, ci, p, distribution = bootstrap_isc(simisc, pairwise=True, n_bootstraps=200 )
    allp_sw.extend( p)

    observed, ci, p, distribution = bootstrap_isc(simisc, pairwise=False, n_bootstraps=200)
    allp_ew.extend( p)

fig, ax = plt.subplots(nrows=2)
ax[0].hist(allp_sw, bins=20)
ax[0].set_title('Subject wise permutation')
ax[1].hist(allp_ew, bins=20)
ax[1].set_title('Element wise permutation')
plt.savefig('allp.jpg')
print(f'SW false positive rate (alpha=0.05) {np.sum(np.array(allp_sw)<0.05)/nsim}')
print(f'EW false positive rate (alpha=0.05) {np.sum(np.array(allp_ew)<0.05)/nsim}')

#print(allp)
a=1
