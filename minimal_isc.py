from brainiak.isc import isc, bootstrap_isc
import numpy as np
from brainiak.utils.utils import (array_correlation,
                                  phase_randomize,
                                  p_from_null,
                                  _check_timeseries_input)

import scipy

ntp = 200   # number of timepoints
nsubj = 20  # number of subjects
nsim = 200  # number of simulations
nvox = 10 # number of voxels
alpha = 0.05    # expected false positive level

allp_sw = []  # subject-wise
allp_ew = []  # element-wise

for simind in range(nsim):
    y = np.random.randn(ntp, nvox, nsubj)  # no signal, just null distribution
    simisc = isc(y, pairwise=True)  # get isc matrix

    # subject-wise
    observed, ci, p, distribution = bootstrap_isc(
        simisc, pairwise=True, n_bootstraps=1000)
    allp_sw.extend(p)

    # element-wise
    observed, ci, p, distribution = bootstrap_isc(
        simisc, pairwise=False, n_bootstraps=1000)   
    allp_ew.extend(p)

print(f'SW false positive rate (alpha={alpha}) {np.mean(np.array(allp_sw)<alpha)}')
print(f'EW false positive rate (alpha={alpha}) {np.mean(np.array(allp_ew)<alpha)}')

