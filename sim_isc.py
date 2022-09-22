from brainiak.isc import isc, bootstrap_isc, compute_summary_statistic
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform
from brainiak.utils.utils import (array_correlation,
                                  phase_randomize,
                                  p_from_null,
                                  _check_timeseries_input)
import pandas as pd


import scipy

ntp = 200
nsubj = 20
nsim = 50
norder = 6  # normally the voxels dimension, here I'm using it for our different orders

amp_noise_mean = 2
amp_noise_std = 1
# some orders have less signal and more noise
amp_order_factor = [1, 1, 1, 1.2, 0.8, 0.7]

# Noise
allisc = []
allp_sw = []
allp_combined_method1_sw = []
allp_combined_method2_sw = []
allp_ew = []
allmean_sw = []
allstd_sw = []
allmean_combined_method2_sw = []
allstd_combined_method2_sw = []

# Signal amplitude

df = pd.DataFrame()

for amp_sig in np.arange(0, 1, 0.25):

    for simind in range(nsim):
        y = np.zeros((ntp, norder, nsubj))
        for orderind in range(norder):
            # Signal for a given orderel (i.e., order)
            sig = amp_order_factor[orderind] * amp_sig * np.random.randn(ntp)
            for subjind in range(nsubj):
                # add noise for each subject (some better than others)
                subjnoise = amp_noise_mean + np.random.randn(1)*amp_noise_std
                y[:, orderind, subjind] = np.random.randn(
                    ntp) / amp_order_factor[orderind] * subjnoise + sig  # signal same for all subjects
        simisc = isc(y, pairwise=True)
        if not simind:
            plt.figure()
            plt.imshow(squareform(simisc[:, 0]))
            plt.colorbar()
            plt.savefig('exampleisc.jpg')

        observed, ci, p, distribution = bootstrap_isc(
            simisc, pairwise=True, n_bootstraps=200)
        #allp_sw.append( p)

        # Shift bootstrap distribution to 0 for hypothesis test
        shifted = distribution - observed

        # Get p-value for actual median from shifted distribution
        p = p_from_null(observed, shifted,
                        side='right', exact=False,
                        axis=0)
        allp_sw.append(p)
        allmean_sw.append(np.mean(distribution, axis=0))
        allstd_sw.append(np.std(distribution, axis=0))

        # Alternatively single p from combined data

        # METHOD 1 - average SHIFTED
        combined_p = p_from_null(np.mean(observed), np.mean(shifted, axis=1),
                                 side='right', exact=False,
                                 axis=0)
        allp_combined_method1_sw.append(combined_p)

        if not simind:
            plt.figure()
            plt.hist(shifted.ravel(), bins=50)
            plt.title(
                f'Example null distribution (observed={np.mean(observed)})')
            plt.savefig(f'null_dist_method1.jpg')

        # METHOD 2 - average DISTRIBUTION
        distribution = np.mean(np.array(distribution), axis=1)
        # Shift bootstrap distribution to 0 for hypothesis test
        observed2 = np.mean(
            compute_summary_statistic(simisc, 'median', axis=0))
        shifted = distribution - observed2
        combined_p = p_from_null(observed2, shifted,
                                 side='right', exact=False,
                                 axis=0)
        allp_combined_method2_sw.append(combined_p)
        allmean_combined_method2_sw.append(np.mean(distribution))
        allstd_combined_method2_sw.append(np.std(distribution))

        if not simind:
            plt.figure()
            plt.hist(shifted.ravel(), bins=50)
            plt.title(f'Example null distribution (observed={observed2})')
            plt.savefig(f'null_dist_method2.jpg')

    print(f'signal {amp_sig}')
    print(
        f' SW individual false positive rate (alpha=0.05) {np.mean(np.array(allp_sw)<0.05, axis=0)}')
    print(
        f' SW combined method 1 false positive rate (alpha=0.05) {np.mean(np.array(allp_combined_method1_sw)<0.05, axis=0)}')
    print(
        f' SW combined method 2 false positive rate (alpha=0.05) {np.mean(np.array(allp_combined_method2_sw)<0.05, axis=0)}')

    df = df.append({'amp_sig': amp_sig, 
        'fpr_ind': np.mean(np.array(allp_sw) < 0.05, axis=0), 
        'fpr_comb1': np.mean(np.array(allp_combined_method1_sw) < 0.05, axis=0), 
        'fpr_comb2': np.mean(np.array(allp_combined_method2_sw) < 0.05, axis=0), 
        'allmean_sw': np.mean(np.array(allmean_sw), axis=0), 
        'allstd_sw': np.mean(np.array(allstd_sw), axis=0), 
        'allmean_combined_method2_sw': np.mean(np.array(allmean_combined_method2_sw), axis=0), 
        'allstd_combined_method2_sw': np.mean(np.array(allstd_combined_method2_sw), axis=0), 
        }, ignore_index=True)

df.to_csv('fpr_by_signal.csv')
plt.figure()
for ind in range(norder):
    plt.plot(df['amp_sig'], [x[ind] for x in df['fpr_ind']], color='gray')
plt.plot('amp_sig', 'fpr_comb1', 'bx', data=df)
plt.plot('amp_sig', 'fpr_comb2', 'r+', data=df)
plt.xlabel('Amplitude of signal')
plt.ylabel('Probability of rejecting H0')
plt.savefig('fpr_by_signal.jpg')

# Check mean against analytic calc
plt.figure()
for ind in range(norder):
    plt.plot(df['amp_sig'], [x[ind] for x in df['allmean_sw']], color='gray')
plt.plot(df['amp_sig'], [np.mean(x) for x in df['allmean_sw']], 'go')
plt.plot('amp_sig', 'allmean_combined_method2_sw', 'r+', data=df)

plt.legend('Predicted','method2')
plt.xlabel('Amplitude of signal')
plt.ylabel('Mean')
plt.savefig('mean_by_signal.jpg')

# Check standard dev against analytic calc
plt.figure()
for ind in range(norder):
    plt.plot(df['amp_sig'], [x[ind] for x in df['allstd_sw']], color='gray')
plt.plot(df['amp_sig'], [np.sqrt(np.sum(x**2))/norder for x in df['allstd_sw']], 'go')
plt.plot('amp_sig', 'allstd_combined_method2_sw', 'r+', data=df)

plt.legend('Predicted','method2')
plt.xlabel('Amplitude of signal')
plt.ylabel('Standard deviation')
plt.savefig('std_by_signal.jpg')

# print(allp)
a = 1
