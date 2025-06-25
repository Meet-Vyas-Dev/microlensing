# Filter for microlensing events designed for the ANTARES broker

# Authors: Somayeh Khakpash, Natasha Abrams, Rachel Street

import antares
import antares.devkit as dk
from statsmodels.stats.weightstats import DescrStatsW
import numpy as np
from astropy.table import MaskedColumn
import warnings
import astropy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import skew

# Initialize development kit client
dk.init()

# Define a Paczyński microlensing model
def paczynski(t, t0, u0, tE, F_s):
    """
    Paczyński microlensing light curve model
    t0 : peak time
    u0 : impact parameter
    tE : Einstein crossing time
    F_s : source flux
    F_b : blended flux
    """
    u = np.sqrt(u0 ** 2 + ((t - t0) / tE) ** 2)
    A = (u ** 2 + 2) / (u * np.sqrt(u ** 2 + 4))
    return F_s * (A - 1) + (1 - F_s)


def mag_to_flux(mag, F0=1.0):
    """
    Convert magnitude to flux.

    Parameters:
    - mag : magnitude (float or array)
    - F0 : reference flux (zeropoint), default=1.0 for relative flux

    Returns:
    - flux : flux corresponding to the magnitude
    """
    flux = F0 * 10 ** (-0.4 * mag)
    flux = flux / np.min(flux)
    return flux


def magerr_to_fluxerr(mag, mag_err, F0=1.0):
    """
    Convert magnitude uncertainty to flux uncertainty.

    Parameters:
    - mag : magnitude value or array
    - mag_err : magnitude uncertainty value or array
    - F0 : zeropoint flux (default=1.0 for relative flux)

    Returns:
    - flux_err : flux uncertainty
    """
    flux = mag_to_flux(mag, F0)
    flux_err = 0.4 * np.log(10) * flux * mag_err
    return flux_err


class microlensing(dk.Filter):    
    INPUT_LOCUS_PROPERTIES = [
        'ztf_object_id',
    ]
    
    OUTPUT_TAGS = [
        {
            'name': 'microlensing_candidate',
            'description': 'Locus - a transient candidate - exhibits a microlensing-like variability',
        }
    ]


    def make_lc(self, locus):

        with warnings.catch_warnings():
            # The cast of locus.timeseries: astropy.table.Table to a pandas
            # dataframe results in the conversion of some integer-valued
            # columns to floating point represntation. This can result in a
            # number of noisy warning so we will catch & ignore them for the
            # next couple of lines.
            warnings.simplefilter("ignore", astropy.table.TableReplaceWarning)
            df = locus.timeseries.to_pandas()

        data = df[['ant_mjd', 'ztf_fid', 'ztf_magpsf', 'ztf_sigmapsf']]
        
        dn = data.dropna()
        times=dn['ant_mjd'][dn['ztf_fid']==1]
        mags = dn['ztf_magpsf'][dn['ztf_fid']==1]
        mags_err = dn['ztf_sigmapsf'][dn['ztf_fid']==1]
        flxs = mag_to_flux(mags)
        flx_errs = magerr_to_fluxerr(mags, mags_err)
        
        t0_guess = times[np.argmin(mags)]  # Min mag is peak time
        u0_guess = 1/(np.max(flxs))

        initial_fit  = paczynski(times,
                                     t0_guess, 
                                     u0_guess, 
                                     20, 
                                     0.5)

        # plt.gca().invert_yaxis()
        plt.scatter(times, 
                    flxs, color='g', label='g_band')
        plt.plot(times, 
                 initial_fit, color='b', label='initial fit')
        
        plt.xlabel('Time (mjd)')
        plt.ylabel('Flux')
        plt.legend()


    def is_microlensing_candidate(self, times, mags, errors):
        """
        Example of a set of Microlensing detection criteria
        """
        if len(times) < 10:  # Too few data points
            return False

        # TODO - Rache add variability flag if relevant - maybe based on historical
        # i.e. check his

        # Sort data by time
        sorted_idx = np.argsort(times)
        times, mags, errors = times[sorted_idx], mags[sorted_idx], errors[sorted_idx]

        # 1. Check for smoothness (low skewness means symmetric light curve)
        # TODO: Check for threshold with parallax and maybe remove or lower threshold
        if abs(skew(mags)) > 1:
            return False

        # TODO - Natasha Add von Neumann parameter

        # 2. Check variability (microlensing should have a clear peak)
        # Decrease threshold with longer baseline
        # Q for broker - 365 days or full lightcurve?
        if np.ptp(mags) < 0.5:  # Peak-to-peak magnitude difference
            return False

        flxs = mag_to_flux(mags)
        flx_errs = magerr_to_fluxerr(mags, errors)

        # 3. Perform a lightweight template fit (Paczyński model)
        # TODO - Somayeh switch to KMTNet algorithm
        t0_guess = times[np.argmin(mags)]  # Min mag is peak time
        u0_guess = 1/(np.max(flxs))
        initial_guess = [t0_guess, 
                         u0_guess, 
                         20, 
                         0.5]  # Initial params

        # try:
        popt, _ = curve_fit(paczynski, times, flxs, p0=initial_guess, sigma=flx_errs)
        chi2 = np.sum(((flxs - paczynski(times, *popt)) / flx_errs) ** 2) / len(times)

        # 4. Apply a simple chi2 threshold
        if chi2 < 2:  # Well-fit light curves pass
            return True
        # except RuntimeError:
        #     return False  # Fit failed

        # TODO - Natasha add von Neumann residual (subtract out microlensing and see if you're still correlated)

        # TODO - Rache potentially query full lightcurve if not already there and if possible

        # TODO - Natasha add parallax microlensing fit

        return False
    def run(self, locus):
        print('Processing Locus:', locus.locus_id)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", astropy.table.TableReplaceWarning)
            df = locus.timeseries.to_pandas()

        data = df[['ant_mjd', 'ztf_fid', 'ztf_magpsf', 'ztf_sigmapsf']].dropna()

        
        
        # Split into g-band and i-band
        for band in [1, 2]:  # 1 = g-band, 2 = i-band
            band_data = data[data['ztf_fid'] == band]
            times, mags, errors = band_data['ant_mjd'].values, band_data['ztf_magpsf'].values, band_data['ztf_sigmapsf'].values
            
            if self.is_microlensing_candidate(times, mags, errors):
                print(f'Locus {locus.locus_id} is a microlensing candidate in band {band}')
                locus.tag('microlensing_candidate')
        
        
        return