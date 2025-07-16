import numpy as np
import pandas as pd
import os
import spectral.io.envi as envi
import glob
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
stats=importr('stats')
scales=importr('scales')

import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri

def read_fit_spectra(spectra_path, wavelengths, plot=False, spectra_noise_threshold = 0.01):
    """Read a published spectra and fit to wavelengths and range of hyperspectral camera

    :param spectra_path: A string indicating the path to the file for a spectra (in format from fpbase.com)
    :param wavelengths: A list of precise wavelengths for which this function will fit the published spectra
    :param plot: Logical; if True, produce a plot of the single, individual spectra being fit.
    :spectra_noise_threshold: A float indicating a threshold below which fitted spectra values are set to zero (default value is 0.01, an approximate noise threshold of the Andor hyperspectral camera)
    ...
    :return: A pandas DataFrame with one column for wavelength and another for signal intensity
    """
    # numpy2ri.activate() removed - deprecated in newer rpy2 versions
    # pandas2ri.converter in localconverter below handles conversions

    pub_emission_spectrum = pd.read_csv(spectra_path)
    pub_emission_spectrum = pub_emission_spectrum.rename(columns={'emission wavelength (nm)': 'wavelength'})
    pub_emission_spectrum = pub_emission_spectrum.rename(columns={'Normalized emission': 'intensity'})

    # Convert pandas object to something R-friendly https://rpy2.github.io/doc/latest/html/pandas.html
    with localconverter(ro.default_converter + ro.pandas2ri.converter):
        r_from_pd_df = ro.conversion.py2rpy(pub_emission_spectrum) # If not in rpy2>=3.0.5 may have error here due to no attribute 'py2rpy'

    # Use rpy2 so we can use loess from R, as in old version, get same exact results
    # Fit
    fit = stats.loess(ro.Formula('intensity~wavelength'), data=r_from_pd_df, span=0.1)

    # Predict
    predictions = stats.predict(fit, wavelengths)
    predictions = scales.rescale(predictions, center='FALSE') # by default, scales between 0 and 1.

    # Scale
    scaled_emission_spectra = {'wavelength': wavelengths, 'intensity': predictions}
    scaled_emission_spectra = pd.DataFrame(scaled_emission_spectra)

    # Denoise
    scaled_emission_spectra.loc[scaled_emission_spectra['intensity'] < spectra_noise_threshold, 'intensity'] = 0

    if plot == True:
        print('Plotting not supported here yet.')

    # numpy2ri.deactivate() removed - deprecated in newer rpy2 versions

    return(scaled_emission_spectra)

def build_X(fluorophore_ID_vector, spectral_library_path,
            intercept, wavelengths, spectra_noise_threshold,
            min_desired_wavelength, max_desired_wavelength):
    """Build an X matrix of spectral components (and intercept) to be used for regression, with individual calls to read_fit_spectra for each component (and intercept) and combination, formatting of results.

    :param fluorophore_ID_vector: A list of spectral components in the spectral library
    :param spectral_library_path: A string indicating the directory where spectra can be found
    :param intercept: If 0, no intercept is added to X. If 1, a vector of 1s equal to # spectra is prepended to X.
    :param wavelengths: A list of precise wavelengths to be passed to read_fit spectra, which will fit the published spectra
    :param spectra_noise_threshold: A float indicating a threshold below which fitted spectra values are set to zero (default value is 0.01, an approximate noise threshold of the Andor hyperspectral camera), passed to read_fit_spectra
    :param min_desired_wavelength: A numeric value indicating a threshold BELOW which spectral data is excluded
    :param max_desired_wavelength: A numeric value indicating a threshold ABOVE which spectral data is excluded
    ...
    :return: An X matrix for running multiple linear regression, with a column for each fluorophore (and the intercept, if applicable)
    """
    from gmodetector_py import find_desired_indices
    for i in range(0, (len(fluorophore_ID_vector))):
        path = (spectral_library_path + str(fluorophore_ID_vector[i]) + ".csv")
        if(os.path.isfile(path) == False):
            raise NameError("Error: Spectrum for " + fluorophore_ID_vector[i] + " not found in spectra_library folder")

        spectra = read_fit_spectra(spectra_path = path,
                                   wavelengths = wavelengths,
                                   plot = False,
                                   spectra_noise_threshold = spectra_noise_threshold)

        if(i==0 and intercept==1):
            intercept_vector = [1] * len(wavelengths)
            # https://stackoverflow.com/questions/43961585/cbind-r-function-equivalent-in-numpy
            mm = np.c_[(intercept_vector, np.asarray(spectra['intensity']))]

        if(i==0 and intercept==0):
            mm = np.array(spectra['intensity'])

        if(i>0):
            # https://stackoverflow.com/questions/43961585/cbind-r-function-equivalent-in-numpy
            mm = np.c_[mm, np.array(spectra['intensity'])]

    #wavelengths = np.asarray(wavelengths)

    # https://stackoverflow.com/questions/13869173/numpy-find-index-of-the-elements-within-range
    #wavelength_indices_desired = np.where(np.logical_and(wavelengths.astype(float)>=min_desired_wavelength,
    #                                                     wavelengths.astype(float)<=max_desired_wavelength))

    wavelength_indices_desired = find_desired_indices(wavelengths = wavelengths,
                                                      min_desired_wavelength = min_desired_wavelength,
                                                      max_desired_wavelength = max_desired_wavelength)

    return(np.squeeze(mm[wavelength_indices_desired,]))
