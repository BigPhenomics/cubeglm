# Import classes
# Check version
from gmodetector_py import version
print('Running GMOdetector version ' + version.__version__)

from gmodetector_py import XMatrix
from gmodetector_py import Hypercube
from gmodetector_py import WeightArray
from gmodetector_py import ImageChannel
from gmodetector_py import FalseColor

# Import the one function we need to use before calling classes
from gmodetector_py import read_wavelengths

# The following are needed for this specific wrapper script only
import warnings
import argparse
import time
from datetime import datetime
import ntpath

import os # needed for basename

parser = argparse.ArgumentParser(description="""This script provides a wrapper
for analyzing a sample by a start-to-finish hyperspectral regression workflow,
including plotting and saving of weight arrays for each spectral component.""")

parser.add_argument('--file_path', type = str,
                    help = """Filepath to the metadata file (.hdr) for
                    a sample to be analyzed""")
parser.add_argument('--fluorophores', type = str, nargs = '+',
                    dest = 'fluorophore_ID_vector',
                    help = 'A list of spectral components in the spectral library')
parser.add_argument('--min_desired_wavelength', type = float,
                    help = """A numeric value indicating a threshold BELOW
                    which spectral data is excluded""")
parser.add_argument('--max_desired_wavelength', type = float,
                    help = """A numeric value indicating a threshold ABOVE
                    which spectral data is excluded""")
parser.add_argument('--green_channel', type = str,
                    help = """A str matching the ID of the spectral component
                    for which weights will be plotted (e.g. 'GFP')
                    with GREEN false color""")
parser.add_argument('--red_channel', type = str,
                    help = """A str matching the ID of the spectral component
                    for which weights will be plotted (e.g. 'GFP')
                    with RED false color""")
parser.add_argument('--blue_channel', type = str,
                    help = """A str matching the ID of the spectral component
                    for which weights will be plotted (e.g. 'GFP')
                    with BLUE false color""")
parser.add_argument('--green_cap', type = str,
                    help = """A numeric value of the spectral intensity value
                    of the GREEN channel (specified by `--green-channel` that
                    will have maximum brightness in the plot.
                    All with greater intensity will have the same level of
                    brightness. Think of this as image exposure on a camera.""")
parser.add_argument('--red_cap', type = str,
                    help = """A numeric value of the spectral intensity value
                    of the RED channel (specified by `--red-channel` that
                    will have maximum brightness in the plot.
                    All with greater intensity will have the same level of
                    brightness. Think of this as image exposure on a camera.""")
parser.add_argument('--blue_cap', type = str,
                    help = """A numeric value of the spectral intensity value
                    of the BLUE channel (specified by `--blue-channel` that
                    will have maximum brightness in the plot.
                    All with greater intensity will have the same level of
                    brightness. Think of this as image exposure on a camera.""")
parser.add_argument('--weight_format', type = str, default = 'hdf',
                    help = """The format in which the weight array will be saved;
                    Currently supported are `csv` and `hdf` (h5py) formats.""")
parser.add_argument('--plot', type = bool, default = False,
                    help = """Boolean (True or False) indicating whether to
                    create false color plots of spectral components. Note that
                    at the time of writing documentation, there is no support
                    for producing false color plots of hypercubes prior to
                    regression in `analyze_sample`; this must be done
                    independently for a given sample as described in
                    documentation for the `Hypercube` class.""")
parser.add_argument('--spectral_library_path', type = str,
                    default = './spectral_library/',
                    help = """Path to a folder in which all spectra listed by the
                    `--fluorophores` flag are included in the appropriate format
                    (See documentation for `XMatrix` object, or examples for
                    format details""")
parser.add_argument('--intercept', type = int, default = 1,
                    help = """If 0, no intercept is added to X. If 1, a vector
                    of 1's equal to # spectra is prepended to X during regression.
                    This value should be set to 1 if there is any significant
                    level of background noise in hypercubes (Y) being
                    analyzed""")
parser.add_argument('--spectra_noise_threshold', type = float,
                    default = 0.01,
                    help = """A float indicating a threshold below which fitted
                    spectra values are set to zero (default value is 0.01,
                    an approximate noise threshold of the Andor hyperspectral
                    camera with settings used by Strauss Lab at the time of
                    writing documentation), passed to read_fit_spectra)""")
parser.add_argument('--normalize', type = bool, default = False,
                    help = """Boolean (True or False) indicating whether
                    to normalize the experimental sample against
                    a user-provided chroma standard""")
parser.add_argument('--rescale', type = bool, default = True,
                    help = """Boolean (True or False) indicating whether to
                    rescale the experimental sample's hypercube after
                    normalization to bring the normalized spectra
                    to approximately the sample mean intensity as prior to
                    normalization""")
parser.add_argument('--chroma_width', type = int, default = 10,
                    help = """The number of pixels to be extracted from the
                    center of the chroma standard hypercube, over which the
                    mean for each row will be taken and used for normalizing
                    fluctuations in laser and/or signal intensity""")
parser.add_argument('--chroma_path', default = None,
                    help = """The filepath to a chroma standard against which
                    experimental samples will be normalized. Please note the
                    `--normalize` flag must be set to `True` if normalizing.""")
parser.add_argument('--relu' , type = bool, default = True,
                    dest = 'relu_before_plot',
                    help = """Whether to replace values below zero in the weight
                    array with zero before making plots; needed for scales to be
                    consistent across images with the same color/cap settings""")
parser.add_argument('--output_dir', type = str, default = './',
                    help = 'Filepath to directory for saving outputs')
parser.add_argument('--threshold', type = float, default = 0,
                    help = """Threshold above which pixels will be counted as
                    significant in the summary statistic spreadsheet output""")

args = parser.parse_args()


def analyze_sample(file_path,fluorophore_ID_vector,
                   min_desired_wavelength, max_desired_wavelength,
                   green_channel, red_channel, blue_channel,
                   green_cap, red_cap, blue_cap, weight_format = 'hdf', plot = True,
                   spectral_library_path = './spectral_library/', intercept = 1,
                   spectra_noise_threshold = 0.01, normalize = False, rescale = True,
                   chroma_width = 10, chroma_hypercube = None, chroma_path = None,
                   relu_before_plot = True, output_dir = './', relu = True,
                   threshold = 0):
    """This function provides a wrapper for analyzing a sample by a
    start-to-finish hyperspectral regression workflow, including plotting
    and saving of weight arrays for each spectral component. At the time of
    writing documentation, this function is intended to be accessed from the
    command line. To see further documentation, including for each argument
    passed from the command line to this function, run
    `analyze_sample.py --help` from the command line.

    All arguments to this function are accesible from command line at the time
    of documentation except for `chroma_hypercube`, an preloaded alternative to
    `chroma_path` that allows for a common chroma standard to be stored in shared
    memory when parallelization in Python (rather than just command line) is
    used."""

    filepath_sansdir_basename = os.path.splitext(ntpath.basename(file_path))[0]
    #log_path = output_dir + filepath_sansdir_basename + '.log'
    #if os.path.isfile(log_path):
    #    warnings.warn('Sample' + filepath_sansdir_basename + """has already been 
    #    run, with output saved to the same dir specified: """ + output_dir + 
    #    'Results and log will be overwritten.')
    
    from datetime import datetime
    # datetime object containing current date and time
    start_time = datetime.now()
    
    # dd/mm/YY H:M:S
    dt_string = start_time.strftime("%d/%m/%Y %H:%M:%S")
    #print("Analysis for this sample is starting at " +
    #      dt_string + '\n'
    #      ' with parameters: ' + str(locals()))
    #      #file=open(log_path, "w"))
    
    time_pre_read_partial = time.perf_counter()

    if chroma_hypercube is not None and chroma_path is not None:
        raise Exception("""Chroma standard was submitted both as a pre-loaded
                        hypercube (`chroma_hypercube`) and
                        a filepath (`chroma_path`). Please choose one.""")

    if normalize == True and (chroma_path is None and chroma_hypercube is None):
        raise Exception("""Normalization is on (via `normalize` option) but
                        no chroma standard has been provided for normalization
                        (via either `chroma_path` or `chroma_hypercube`).""")

    if normalize == False and (chroma_path is not None or chroma_hypercube is not None):
        raise Exception("""Normalization is off (via `normalize` option) but
                        a chroma standard has been provided. Set `normalize` to
                        True if you wish to perform normalization of the sample.""")

    if chroma_hypercube is None and chroma_path is None:
        warnings.warn('Sample is not being normalized with a chroma standard.')

    if chroma_hypercube is None and chroma_path is not None:
        chroma_hypercube = Hypercube(file_path,
        min_desired_wavelength = min_desired_wavelength,
        max_desired_wavelength = max_desired_wavelength)

    if normalize == True:
        normalize(self, chroma_hypercube, chroma_width, rescale = rescale)

    wavelengths = read_wavelengths(file_path = file_path)

    test_matrix = XMatrix(fluorophore_ID_vector = fluorophore_ID_vector,
                      spectral_library_path = spectral_library_path,
                      intercept = intercept,
                      wavelengths = wavelengths,
                      spectra_noise_threshold = spectra_noise_threshold,
                      min_desired_wavelength = min_desired_wavelength,
                      max_desired_wavelength = max_desired_wavelength)


    test_cube = Hypercube(file_path,
                          min_desired_wavelength = min_desired_wavelength,
                          max_desired_wavelength = max_desired_wavelength)

    weight_array = WeightArray(test_matrix=test_matrix,
                               test_cube=test_cube,
                               relu = relu_before_plot)
    
    # Subset weight array from after intercept (if exists) until last fluorophore
    weight_array_sans_intercept = weight_array.weights[:, :, intercept:weight_array.weights.shape[2]]

    # Only save weight array if laser is on 
    if weight_array_sans_intercept.max() > 50:
        weight_array.save(ntpath.basename(weight_array.source),
                          format = weight_format,
                          output_dir = output_dir,
                          threshold = threshold)
    if weight_array_sans_intercept.max() <= 50:
        print("ALERT! The laser did not turn on during imaging of sample " + filepath_sansdir_basename)

    if plot == True:

        if relu_before_plot == True:
            weight_array.relu()

        stacked_component_image = FalseColor([ImageChannel(weight_array = weight_array,
                                                           desired_component_or_wavelength = green_channel,
                                                           color = 'green',
                                                           cap = green_cap),
                                              ImageChannel(weight_array = weight_array,
                                                           desired_component_or_wavelength = red_channel,
                                                           color = 'red',
                                                           cap = red_cap),
                                              ImageChannel(weight_array = weight_array,
                                                           desired_component_or_wavelength = blue_channel,
                                                           color = 'blue',
                                                           cap = blue_cap)])

        stacked_component_image.save(stacked_component_image.source,
        output_dir = output_dir)

    time_post_read_partial = time.perf_counter() - time_pre_read_partial
    
    import gc
    gc.collect()
    
    print('\nFinished running sample ' + 
          file_path + ' in ' + 
          str(time_post_read_partial) + 's' + '\n')
    
    #print('Finished running sample ' + 
    #      file_path + ' in ' + 
    #      str(time_post_read_partial) + 's' + '\n',
    #      file=open(log_path, "a"))
    
    #print('Log saved to: ' + log_path + '\n')
        
if __name__ == "__main__":
    analyze_sample(file_path = args.file_path, # needed to avoid TypeError: expected str, bytes or os.PathLike object, not list
    fluorophore_ID_vector = args.fluorophore_ID_vector,
    min_desired_wavelength = args.min_desired_wavelength,
    max_desired_wavelength = args.max_desired_wavelength,
    green_channel = args.green_channel,
    red_channel = args.red_channel,
    blue_channel = args.blue_channel,
    green_cap = args.green_cap,
    red_cap = args.red_cap,
    blue_cap = args.blue_cap,
    weight_format = args.weight_format,
    plot = args.plot,
    spectral_library_path = args.spectral_library_path,
    intercept = args.intercept,
    spectra_noise_threshold = args.spectra_noise_threshold,
    normalize = args.normalize,
    rescale = args.rescale,
    chroma_width = args.chroma_width,
    #chroma_hypercube = None,
    chroma_path = args.chroma_path,
    relu_before_plot = args.relu_before_plot,
    output_dir = args.output_dir,
    threshold = args.threshold)
    
    #print(args)
