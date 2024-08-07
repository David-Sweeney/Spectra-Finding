import sys

import numpy as np
from scipy.signal import find_peaks
from astropy.io import fits

def load_data(filepath):
    """
    Load data from a FITS file and crop the edges to remove 
    meta pixels.

    Parameters
    ----------
    filepath : str
        Filepath to the FITS file.

    Returns
    -------
    numpy array
        3D array containing the datacube from the FITS file.
    """

    assert filepath.lower().endswith('.fits'), \
                f'Must provide a FITS file, not {filepath}'
    with fits.open(filepath) as hdul:
        data = hdul[0].data[:, 3:-3, 3:-3]
    
    data = np.clip(data, 0., None)
    return data

def save_data(original_filepath, spectra, verbose=True):
    """
    Save the spectra to a .npy file with the same name as the 
    original file, but with a _spectra added.

    Parameters
    ----------
    original_filepath : str
        Filepath to the original FITS file.
    spectra : numpy array
        3D array containing the extracted spectra.
    verbose : bool, optional
        Whether to print the filepath of the saved file. The default
        is True.
    """

    filepath = original_filepath.replace('.fits', '_spectra.npy')
    np.save(filepath, spectra)
    if verbose:
        print(f'Saved to {filepath}')

def find_spectra(data):
    """
    Find the spectra in the data.
    
    The 12 spectra are identified and the top of the box to be 
    drawn around each of them. 

    Parameters
    ----------
    data : numpy array
        2D array containing a frame from the FITS file.

    Returns
    -------
    numpy array
        1D array containing the x-coordinates of the peaks.
    int
        The y-coordinate of the top of the box to be drawn around
        the spectra.
    """

    # Rescale data
    x_axis = data.mean(axis=0)
    x_axis = x_axis - np.percentile(x_axis, 1)
    x_axis = x_axis / np.percentile(x_axis, 99)

    # Find peaks
    height = 0.1
    peaks = find_peaks(x_axis, height=height, distance=25)[0]

    # Retry peak finding until successful
    if len(peaks) != 12:
        min_height = -0.05
        max_height = 1.
        while len(peaks) != 12:
            if len(peaks) < 12:
                max_height = height
            else:
                min_height = height
            
            height = (min_height + max_height)/2
            print(f'Trying {height:.2f}')
            peaks = find_peaks(x_axis, height=height, distance=25)[0]
            
    # Check peaks are valid
    peak_offsets = peaks[1:] - peaks[:-1]
    assert peak_offsets.min() > 27, f'Bad peaks (too close): {peaks}'
    assert peak_offsets.max() < 66, f'Bad peaks (too far): {peaks}'

    # Find box locations
    y_axis = np.mean(data, axis=1)
    
    # Take moving average to smooth out noise
    half_window = 5
    y_axis = np.pad(y_axis, half_window, mode='edge')
    cum_sum = np.cumsum(y_axis)
    y_axis = (cum_sum[2*half_window:] - cum_sum[:-2*half_window])/(2*half_window)
    
    baseline = np.median(y_axis[:10])
    box_top = np.argmax(y_axis > 1.2*baseline) - 25
    return peaks, box_top

def extract_spectra(data, peaks, box_top):
    """
    Extract the spectra from the data using the provided peaks
    and box top.

    Parameters
    ----------
    data : numpy array
        3D array containing the frame from the FITS file.
    peaks : numpy array
        1D array containing the x-coordinates of the peaks.
    box_top : int
        The y-coordinate of the top of the box to be drawn around
        the spectra.

    Returns
    -------
    numpy array
        3D array containing the extracted spectra. The array is
        of shape (num_frames, num_spectra, spectrum_length).
    """

    box_half_width = 2

    # Extract spectra
    spectra = []
    for peak in peaks:
        spectrum = data[:, box_top:, peak-box_half_width:peak+box_half_width]
        spectra.append(spectrum.sum(axis=2))

    spectra = np.array(spectra)
    return np.transpose(spectra, (1, 0, 2))
    
if __name__ == '__main__':
    filepath = sys.argv[1]
    data = load_data(filepath)
    peaks, box_top = find_spectra(data[0])
    spectra = extract_spectra(data, peaks, box_top)
    save_data(filepath, spectra)