import sys

import numpy as np
from scipy.signal import find_peaks
from astropy.io import fits

def load_data(filepath, dark_filepath=None):
    """
    Load data from a FITS file and crop the edges to remove 
    meta pixels.

    Parameters
    ----------
    filepath : str
        Filepath to the FITS file.
    dark_filepath : str, optional
        Filepath to the FITS file containing the dark. The 
        default is None.

    Returns
    -------
    numpy array
        3D array containing the datacube from the FITS file.
    """

    assert filepath.lower().endswith('.fits'), \
                f'Must provide a FITS file, not {filepath}'
    assert dark_filepath is None or dark_filepath.lower().endswith('.fits'), \
                f'Must provide a FITS file, not {dark_filepath}'
                
    with fits.open(filepath) as hdul:
        data = hdul[0].data[:, 3:-3, 3:-3]
        
    if dark_filepath is not None:
        with fits.open(dark_filepath) as hdul:
            dark = hdul[0].data[:, 3:-3, 3:-3]
        
        data = data - dark
        
    data = np.clip(data, 1e-3, None)
    return data

def check_data_orientation(data):
    """
    Identifies the orientation of the data, rotating it if it is
    oriented so that the spectra are horizontal.
    
    Parameters
    ----------
    data : numpy array
        3D array containing the datacube from the FITS file.

    Returns
    -------
    numpy array
        3D array containing the datacube with the spectra oriented
        vertically
    """
    
    # Summing over the spectra gives a spikier output than summing
    # over the frames if the spectra are horizontal
    x_axis = data[0].mean(axis=0)
    x_axis = x_axis - np.percentile(x_axis, 1)
    x_axis = x_axis / np.percentile(x_axis, 99)
    
    y_axis = data[0].mean(axis=1)
    y_axis = y_axis - np.percentile(y_axis, 1)
    y_axis = y_axis / np.percentile(y_axis, 99)
    
    # Check if the data is already oriented correctly
    if x_axis.max() > y_axis.max():
        return data
    
    # Otherwise, rotate the data
    return np.transpose(data, (0, 2, 1))
    
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
    
    The 6, 11 or 12 spectra are identified and the top of the box 
    to be drawn around each of them. 

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
    int
        The y-coordinate of the bottom of the box to be drawn around
        the spectra. This is only reliable if the dark has been 
        subtracted.
    """

    # Rescale data
    x_axis = data.mean(axis=0)
    x_axis = x_axis - np.percentile(x_axis, 1)
    x_axis = x_axis / np.percentile(x_axis, 99)

    # Find peaks
    height = 0.1
    peaks = find_peaks(x_axis, height=height, distance=25)[0]

    allowed_num_of_peaks = [12, 11, 7]

    # Retry peak finding until successful
    if len(peaks) not in allowed_num_of_peaks:
        min_height = -0.05
        max_height = 0.5
        while len(peaks) not in allowed_num_of_peaks:
            if len(peaks) < min(allowed_num_of_peaks):
                max_height = height
            else:
                min_height = height
            
            height = (min_height + max_height)/2
            print(f'Trying {height:.2f}')
            peaks = find_peaks(x_axis, height=height, distance=25)[0]
            if max_height - min_height < 1e-3:
                raise ValueError('Unable to find peaks.')
            
    # Check peaks are valid
    if len(peaks) == 12:
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

    # Find bottom of box
    above = (y_axis > 1.2*baseline)
    idxs = np.flatnonzero(above[:-1] != above[1:])
    idxs = np.hstack(([0], idxs, [len(above)]))

    sequence_lengths = idxs[1:]-idxs[:-1]
    if above[0]:
        sequence_lengths[1::2] *= -1
    else:
        sequence_lengths[::2] *= -1

    assert np.max(sequence_lengths) > 50, f'Very small sequence found: {np.max(sequence_lengths)}'
    longest_indx = np.argmax(sequence_lengths)+1
    box_top, box_bottom = idxs[longest_indx-1:longest_indx+1]
    return peaks, box_top, box_bottom

def extract_spectra(data, peaks, box_top, box_bottom=None):
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
    box_bottom : int, optional
        The y-coordinate of the bottom of the box to be drawn around
        the spectra. The default is None which means the box will
        extend to the bottom of the frame.

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
        spectrum = data[:, box_top:box_bottom, peak-box_half_width:peak+box_half_width]
        spectra.append(spectrum.sum(axis=2))

    spectra = np.array(spectra)
    return np.transpose(spectra, (1, 0, 2))
    
if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Usage: python find_spectra.py <filepath>')
        sys.exit(1)
    elif len(sys.argv) > 3:
        print('Too many arguments provided.')
        sys.exit(1)
    elif len(sys.argv) == 3:
        dark_filepath = sys.argv[2]
    else:
        dark_filepath = None
    filepath = sys.argv[1]
    data = load_data(filepath, dark_filepath)
    data = check_data_orientation(data)
    peaks, box_top, box_bottom = find_spectra(data[0])
    if dark_filepath is None:
        box_bottom = None
    spectra = extract_spectra(data, peaks, box_top, box_bottom)
    save_data(filepath, spectra)
