import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import welch

from glob import glob


def get_data(path):
    """
    function for getting data from files
    :param path: path to file
    :return:
    """
    data = pd.read_csv(path, delimiter="\t")
    rr = data.iloc[:, 0].values; flags = data.iloc[:, 1].values
    timetrack = np.cumsum(rr)
    return rr, flags, timetrack


def interpolate_non_sinus(rr, flags):
    """
     this function linearly interpolates the non-sinus beats
    :param rr: the RR intervals time series
    :param flags: the RR intervals annotations
    :return:
    """

    rr_local = rr.copy()
    inside_non_sinus = False
    segment_end = 0
    segment_start = 0
    good_segment_start = 0
    good_intervals_list = []
    keep_last = True
    for idx in range(len(rr_local)):
        if flags[idx] != 0 and not inside_non_sinus:
            segment_start = idx - 1
            good_intervals_list.append(rr_local[good_segment_start:(segment_start + 1)])
            inside_non_sinus = True
        if inside_non_sinus and (flags[idx] == 0  or idx == len(rr_local) - 1):
            if idx == len(rr_local) - 1:
                keep_last = False
                break
            segment_end = idx
            good_segment_start = segment_end
            interpolated_sequence = optimal_division(rr_local, segment_start, segment_end)
            good_intervals_list.append(interpolated_sequence)
            inside_non_sinus = False
    # now adding the last good segment to good_intervals_list
    if keep_last:
        good_intervals_list.append(rr_local[good_segment_start:])
    return np.concatenate(good_intervals_list)


def optimal_division(rr, start, stop):
    """
    function which divides the non-sinus segment into realistic RR intervals
    :param rr: rr intervals time series
    :param start: beginning of the non-sinus segment
    :param stop: end of the non_sinus segment
    :return: a sequence of rr intervals corresponding to the non-sinus segment
    """
    # the optimal rr interval is to be taken as the mean of the two correct
    # rr intervals surrounding the non-sinus segment
    rr_local_2 = rr.copy()
    optimal_rr = (rr_local_2[start] + rr_local_2[stop]) / 2

    segment_length = np.cumsum(rr_local_2[0:stop])[-1] - np.cumsum(rr_local_2[0:start + 1])[-1]

    # now searching for optimal division
    optimal = False
    divisions = 1
    delta = np.abs(segment_length - optimal_rr)
    while not optimal:
        current_rr = segment_length / (divisions + 1)
        if np.abs(current_rr - optimal_rr) > delta:
            optimal = True
        else:
            delta = np.abs(current_rr - optimal_rr)
        divisions += 1
    optimal_rr = [segment_length / (divisions - 1)]
    return np.array(optimal_rr * (divisions - 1))


def resample_rr(rr, timetrack, period = 250, method = 'cubic'):
    """
    this function resamples the RR time series at period - the default value is 250 ms, which corresponds to 0.25 s or 4 Hz
    :rr: the RR intervals time series
    :timetrack: timetrack
    :period: resampling period, default value 250 ms
    :method: interpolation method, default is cubic splines
    """
    interp_object = interp1d(np.cumsum(rr), rr, kind=method, fill_value='extrapolate')
    timetrack_resampled = np.arange(timetrack[0], timetrack[-1], step = period)
    rr_resampled = interp_object(timetrack_resampled)
    return timetrack_resampled, rr_resampled


def calculate_welch(rr_resampled, fs=4, window='hanning', segment_min = 5,
                    noverlap_frac = 0.5):
    """
    function to calculate the actual Welch periodogram
    :param rr_resampled: resampled RR-intervals time series
    :param fs: resampling frequency
    :param window: type of window
    :param segment_min: length of Welch segments in minutes
    :param noverlap_frac: how much to overlap - default is half
    :return: the spectrum in frequency bands
    """
    w_spectrum = welch(rr_resampled - np.mean(rr_resampled), fs=4, window=window, nperseg=segment_min * 60 * fs,
                       noverlap= segment_min * 60 * fs * noverlap_frac, return_onesided=False, scaling='spectrum')

    return w_spectrum


def calculate_bands(w_spectrum, bands=[0.003, 0.04, 0.15, 0.4], ulf=True):
    """
    function to calculate the spectral bands
    :param w_spectrum: Welch spectrum
    :param bands: bands for calculating spectrum
    :param ulf: should I calculate ULF?
    :return: dictionary with band names as keys and spectral bands as values
    """
    print("pierwsze bandy", bands, bands == [0.003, 0.04, 0.15, 0.4])
    if not ulf and bands == [0.003, 0.04, 0.15, 0.4]:
        bands = [0.04, 0.15, 0.4]
        band_names = ["vls", "lf", "hf"]
    elif bands == [0.003, 0.04, 0.15, 0.4]:
        band_names = ["ulf", "vlf", "lf", "hf"]
    else:
        band_names = [str(_) for _ in bands]
    print(bands, band_names)
    extended_bands = [0]; extended_bands.extend(bands)
    spectral_bands = []
    for band_idx in range(1, len(extended_bands)):
        spectral_bands.append(np.sum(np.abs(w_spectrum[1][np.logical_and(w_spectrum[0] > extended_bands[band_idx - 1],
                                                                  w_spectrum[0] <= extended_bands[band_idx])])) * 2)
    spectral_bands.append(np.sum(spectral_bands))
    band_names.append("tp")
    results = pd.DataFrame([spectral_bands], columns=band_names)

    return results


def all_results(save=False, filepath='welch_results.xlsx'):
    csv_files = glob("*.txt")
    results_dataframe = pd.DataFrame()
    for data_file in csv_files:
        rr, flags, timetrack = get_data(data_file)
        interpolated_rr = interpolate_non_sinus(rr, flags)
        resampled_timetrack, resampled_rr = resample_rr(interpolated_rr, timetrack)
        welch_spectrum = calculate_welch(resampled_rr)

        results_dataframe = results_dataframe.append(calculate_bands(welch_spectrum), ignore_index=True)
    results_dataframe.index = csv_files
    if save:
        results_dataframe.to_excel(filepath)
    return results_dataframe


if __name__ == '__main__':
    print(all_results(save=True))