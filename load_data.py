import os
import numpy as np
import scipy.io
import librosa
import torch
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from concurrent.futures import ThreadPoolExecutor

# ---------------------- Signal Utilities ----------------------

def find_signal_data_key(signal_dict, file_id, sensor_location='DE'):
    """
    Finds the key for signal data in a dictionary,
    prioritizing '_<sensor_location>_time' and falling back to '_time'.
    
    Args:
        signal_dict (dict): The dictionary containing signal data (from scipy.io.loadmat).
        file_id (str): The file ID to search for in the key.
        sensor_location (str): The sensor location ('DE' or 'FE'). Defaults to 'DE'.
    
    Returns:
        str: The key for the signal data.
    
    Raises:
        KeyError: If no matching key is found.
    """
    matching_keys_id_time = [
        key for key in signal_dict.keys()
        if key.endswith(f"_{sensor_location}_time") and file_id in key
    ]
    if matching_keys_id_time:
        return matching_keys_id_time[0]
    else:  # If no '_<sensor_location>_time' key is found, look for '_time'
        matching_keys_time = [
            key for key in signal_dict.keys()
            if key.endswith(f"_{sensor_location}_time")
        ]
        if matching_keys_time:
            return matching_keys_time[0]
        else:
            raise KeyError(f"No key ending with '_{sensor_location}_time' or '_time' and containing '{file_id}' found in the .mat file.")

def segment_signal(signal_data, segment_len, hop_len):
    """
    Segments the signal data into smaller chunks of segment_len length.
    """
    num_segments = (len(signal_data) - segment_len) // hop_len + 1
    segments = [signal_data[i * hop_len:i * hop_len + segment_len]
                for i in range(num_segments)
                if i * hop_len + segment_len <= len(signal_data)]
    return segments

def bandpass_filter(data, lowcut=10, highcut=10000, fs=12000, order=4):
    """
    Applies a bandpass filter to the signal data.
    """
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    return filtfilt(b, a, data)

def normalize_signal(data):
    """
    Normalizes the signal data to zero mean and unit variance.
    """
    data = data - np.mean(data)
    return data / (np.std(data) + 1e-8)

def wavelet_denoise(signal, wavelet='db4', level=4):
    """
    Applies wavelet denoising to the signal data.
    """
    import pywt
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, wavelet)

# ---------------------- Data Loading and Preprocessing ----------------------

def load_cwru_signals(data_root, label_dicts, normal_dict, segment_len=1024, hop_len=512, target_sr=12000, sensor_location='DE'):
    """
    Load CWRU dataset signals, process them, and segment them.
    """
    X_list, y_list = [], []

    # Process Normal baseline data
    for sr, file_ids in normal_dict.items():
        for file_id in file_ids:
            mat_path = os.path.join(data_root, f"{file_id}.mat")
            if os.path.exists(mat_path):
                signal = scipy.io.loadmat(mat_path)
                signal_data_key = find_signal_data_key(signal, file_id, sensor_location)
                signal_data = signal[signal_data_key].flatten()  # Flatten the signal data

                # Resample the signal to the target sampling rate
                signal_data = librosa.resample(signal_data, orig_sr=int(sr.replace('k', '000')), target_sr=target_sr)

                # Apply bandpass filter
                signal_data = bandpass_filter(signal_data, fs=target_sr)

                # Denoise and normalize
                signal_data = wavelet_denoise(signal_data)
                signal_data = normalize_signal(signal_data)

                # Segment the signal
                segments = segment_signal(signal_data, segment_len, hop_len)
                label = [fault_type_map['Normal'], fault_size_map['0']]
                X_list.extend(segments)
                y_list.extend([label] * len(segments))

    # Process Fault data (DE and FE)
    for fault_data in [label_dicts[sampling_rate]]:
        for fault_type, sizes in fault_data.items():
            if isinstance(sizes, dict):
                for fault_size, file_ids in sizes.items():
                    for file_id in file_ids:
                        mat_path = os.path.join(data_root, f"{file_id}.mat")
                        if os.path.exists(mat_path):
                            signal = scipy.io.loadmat(mat_path)
                            signal_data_key = find_signal_data_key(signal, file_id, sensor_location)
                            signal_data = signal[signal_data_key].flatten()

                            # Resample, filter, denoise, normalize
                            signal_data = librosa.resample(signal_data, orig_sr=int(sampling_rate.replace('k', '000')), target_sr=target_sr)
                            signal_data = bandpass_filter(signal_data, fs=target_sr)
                            signal_data = wavelet_denoise(signal_data)
                            signal_data = normalize_signal(signal_data)

                            # Segment the signal
                            segments = segment_signal(signal_data, segment_len, hop_len)
                            fault_size_label = fault_size_map.get(fault_size, fault_size_map['0'])
                            label = [fault_type_map[fault_type], fault_size_label]
                            X_list.extend(segments)
                            y_list.extend([label] * len(segments))

            elif isinstance(sizes, list):
                for file_id in sizes:
                    mat_path = os.path.join(data_root, f"{file_id}.mat")
                    if os.path.exists(mat_path):
                        signal = scipy.io.loadmat(mat_path)
                        signal_data_key = find_signal_data_key(signal, file_id, sensor_location)
                        signal_data = signal[signal_data_key].flatten()

                        # Resample, filter, denoise, normalize
                        signal_data = librosa.resample(signal_data, orig_sr=int(sampling_rate.replace('k', '000')), target_sr=target_sr)
                        signal_data = bandpass_filter(signal_data, fs=target_sr)
                        signal_data = wavelet_denoise(signal_data)
                        signal_data = normalize_signal(signal_data)

                        # Segment the signal
                        segments = segment_signal(signal_data, segment_len, hop_len)
                        fault_size = next((size for size in fault_size_map if size in file_id), '0')
                        fault_size_label = fault_size_map.get(fault_size, fault_size_map['0'])
                        label = [fault_type_map[fault_type], fault_size_label]
                        X_list.extend(segments)
                        y_list.extend([label] * len(segments))

    # Pad segments to segment_len if necessary
    X_padded = [np.pad(x, (0, segment_len - len(x)), 'constant') if len(x) < segment_len else x for x in X_list]

    return np.array(X_padded), np.array(y_list)

# ---------------------- Plotting and Visualization ----------------------

def plot_signal(signal, signal_type='Original', ax=None):
    """
    Plots a signal on the given axes.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    ax.plot(signal)
    ax.set_title(f'{signal_type} Signal')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Amplitude')
    ax.grid(True)

    return ax

def visualize_fault_segments(X, y, segment_len=1024):
    """
    Visualizes the first and last segments for each fault type and size.
    """
    unique_fault_types = torch.unique(y[:, 0])
    unique_fault_sizes = torch.unique(y[:, 1])

    fig, axes = plt.subplots(len(unique_fault_types), len(unique_fault_sizes), figsize=(12, 6))

    segment_counts = {}

    for i, fault_type in enumerate(unique_fault_types):
        for j, fault_size in enumerate(unique_fault_sizes):
            indices = (y[:, 0] == fault_type) & (y[:, 1] == fault_size)
            selected_segments = X[indices]

            num_segments = len(selected_segments)
            segment_counts[(fault_type.item(), fault_size.item())] = num_segments

            if num_segments > 0:
                first_segment = selected_segments[0]
                last_segment = selected_segments[-1]

                plot_signal(first_segment, signal_type=f"Fault Type: {fault_type.item()}, Size: {fault_size.item()} - First", ax=axes[i, j])
                plot_signal(last_segment, signal_type=f"Fault Type: {fault_type.item()}, Size: {fault_size.item()} - Last", ax=axes[i, j])

    plt.tight_layout()
    plt.show()

    print("Segment Counts for each Fault Type and Size:")
    for (fault_type, fault_size), count in segment_counts.items():
        print(f"Fault Type: {fault_type}, Fault Size: {fault_size} -> Number of Segments: {count}")

# Example usage of the integrated functions:
# X, y = load_cwru_signals(data_root, label_dicts, normal_dict)
# visualize_fault_segments(X, y)
