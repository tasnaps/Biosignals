#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import butter, filtfilt, find_peaks, welch  # Added welch here
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def load_opensignals_file(filepath):
    """
    Load an OpenSignals Text File.
    The file is assumed to have a header ending with "# EndOfHeader" (with optional JSON metadata),
    and the data section contains whitespace-delimited numeric values.
    Returns a DataFrame of numeric data and metadata.
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()

    header_lines = []
    data_lines = []
    header_end_index = None

    # Separate header and data using the "# EndOfHeader" marker.
    for i, line in enumerate(lines):
        if "# EndOfHeader" in line:
            header_end_index = i
            break
        header_lines.append(line)

    if header_end_index is None:
        raise ValueError("Could not find '# EndOfHeader' in the file.")

    data_lines = [line.strip() for line in lines[header_end_index + 1:] if line.strip()]

    # Parse metadata from a header line that starts with JSON.
    metadata = {}
    column_names = ["col1", "col2", "col3"]
    for line in header_lines:
        line = line.strip()
        if line.startswith("# {"):
            try:
                json_str = line.lstrip("# ").strip()
                metadata = json.loads(json_str)
                device_key = list(metadata.keys())[0]
                device_info = metadata[device_key]
                if "column" in device_info:
                    column_names = device_info["column"]
                elif "label" in device_info:
                    column_names = device_info["label"]
            except Exception as e:
                print("Error parsing JSON metadata:", e)
            break

    data = [line.split() for line in data_lines]
    df = pd.DataFrame(data, columns=column_names)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df, metadata


def apply_ecg_transfer_function(raw_signal, channel="A2"):
    """
    Convert raw ECG signal to millivolts using:
      ecg (mV) = (raw_signal / 2**10 - 0.5) * (3 / 1019) * 1000
    """
    converted_signal = (raw_signal / (2 ** 10) - 0.5) * (3 / 1019) * 1000
    return converted_signal


def bandpass_filter(data, fs, lowcut=5, highcut=15, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def process_ecg(df, metadata):
    """
    Process the ECG data.
    Returns a dictionary containing:
      - fs: sampling rate
      - t: time vector (seconds)
      - ecg_converted, ecg_filtered, ecg_squared, ecg_integrated signals
      - peaks: detected R-peak indices (via the custom algorithm)
    """
    fs = 125
    if metadata:
        try:
            device_key, device_info = next(iter(metadata.items()))
            fs = int(device_info.get("sampling rate", fs))
        except Exception as e:
            print("Error retrieving sampling rate from metadata:", e)
    if "A2" not in df.columns:
        raise ValueError("ECG channel 'A2' not found in data.")
    raw_ecg = df["A2"].values
    t = np.arange(len(raw_ecg)) / fs

    ecg_converted = apply_ecg_transfer_function(raw_ecg, channel="A2")
    ecg_filtered = bandpass_filter(ecg_converted, fs)
    ecg_squared = ecg_filtered ** 2
    window_size = int(0.15 * fs)
    ecg_integrated = np.convolve(ecg_squared, np.ones(window_size) / window_size, mode="same")
    threshold = 0.5 * np.max(ecg_integrated)
    min_distance = int(0.2 * fs)
    peaks, _ = find_peaks(ecg_integrated, height=threshold, distance=min_distance)

    return {
        "fs": fs,
        "t": t,
        "ecg_converted": ecg_converted,
        "ecg_filtered": ecg_filtered,
        "ecg_squared": ecg_squared,
        "ecg_integrated": ecg_integrated,
        "peaks": peaks
    }


def plot_ecg_results(processed_data):
    """Plot the ECG signals and detected peaks in separate figure windows."""
    t = processed_data["t"]
    ecg_converted = processed_data["ecg_converted"]
    ecg_filtered = processed_data["ecg_filtered"]
    ecg_squared = processed_data["ecg_squared"]
    ecg_integrated = processed_data["ecg_integrated"]
    peaks = processed_data["peaks"]

    plt.figure(figsize=(12, 6))
    plt.plot(t, ecg_converted, label="Original ECG (Converted)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.title("Original ECG Signal")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(t, ecg_filtered, label="Filtered ECG")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.title("Bandpass Filtered ECG (5–15 Hz)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(t, ecg_squared, label="Squared ECG")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Squared ECG Signal")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(t, ecg_integrated, label="Integrated ECG")
    threshold = 0.5 * np.max(ecg_integrated)
    plt.axhline(threshold, color="red", linestyle="--", label="Threshold")
    plt.plot(t[peaks], ecg_integrated[peaks], "ko", label="Detected Peaks")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Integrated ECG with Detected Peaks")
    plt.legend()
    plt.show()


def compute_hrv(processed_data):
    """
    Compute custom HRV metrics (mean R–R, SDNN, RMSSD) in milliseconds.
    """
    t = processed_data["t"]
    peaks = processed_data["peaks"]
    rr_intervals = np.diff(t[peaks])
    if len(rr_intervals) == 0:
        raise ValueError("Not enough peaks detected to compute R–R intervals.")
    rr_intervals_ms = rr_intervals * 1000
    mean_rr = np.mean(rr_intervals_ms)
    sdnn = np.std(rr_intervals_ms, ddof=1)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals_ms) ** 2))
    return {
        "mean_rr": mean_rr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "rr_intervals": rr_intervals_ms
    }


def plot_hrv(rr_intervals):
    """Plot a histogram of the custom HRV R–R intervals in milliseconds."""
    plt.figure(figsize=(10, 6))
    plt.hist(rr_intervals, bins=20, edgecolor="black")
    plt.title("Histogram of R–R Intervals (Custom HRV) [ms]")
    plt.xlabel("R–R Interval (ms)")
    plt.ylabel("Frequency")
    plt.show()


def plot_hrv_neurokit_extensive(processed_data):
    """
    Generate extensive NeuroKit2 HRV plots.

    This function calls:
      - nk.hrv() to generate aggregated HRV indices and plots,
      - nk.hrv_frequency() for frequency-domain analysis plots, and
      - nk.hrv_nonlinear() for non-linear analysis plots.
    All functions are called with show=True.

    Returns a dictionary containing the DataFrames returned from each call.
    """
    fs = processed_data["fs"]
    peaks = processed_data["peaks"]
    peaks_list = peaks.tolist() if hasattr(peaks, "tolist") else peaks

    print("Generating NeuroKit2 aggregated HRV analysis plots...")
    hrv_all = nk.hrv(peaks_list, sampling_rate=fs, show=True)

    print("Generating NeuroKit2 frequency-domain HRV plots...")
    hrv_freq = nk.hrv_frequency(peaks_list, sampling_rate=fs, show=True)

    print("Generating NeuroKit2 non-linear HRV analysis plots...")
    hrv_nonlinear = nk.hrv_nonlinear(peaks_list, sampling_rate=fs, show=True)

    return {"hrv_all": hrv_all, "hrv_freq": hrv_freq, "hrv_nonlinear": hrv_nonlinear}


def plot_welch_psd(processed_data, signal="ecg_filtered", **kwargs):
    """
    Compute and plot the power spectral density (PSD) of a selected signal using Welch's method.

    Parameters:
      processed_data : dict
          Dictionary returned by process_ecg.
      signal : str, optional
          Key in processed_data that holds the desired signal (default is "ecg_filtered").
      **kwargs : dict
          Additional keyword arguments to pass to scipy.signal.welch (e.g., window, nperseg, noverlap, nfft, detrend).

    Returns:
      f : ndarray
          Array of sample frequencies.
      Pxx : ndarray
          Power spectral density of the signal.
    """
    fs = processed_data["fs"]
    x = processed_data.get(signal)
    if x is None:
        raise ValueError(f"Signal '{signal}' not found in processed_data")
    f, Pxx = welch(x, fs=fs, **kwargs)

    plt.figure(figsize=(10, 6))
    plt.semilogy(f, Pxx)
    plt.xlabel("Frequency (Hz)")
    default_scaling = kwargs.get("scaling", "density")
    ylabel = "PSD (V²/Hz)" if default_scaling == "density" else "Power (V²)"
    plt.ylabel(ylabel)
    plt.title(f"Power Spectral Density using Welch's Method ({signal})")
    plt.grid(True)
    plt.show()
    return f, Pxx


# Optional main routine for testing the module directly.
def main():
    root = Tk()
    root.withdraw()
    filepath = askopenfilename(
        title="Select OpenSignals File",
        filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
    )
    if not filepath:
        raise ValueError("No file was selected.")
    df, metadata = load_opensignals_file(filepath)
    print("Metadata extracted:")
    print(json.dumps(metadata, indent=2))
    processed_data = process_ecg(df, metadata)
    plot_ecg_results(processed_data)
    hrv_metrics = compute_hrv(processed_data)
    print("Custom HRV Metrics (ms):")
    print(f"  Mean R–R Interval: {hrv_metrics['mean_rr']:.3f} ms")
    print(f"  SDNN: {hrv_metrics['sdnn']:.3f} ms")
    print(f"  RMSSD: {hrv_metrics['rmssd']:.3f} ms")
    plot_hrv(hrv_metrics["rr_intervals"])
    hrv_nk = plot_hrv_neurokit_extensive(processed_data)
    print("NeuroKit2 HRV Metrics:")
    print(hrv_nk)
    # Example call of Welch PSD plot on filtered ECG signal.
    plot_welch_psd(processed_data, signal="ecg_filtered", window='hann', nperseg=256)


if __name__ == "__main__":
    main()
