#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import butter, filtfilt, find_peaks, welch
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def load_opensignals_file(filepath):
    """
    Load an OpenSignals Text File.
    Returns a DataFrame of numeric data and metadata.
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()

    header_lines = []
    data_lines = []
    header_end_index = None

    for i, line in enumerate(lines):
        if "# EndOfHeader" in line:
            header_end_index = i
            break
        header_lines.append(line)

    if header_end_index is None:
        raise ValueError("Could not find '# EndOfHeader' in the file.")

    data_lines = [line.strip() for line in lines[header_end_index + 1:] if line.strip()]

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
    """Convert raw ECG signal to millivolts."""
    return (raw_signal / (2 ** 10) - 0.5) * (3 / 1019) * 1000

def bandpass_filter(data, fs, lowcut=5, highcut=15, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)

def process_ecg(df, metadata):
    """
    Process the ECG data.
    Returns a dict with fs, t, ecg_converted, ecg_filtered,
    ecg_squared, ecg_integrated, and peaks via custom Pan–Tompkins.
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

    ecg_converted = apply_ecg_transfer_function(raw_ecg)
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
    """Plot the ECG processing stages and detected peaks."""
    t = processed_data["t"]
    ecg_converted = processed_data["ecg_converted"]
    ecg_filtered = processed_data["ecg_filtered"]
    ecg_squared = processed_data["ecg_squared"]
    ecg_integrated = processed_data["ecg_integrated"]
    peaks = processed_data["peaks"]

    plt.figure(figsize=(12,6))
    plt.plot(t, ecg_converted, label="Original ECG (mV)")
    plt.title("Original ECG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(t, ecg_filtered, label="Filtered ECG")
    plt.title("Bandpass Filtered ECG (5–15 Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(t, ecg_squared, label="Squared ECG")
    plt.title("Squared ECG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(t, ecg_integrated, label="Integrated ECG")
    thr = 0.5*np.max(ecg_integrated)
    plt.axhline(thr, color="red", linestyle="--", label="Threshold")
    plt.plot(t[peaks], ecg_integrated[peaks], "ko", label="Detected Peaks")
    plt.title("Integrated ECG with Peaks")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

def compute_hrv(processed_data):
    """Compute custom HRV metrics (mean NN, SDNN, RMSSD) in ms."""
    t = processed_data["t"]
    peaks = processed_data["peaks"]
    rr = np.diff(t[peaks]) * 1000  # convert to ms
    if len(rr) < 2:
        raise ValueError("Not enough RR intervals.")
    return {
        "mean_rr": np.mean(rr),
        "sdnn": np.std(rr, ddof=1),
        "rmssd": np.sqrt(np.mean(np.diff(rr)**2)),
        "rr_intervals": rr
    }

def plot_hrv(rr_intervals):
    """Plot histogram of RR intervals (HRV) in ms."""
    plt.figure(figsize=(10,6))
    plt.hist(rr_intervals, bins=20, edgecolor="black")
    plt.title("Histogram of R–R Intervals [ms]")
    plt.xlabel("R–R Interval (ms)")
    plt.ylabel("Frequency")
    plt.show()

def plot_hrv_neurokit_extensive(
    processed_data,
    interpolation_rate: float = 4,
    psd_method: str = "welch"
):
    """
    Generate NeuroKit2 HRV plots:
      - hrv (aggregated)
      - hrv_frequency (with interpolation check & 5-min windows)
      - hrv_nonlinear
    """
    fs = processed_data["fs"]
    peaks = processed_data["peaks"]
    peaks_list = peaks.tolist() if hasattr(peaks, "tolist") else peaks

    # 1) Aggregated
    print("Generating aggregated HRV indices/plots…")
    hrv_all = nk.hrv(peaks_list, sampling_rate=fs, show=True)

    # 2) Frequency-domain with interpolation & segmentation
    if interpolation_rate < 4:
        print(f"⚠️ interpolation_rate={interpolation_rate} Hz too low; using 4 Hz instead.")
        interpolation_rate = 4
    print(f"Frequency-domain using interpolation_rate={interpolation_rate} Hz, method={psd_method}")

    total_time = processed_data["t"][-1]
    hrv_freq_segments = []
    if total_time >= 300:
        starts = np.arange(0, total_time, 300)
        for start in starts:
            end = min(start + 300, total_time)
            idx0 = int(start * fs)
            idx1 = int(end   * fs)
            seg_peaks = [p for p in peaks_list if idx0 <= p < idx1]
            if len(seg_peaks) < 3:
                print(f"  • Window {start:.0f}-{end:.0f}s: skip ({len(seg_peaks)} peaks)")
                continue
            print(f"  • Window {start:.0f}-{end:.0f}s: {len(seg_peaks)} peaks → PSD")
            freq_df = nk.hrv_frequency(
                seg_peaks,
                sampling_rate=fs,
                show=True,
                interpolation_rate=interpolation_rate,
                psd_method=psd_method
            )
            hrv_freq_segments.append(freq_df)
        hrv_freq = pd.concat(hrv_freq_segments, ignore_index=True) if hrv_freq_segments else pd.DataFrame()
    else:
        print("Generating single-block frequency-domain HRV plot…")
        hrv_freq = nk.hrv_frequency(
            peaks_list,
            sampling_rate=fs,
            show=True,
            interpolation_rate=interpolation_rate,
            psd_method=psd_method
        )

    # 3) Non-linear
    print("Generating non-linear HRV plots…")
    hrv_nonlinear = nk.hrv_nonlinear(peaks_list, sampling_rate=fs, show=True)

    return {
        "hrv_all": hrv_all,
        "hrv_freq": hrv_freq,
        "hrv_nonlinear": hrv_nonlinear
    }

def plot_welch_psd(processed_data, signal="ecg_filtered", **kwargs):
    """
    Plot PSD of processed_data[signal] using scipy.signal.welch.
    Returns f, Pxx.
    """
    fs = processed_data["fs"]
    x = processed_data.get(signal)
    if x is None:
        raise ValueError(f"Signal '{signal}' not found")
    f, Pxx = welch(x, fs=fs, **kwargs)

    plt.figure(figsize=(10,6))
    plt.semilogy(f, Pxx)
    plt.title(f"Welch PSD ({signal})")
    plt.xlabel("Frequency (Hz)")
    scaling = kwargs.get("scaling", "density")
    plt.ylabel("PSD (V²/Hz)" if scaling=="density" else "Power (V²)")
    plt.grid(True)
    plt.show()
    return f, Pxx

# Optional self-test
if __name__ == "__main__":
    root = Tk(); root.withdraw()
    fp = askopenfilename(title="Select OpenSignals File", filetypes=[("Text files","*.txt")])
    df, md = load_opensignals_file(fp)
    pd_data = process_ecg(df, md)
    plot_ecg_results(pd_data)
    hr = compute_hrv(pd_data)
    print(hr)
    plot_hrv(hr["rr_intervals"])
    nkr = plot_hrv_neurokit_extensive(pd_data)
    print(nkr)
    plot_welch_psd(pd_data, window="hann", nperseg=256)
