import h5py
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import json
import numpy as np
import neurokit2 as nk
from scipy.signal import butter, filtfilt, find_peaks

#Simulate ECG signal
#Sine formula: y(t) = Asin(2πft + φ) = Asin(ωt + φ)


length = 15
fs = 125
ecg = nk.ecg_simulate(duration=length, sampling_rate=fs, heart_rate=60)
t = np.linspace(0, stop=15, num=length*fs, endpoint=False)

#noise to be added to the signal.
sine = np.sin(2*np.pi*50*t)
amplitude = 0.1
#TODO Add high frequency noise
# Add the noise to the clean ECG signal
ecg_noisy = ecg + amplitude * sine

# Plot the clean and noisy ECG signals
plt.figure(figsize=(12, 6))
plt.plot(t, ecg, label="Clean ECG")
plt.plot(t, ecg_noisy, label="Noisy ECG", alpha=0.75)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("ECG Signal with Added 50Hz Noise")
plt.legend()
plt.show()



# Step 1: Preprocessing - Bandpass Filter
# Define a bandpass filter function (e.g., 5–15 Hz, which is common for detecting the QRS complex).
def bandpass_filter(data, fs, lowcut=5, highcut=15, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)

# Filter the noisy ECG signal
ecg_filtered = bandpass_filter(ecg_noisy, fs)

# Plot the filtered signal
plt.figure(figsize=(12, 6))
plt.plot(t, ecg_filtered, label="Filtered ECG")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Filtered ECG Signal (Bandpass 5–15 Hz)")
plt.legend()
plt.show()

# Step 2: Squaring the Signal
# Squaring enhances large differences in signal amplitude (emphasizes the QRS peaks).
ecg_squared = ecg_filtered ** 2

# Plot the squared signal
plt.figure(figsize=(12, 6))
plt.plot(t, ecg_squared, label="Squared ECG")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Squared ECG Signal")
plt.legend()
plt.show()

# Step 3: Moving Window Integration
# Integration using a moving window (here, about 150 ms is a common choice for integration).
window_size = int(0.15 * fs)  # 150 ms window
ecg_integrated = np.convolve(ecg_squared, np.ones(window_size)/window_size, mode="same")

# Plot the integrated signal
plt.figure(figsize=(12, 6))
plt.plot(t, ecg_integrated, label="Integrated ECG")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Integrated ECG Signal (Moving Window Integration)")
plt.legend()
plt.show()

# Step 4: Thresholding and Peak Detection
# Set a threshold (for example, 50% of the maximum of the integrated signal) to aid in detection.
threshold = 0.5 * np.max(ecg_integrated)

# Find peaks that are above the threshold and separated by at least 200 ms to avoid multiple detections for one beat.
peaks, properties = find_peaks(ecg_integrated, height=threshold, distance=int(0.2 * fs))

# Plot the integrated signal with the threshold and marked peaks.
plt.figure(figsize=(12, 6))
plt.plot(t, ecg_integrated, label="Integrated ECG")
plt.axhline(threshold, color="red", linestyle="--", label="Threshold")
plt.plot(t[peaks], ecg_integrated[peaks], 'ko', label="Detected Peaks")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Integrated ECG with Threshold and Detected Peaks")
plt.legend()
plt.show()

# Finally, plot the detected R-peaks on the filtered ECG signal to verify that the correct peaks are being identified.
plt.figure(figsize=(12, 6))
plt.plot(t, ecg_filtered, label="Filtered ECG")
plt.plot(t[peaks], ecg_filtered[peaks], 'ro', label="Detected R-Peaks")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Detected R-Peaks on Filtered ECG Signal")
plt.legend()
plt.show()