#!/usr/bin/env python3
import sys
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import ecg_analysis


def main():
    # Interactive numbered prompt.
    print("Choose Plot Option:")
    print("1: Plot ECG analysis (multiple ECG plots)")
    print("2: Plot HRV analysis (Custom HRV histogram + NeuroKit2 extensive HRV plots)")
    print("3: Plot Both ECG and HRV")
    print("4: No plots")
    print("5: Plot Welch PSD (on filtered ECG signal)")
    user_choice = input("Enter your choice (1/2/3/4/5): ").strip()

    choice_map = {
        "1": "ecg",
        "2": "hrv",
        "3": "both",
        "4": "none",
        "5": "welch"
    }
    plot_option = choice_map.get(user_choice, "none")

    # File selection.
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filepath = askopenfilename(
        title="Select OpenSignals File",
        filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
    )
    if not filepath:
        print("No file was selected. Exiting.")
        sys.exit(1)

    df, metadata = ecg_analysis.load_opensignals_file(filepath)
    processed_data = ecg_analysis.process_ecg(df, metadata)

    # Compute and display custom HRV metrics.
    custom_hrv = ecg_analysis.compute_hrv(processed_data)
    print("Custom HRV Metrics (ms):")
    print(f"  Mean R–R Interval: {custom_hrv['mean_rr']:.3f} ms")
    print(f"  SDNN: {custom_hrv['sdnn']:.3f} ms")
    print(f"  RMSSD: {custom_hrv['rmssd']:.3f} ms")

    # Compute and display extensive NeuroKit2 HRV analyses.
    neurokit_hrv = ecg_analysis.plot_hrv_neurokit_extensive(processed_data)
    print("\nNeuroKit2 HRV Metrics:")
    print(neurokit_hrv)

    # Display plots based on user's choice.
    if plot_option == "ecg":
        ecg_analysis.plot_ecg_results(processed_data)
    elif plot_option == "hrv":
        ecg_analysis.plot_hrv(custom_hrv["rr_intervals"])
        # Extensive NeuroKit2 HRV plots were generated above.
    elif plot_option == "both":
        ecg_analysis.plot_ecg_results(processed_data)
        ecg_analysis.plot_hrv(custom_hrv["rr_intervals"])
        # Extensive NeuroKit2 HRV plots were generated above.
    elif plot_option == "welch":
        # Plot Welch PSD (default on filtered ECG signal) with some default parameters.
        ecg_analysis.plot_welch_psd(processed_data, signal="ecg_filtered", window='hann', nperseg=256)
    else:
        print("No plots selected.")


if __name__ == "__main__":
    main()
