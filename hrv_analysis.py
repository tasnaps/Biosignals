#!/usr/bin/env python3
import sys
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import ecg_analysis

def main():
    # 1) Plot choice
    print("Choose Plot Option:")
    print("1: Plot ECG analysis")
    print("2: Plot HRV analysis (Custom + NeuroKit2)")
    print("3: Plot Both ECG and HRV")
    print("4: No plots")
    print("5: Plot Welch PSD")
    choice = input("Enter your choice (1/2/3/4/5): ").strip()
    plot_option = {"1":"ecg","2":"hrv","3":"both","4":"none","5":"welch"}.get(choice, "none")

    # 2) Load file
    root = Tk(); root.withdraw(); root.attributes("-topmost", True)
    filepath = askopenfilename(title="Select OpenSignals File",
                               filetypes=[("Text files","*.txt"),("All files","*.*")])
    if not filepath:
        print("No file selected. Exiting."); sys.exit(1)

    df, metadata = ecg_analysis.load_opensignals_file(filepath)
    processed = ecg_analysis.process_ecg(df, metadata)

    # 3) Peak detection method
    print("\nChoose R‑peak detection method:")
    print("1: Custom Pan–Tompkins")
    print("2: NeuroKit2 ecg_peaks")
    m = input("Enter your choice (1/2): ").strip()
    if m == "2":
        import neurokit2 as nk
        print("Running NeuroKit2 peak detection…")
        cleaned = nk.ecg_clean(processed["ecg_converted"], sampling_rate=processed["fs"])
        peaks, info = nk.ecg_peaks(cleaned,
                                   sampling_rate=processed["fs"],
                                   correct_artifacts=True)
        processed["peaks"] = info["ECG_R_Peaks"]
        print(f"Detected {len(processed['peaks'])} R‑peaks with NeuroKit2.")
    else:
        print("Using custom Pan–Tompkins peaks.")

    # 4) Custom HRV metrics
    custom = ecg_analysis.compute_hrv(processed)
    print("\nCustom HRV Metrics (ms):")
    print(f"  Mean R–R: {custom['mean_rr']:.3f} ms")
    print(f"  SDNN:     {custom['sdnn']:.3f} ms")
    print(f"  RMSSD:    {custom['rmssd']:.3f} ms")

    # 5) NeuroKit2 HRV (ask interpolation & method)
    interp = input("\nInterpolation rate for freq-domain (Hz, default=4): ").strip()
    interp_rate = float(interp) if interp else 4.0
    method = input("PSD method [welch/burg/lomb/multitapers] (default=welch): ").strip() or "welch"

    nk_results = ecg_analysis.plot_hrv_neurokit_extensive(
        processed,
        interpolation_rate=interp_rate,
        psd_method=method
    )
    print("\nNeuroKit2 HRV Metrics:")
    print(nk_results)

    # 6) Finally: display chosen plots
    if plot_option == "ecg":
        ecg_analysis.plot_ecg_results(processed)
    elif plot_option == "hrv":
        ecg_analysis.plot_hrv(custom["rr_intervals"])
    elif plot_option == "both":
        ecg_analysis.plot_ecg_results(processed)
        ecg_analysis.plot_hrv(custom["rr_intervals"])
    elif plot_option == "welch":
        ecg_analysis.plot_welch_psd(processed,
                                    signal="ecg_filtered",
                                    window="hann",
                                    nperseg=256)
    else:
        print("No plots selected—done.")

if __name__ == "__main__":
    main()
