import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from scipy import stats

# Load dataset
df = pd.read_csv('/content/_all_data_ (1).csv')

# Separate baseline and postural
baseline = df[df['state'].str.lower() == 'baseline']
postural = df[df['state'].str.lower() == 'postural']

# ===== EDA FFT T-TEST (0-0.1 Hz range) =====
baseline_eda_signal = baseline['eda_microsiemens'].dropna().values
postural_eda_signal = postural['eda_microsiemens'].dropna().values

baseline_eda_clean = baseline_eda_signal - np.mean(baseline_eda_signal)
postural_eda_clean = postural_eda_signal - np.mean(postural_eda_signal)

fft_baseline_eda = fft(baseline_eda_clean)
fft_postural_eda = fft(postural_eda_clean)

freq_baseline_eda = fftfreq(len(baseline_eda_clean))
freq_postural_eda = fftfreq(len(postural_eda_clean))

mask_baseline = (freq_baseline_eda >= 0) & (freq_baseline_eda <= 0.1)
mask_postural = (freq_postural_eda >= 0) & (freq_postural_eda <= 0.1)

baseline_eda_fft_limited = np.abs(fft_baseline_eda[mask_baseline])
postural_eda_fft_limited = np.abs(fft_postural_eda[mask_postural])

t_stat_eda, p_value_eda = stats.ttest_ind(
    baseline_eda_fft_limited,
    postural_eda_fft_limited,
    equal_var=False
)

print("===== EDA FFT T-Test (0-0.1 Hz range) =====")
print(f"T-statistic: {t_stat_eda:.4f}")
print(f"P-value: {p_value_eda:.6f}")
print(f"Baseline mean amplitude: {np.mean(baseline_eda_fft_limited):.2f}")
print(f"Postural mean amplitude: {np.mean(postural_eda_fft_limited):.2f}")

if p_value_eda < 0.05:
    print("✅ SIGNIFICANT difference (p < 0.05)")
else:
    print("❌ NOT significant (p >= 0.05)")

print()

# ===== RR INTERVAL FFT T-TEST (0-0.1 Hz range) =====
baseline_rr_signal = baseline['rr_interval_milliseconds'].dropna().values
postural_rr_signal = postural['rr_interval_milliseconds'].dropna().values

baseline_rr_clean = baseline_rr_signal - np.mean(baseline_rr_signal)
postural_rr_clean = postural_rr_signal - np.mean(postural_rr_signal)

fft_baseline_rr = fft(baseline_rr_clean)
fft_postural_rr = fft(postural_rr_clean)

freq_baseline_rr = fftfreq(len(baseline_rr_clean))
freq_postural_rr = fftfreq(len(postural_rr_clean))

mask_baseline_rr = (freq_baseline_rr >= 0) & (freq_baseline_rr <= 0.1)
mask_postural_rr = (freq_postural_rr >= 0) & (freq_postural_rr <= 0.1)

baseline_rr_fft_limited = np.abs(fft_baseline_rr[mask_baseline_rr])
postural_rr_fft_limited = np.abs(fft_postural_rr[mask_postural_rr])

t_stat_rr, p_value_rr = stats.ttest_ind(
    baseline_rr_fft_limited,
    postural_rr_fft_limited,
    equal_var=False
)

print("===== RR Interval FFT T-Test (0-0.1 Hz range) =====")
print(f"T-statistic: {t_stat_rr:.4f}")
print(f"P-value: {p_value_rr:.6f}")
print(f"Baseline mean amplitude: {np.mean(baseline_rr_fft_limited):.2f}")
print(f"Postural mean amplitude: {np.mean(postural_rr_fft_limited):.2f}")

if p_value_rr < 0.05:
    print("✅ SIGNIFICANT difference (p < 0.05)")
else:
    print("❌ NOT significant (p >= 0.05)")
