import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from scipy import stats

# Load dataset
df = pd.read_csv('_all_data_ (1).csv')

baseline = df[df['state'].str.lower() == 'baseline']
postural = df[df['state'].str.lower() == 'postural']

print("=== DIAGNOSTIC CHECK ===")
print(f"Baseline rows: {len(baseline)}")
print(f"Postural rows: {len(postural)}")
print()

# ----- EDA -----
baseline_eda = baseline['eda_microsiemens'].dropna().values
postural_eda = postural['eda_microsiemens'].dropna().values

print(f"Baseline EDA - count: {len(baseline_eda)}, mean: {np.mean(baseline_eda):.6f}, std: {np.std(baseline_eda):.6f}")
print(f"Postural EDA - count: {len(postural_eda)}, mean: {np.mean(postural_eda):.6f}, std: {np.std(postural_eda):.6f}")
print()

baseline_eda_clean = baseline_eda - np.mean(baseline_eda)
postural_eda_clean = postural_eda - np.mean(postural_eda)

fft_b_eda = fft(baseline_eda_clean)
fft_p_eda = fft(postural_eda_clean)

freq_b_eda = fftfreq(len(baseline_eda_clean))
freq_p_eda = fftfreq(len(postural_eda_clean))

mask_b = (freq_b_eda >= 0) & (freq_b_eda <= 0.1)
mask_p = (freq_p_eda >= 0) & (freq_p_eda <= 0.1)

b_eda_fft_vals = np.abs(fft_b_eda[mask_b])
p_eda_fft_vals = np.abs(fft_p_eda[mask_p])

print(f"Baseline EDA FFT (0-0.1Hz) - n: {len(b_eda_fft_vals)}, mean: {np.mean(b_eda_fft_vals):.4f}")
print(f"Postural EDA FFT (0-0.1Hz) - n: {len(p_eda_fft_vals)}, mean: {np.mean(p_eda_fft_vals):.4f}")

t_eda, p_eda = stats.ttest_ind(b_eda_fft_vals, p_eda_fft_vals, equal_var=False)
print(f"EDA t-test: t={t_eda:.4f}, p={p_eda:.6f}")
print()
print("=" * 50)
print()

# ----- RR -----
baseline_rr = baseline['rr_interval_milliseconds'].dropna().values
postural_rr = postural['rr_interval_milliseconds'].dropna().values

print(f"Baseline RR - count: {len(baseline_rr)}, mean: {np.mean(baseline_rr):.4f}, std: {np.std(baseline_rr):.4f}")
print(f"Postural RR - count: {len(postural_rr)}, mean: {np.mean(postural_rr):.4f}, std: {np.std(postural_rr):.4f}")
print()

