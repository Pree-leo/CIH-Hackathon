import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv('/content/_all_data_ (1).csv')

# Check first 5 rows
print(df.head())

# Check all column names
print(df.columns)

# Check how many rows
print(df.shape)

# Check unique states
print(df['state'].unique())

# Check unique subjects
print(df['subject'].unique())

# Check how many rows per state
print(df['state'].value_counts())

print(df['eda_microsiemens'].head(10))
print(df['rr_interval_milliseconds'].head(10))

# Plot EDA signal
plt.figure(figsize=(12,4))
plt.plot(df['eda_microsiemens'])
plt.title('EDA Signal')
plt.xlabel('Sample')
plt.ylabel('EDA (microsiemens)')
plt.show()

# Plot RR Interval signal
plt.figure(figsize=(12,4))
plt.plot(df['rr_interval_milliseconds'])
plt.title('RR Interval Signal')
plt.xlabel('Sample')
plt.ylabel('RR Interval (ms)')
plt.show()

from scipy.fft import fft, fftfreq

# Extract EDA signal
eda_signal = df['eda_microsiemens'].dropna().values

# Apply FFT
fft_values = fft(eda_signal)
frequencies = fftfreq(len(eda_signal))

# Plot FFT
plt.figure(figsize=(12,4))
plt.plot(frequencies[:len(frequencies)//2],
         np.abs(fft_values[:len(fft_values)//2]))
plt.title('FFT of EDA Signal')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.show()

rr_signal = df['rr_interval_milliseconds'].dropna().values

fft_rr = fft(rr_signal)
freq_rr = fftfreq(len(rr_signal))

plt.figure(figsize=(12,4))
plt.plot(freq_rr[:len(freq_rr)//2],
         np.abs(fft_rr[:len(fft_rr)//2]))
plt.title('FFT of RR Interval Signal')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.show()

print(df['state'].unique())

# Separate by state
baseline = df[df['state'] == 'baseline']
postural = df[df['state'] == 'postural']

# FFT for each state - EDA
for state, label in [(baseline,'baseline'), (postural,'postural')]:
    signal = state['eda_microsiemens'].dropna().values
    fft_vals = fft(signal)
    freqs = fftfreq(len(signal))
    plt.plot(freqs[:len(freqs)//2],
             np.abs(fft_vals[:len(fft_vals)//2]),
             label=label)

plt.title('EDA FFT by State')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

for state, label in [(baseline,'baseline'),
                     (postural,'postural')]:
    signal = state['rr_interval_milliseconds'].dropna().values
    signal= signal - np.mean(signal)
    fft_vals = fft(signal)
    freqs = fftfreq(len(signal))
    plt.plot(freqs[:len(freqs)//2],
             np.abs(fft_vals[:len(fft_vals)//2]),
             label=label)

plt.title('RR Interval FFT by State')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.legend()
plt.xlim(0.001, 0.1)
plt.ylim(0, 50000)
plt.figure(figsize=(12,4))
plt.show()

from scipy import stats

# Separate baseline and postural states
baseline = df[df['state'].str.lower() == 'baseline']
postural = df[df['state'].str.lower() == 'postural']

# ==========================
# EDA FFT T-TEST
# ==========================
baseline_eda_signal = baseline['eda_microsiemens'].dropna().values
postural_eda_signal = postural['eda_microsiemens'].dropna().values

# Remove DC component
baseline_eda_clean = baseline_eda_signal - np.mean(baseline_eda_signal)
postural_eda_clean = postural_eda_signal - np.mean(postural_eda_signal)

# FFT
baseline_eda_fft = np.abs(fft(baseline_eda_clean))
postural_eda_fft = np.abs(fft(postural_eda_clean))

# T-test
t_stat_eda, p_value_eda = stats.ttest_ind(
    baseline_eda_fft,
    postural_eda_fft,
    equal_var=False
)

print("===== EDA FFT T-Test =====")
print(f"T-statistic: {t_stat_eda:.4f}")
print(f"P-value: {p_value_eda:.6f}")

if p_value_eda < 0.05:
    print("✅ SIGNIFICANT difference (p < 0.05)")
else:
    print("❌ NOT significant (p >= 0.05)")

print()

# ==========================
# RR INTERVAL FFT T-TEST
# ==========================
baseline_rr_signal = baseline['rr_interval_milliseconds'].dropna().values
postural_rr_signal = postural['rr_interval_milliseconds'].dropna().values

# Remove DC component
baseline_rr_clean = baseline_rr_signal - np.mean(baseline_rr_signal)
postural_rr_clean = postural_rr_signal - np.mean(postural_rr_signal)

# FFT
baseline_rr_fft = np.abs(fft(baseline_rr_clean))
postural_rr_fft = np.abs(fft(postural_rr_clean))

# T-test
t_stat_rr, p_value_rr = stats.ttest_ind(
    baseline_rr_fft,
    postural_rr_fft,
    equal_var=False
)

print("===== RR Interval FFT T-Test =====")
print(f"T-statistic: {t_stat_rr:.4f}")
print(f"P-value: {p_value_rr:.6f}")

if p_value_rr < 0.05:
    print("✅ SIGNIFICANT difference (p < 0.05)")
else:
    print("❌ NOT significant (p >= 0.05)")
