import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1 - Load data
df = pd.read_csv('reseach data.csv')
print("Data loaded:", df.shape)

# Step 2 - Pick one subject, one state to start
subject1_baseline = df[(df['subject'] == 'subject1') & 
                       (df['state'] == 'baseline')]

subject1_postural = df[(df['subject'] == 'subject1') & 
                       (df['state'] == 'postural')]

# Step 3 - Extract EDA signal
eda_baseline = subject1_baseline['eda_microsiemens'].values
eda_postural = subject1_postural['eda_microsiemens'].values

print("Baseline EDA samples:", len(eda_baseline))
print("Postural EDA samples:", len(eda_postural))

# Step 4 - Apply FFT
def apply_fft(signal, sampling_rate=4):
    fft_result = np.fft.fft(signal)
    magnitude = np.abs(fft_result)
    n = len(signal)
    frequencies = np.fft.fftfreq(n, d=1/sampling_rate)
    # Only positive frequencies
    half = n // 2
    return frequencies[:half], magnitude[:half]

freq_base, mag_base = apply_fft(eda_baseline)
freq_post, mag_post = apply_fft(eda_postural)

# Step 5 - Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Time domain
axes[0,0].plot(eda_baseline, color='blue')
axes[0,0].set_title('EDA Baseline - Time Domain')
axes[0,0].set_xlabel('Samples')
axes[0,0].set_ylabel('EDA (microsiemens)')

axes[0,1].plot(eda_postural, color='red')
axes[0,1].set_title('EDA Postural - Time Domain')
axes[0,1].set_xlabel('Samples')
axes[0,1].set_ylabel('EDA (microsiemens)')

# Frequency domain
axes[1,0].plot(freq_base, mag_base, color='blue')
axes[1,0].set_title('EDA Baseline - After FFT')
axes[1,0].set_xlabel('Frequency (Hz)')
axes[1,0].set_ylabel('Magnitude')

axes[1,1].plot(freq_post, mag_post, color='red')
axes[1,1].set_title('EDA Postural - After FFT')
axes[1,1].set_xlabel('Frequency (Hz)')
axes[1,1].set_ylabel('Magnitude')

plt.tight_layout()
plt.savefig('fft_comparison.png')
#plt.show()
print("Plot saved as fft_comparison.png")

# Extract FFT features for all subjects and all signals
def extract_fft_features(signal, signal_name, sampling_rate=4):
    if len(signal) < 2:
        return {}
    fft_result = np.fft.fft(signal)
    magnitude = np.abs(fft_result)
    n = len(signal)
    frequencies = np.fft.fftfreq(n, d=1/sampling_rate)
    half = n // 2
    pos_mag = magnitude[:half]
    pos_freq = frequencies[:half]
    
    return {
        f'{signal_name}_fft_mean':    np.mean(pos_mag),
        f'{signal_name}_fft_max':     np.max(pos_mag),
        f'{signal_name}_fft_std':     np.std(pos_mag),
        f'{signal_name}_fft_energy':  np.sum(pos_mag**2),
        f'{signal_name}_dominant_freq': pos_freq[np.argmax(pos_mag)]
    }

# Process all 16 subjects
all_rows = []

for subject in df['subject'].unique():
    for state in ['baseline', 'postural']:
        subset = df[(df['subject'] == subject) & 
                    (df['state'] == state)]
        
        if len(subset) < 10:
            continue
        
        row = {'subject': subject, 'state': state}
        
        # Extract FFT features for each signal
        signals_info = {
            'eda':   ('eda_microsiemens', 4),
            'temp':  ('temperature_celcius', 4),
            'bvp':   ('bvp_light_absorption_nW', 64),
            'acc_x': ('accelerometer_x', 32),
            'acc_y': ('accelerometer_y', 32),
            'gyr_x': ('gyroscope_x', 32),
        }
        
        for name, (col, fs) in signals_info.items():
            signal = subset[col].values
            feats = extract_fft_features(signal, name, fs)
            row.update(feats)
        
        all_rows.append(row)

# Save features
feature_df = pd.DataFrame(all_rows)
feature_df.to_csv('fft_features.csv', index=False)
print("Features extracted:", feature_df.shape)
print(feature_df.head())


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Prepare X and y
X = feature_df.drop(columns=['subject','state']).values
y = feature_df['state'].values

# Encode labels: baseline=0, postural=1
le = LabelEncoder()
y = le.fit_transform(y)
print("Labels:", le.classes_)

# Clean missing/zero values
X = SimpleImputer(strategy='mean').fit_transform(X)

# Scale features
X = StandardScaler().fit_transform(X)

# Train and evaluate all models
models = {
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Naive Bayes':         GaussianNB(),
    'SVM':                 SVC(kernel='linear', probability=True),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

print("\n===== MODEL RESULTS =====")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    acc    = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\n{name}")
    print(f"  AUC:      {scores.mean():.2f} ± {scores.std():.2f}")
    print(f"  Accuracy: {acc.mean():.2f} ± {acc.std():.2f}")


    # Plot all signals for subject1 - baseline vs postural
signals_to_plot = {
    'EDA': 'eda_microsiemens',
    'Temperature': 'temperature_celcius',
    'BVP': 'bvp_light_absorption_nW',
    'Accelerometer X': 'accelerometer_x',
    'Accelerometer Y': 'accelerometer_y',
    'Accelerometer Z': 'accelerometer_z'
}

fig, axes = plt.subplots(6, 2, figsize=(16, 20))

for idx, (signal_name, col) in enumerate(signals_to_plot.items()):
    
    baseline_sig = subject1_baseline[col].values
    postural_sig = subject1_postural[col].values
    
    # Time domain
    axes[idx, 0].plot(baseline_sig, color='blue')
    axes[idx, 0].set_title(f'{signal_name} - Baseline')
    axes[idx, 0].set_xlabel('Samples')
    axes[idx, 0].set_ylabel(signal_name)
    
    # FFT
    freq_b, mag_b = apply_fft(baseline_sig)
    axes[idx, 1].plot(freq_b, mag_b, color='red')
    axes[idx, 1].set_title(f'{signal_name} - FFT')
    axes[idx, 1].set_xlabel('Frequency (Hz)')
    axes[idx, 1].set_ylabel('Magnitude')

plt.tight_layout()
plt.savefig('all_signals_fft.png')
print("All signals graph saved")

# ─────────────────────────────────────────
# BASELINE VS POSTURAL COMPARISON - ALL SIGNALS
# ─────────────────────────────────────────

signals_to_plot = {
    'EDA':            'eda_microsiemens',
    'Temperature':    'temperature_celcius',
    'BVP':            'bvp_light_absorption_nW',
    'Accelerometer X':'accelerometer_x',
    'Accelerometer Y':'accelerometer_y',
    'Accelerometer Z':'accelerometer_z'
}

fig, axes = plt.subplots(6, 4, figsize=(20, 24))

for idx, (signal_name, col) in enumerate(signals_to_plot.items()):

    baseline_sig = subject1_baseline[col].values
    postural_sig = subject1_postural[col].values

    # --- Time domain baseline ---
    axes[idx, 0].plot(baseline_sig, color='blue')
    axes[idx, 0].set_title(f'{signal_name} Baseline - Time')
    axes[idx, 0].set_xlabel('Samples')
    axes[idx, 0].set_ylabel(signal_name)

    # --- FFT baseline ---
    freq_b, mag_b = apply_fft(baseline_sig)
    axes[idx, 1].plot(freq_b, mag_b, color='blue')
    axes[idx, 1].set_title(f'{signal_name} Baseline - FFT')
    axes[idx, 1].set_xlabel('Frequency (Hz)')
    axes[idx, 1].set_ylabel('Magnitude')

    # --- Time domain postural ---
    axes[idx, 2].plot(postural_sig, color='red')
    axes[idx, 2].set_title(f'{signal_name} Postural - Time')
    axes[idx, 2].set_xlabel('Samples')
    axes[idx, 2].set_ylabel(signal_name)

    # --- FFT postural ---
    freq_p, mag_p = apply_fft(postural_sig)
    axes[idx, 3].plot(freq_p, mag_p, color='red')
    axes[idx, 3].set_title(f'{signal_name} Postural - FFT')
    axes[idx, 3].set_xlabel('Frequency (Hz)')
    axes[idx, 3].set_ylabel('Magnitude')

plt.tight_layout()
plt.savefig('baseline_vs_postural.png')
print("Baseline vs Postural comparison saved")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Train Random Forest and plot confusion matrix
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=le.classes_)
disp.plot()
plt.title('Random Forest - Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved")

# Feature importance from Random Forest
feature_names = feature_df.drop(
    columns=['subject','state']).columns

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:10]

plt.figure(figsize=(12, 6))
plt.bar(range(10), importances[indices], color='steelblue')
plt.xticks(range(10),
           [feature_names[i] for i in indices],
           rotation=45, ha='right')
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance graph saved")

from sklearn.model_selection import LeaveOneGroupOut

# LOGO-CV - Leave One Subject Out
logo = LeaveOneGroupOut()
groups = feature_df['subject'].values

print("\n===== LOGO-CV RESULTS =====")
for name, model in models.items():
    auc_scores = cross_val_score(
        model, X, y,
        cv=logo,
        groups=groups,
        scoring='roc_auc'
    )
    print(f"\n{name}")
    print(f"  LOGO AUC: {auc_scores.mean():.2f} ± {auc_scores.std():.2f}")
