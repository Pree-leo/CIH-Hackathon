import numpy as np

def hedges_g(group1, group2):
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    pooled_sd = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    cohens_d = (mean1 - mean2) / pooled_sd
    correction = 1 - (3 / (4*(n1+n2) - 9))
    g = cohens_d * correction

    return g, n1, n2

# RR interval mean comparisons
feature_col = 'rr_interval_milliseconds_mean'

sym = df[df['state']=='sympathetic'][feature_col].dropna().values
para = df[df['state']=='parasympathetic'][feature_col].dropna().values
relaxed = df[df['state']=='relaxed'][feature_col].dropna().values

print("===== Hedges' G: RR Interval Mean =====")

g1, n1, n2 = hedges_g(sym, para)
print(f"Sympathetic vs Parasympathetic: g={g1:.4f} (n1={n1}, n2={n2})")

g2, n1, n2 = hedges_g(sym, relaxed)
print(f"Sympathetic vs Relaxed: g={g2:.4f} (n1={n1}, n2={n2})")

g3, n1, n2 = hedges_g(para, relaxed)
print(f"Parasympathetic vs Relaxed: g={g3:.4f} (n1={n1}, n2={n2})")


feature_col_eda = 'eda_microsiemens_mean'

sym_eda = df[df['state']=='sympathetic'][feature_col_eda].dropna().values
para_eda = df[df['state']=='parasympathetic'][feature_col_eda].dropna().values
relaxed_eda = df[df['state']=='relaxed'][feature_col_eda].dropna().values

print("\n===== Hedges' G: EDA Mean =====")

g1e, n1, n2 = hedges_g(sym_eda, para_eda)
print(f"Sympathetic vs Parasympathetic: g={g1e:.4f} (n1={n1}, n2={n2})")

g2e, n1, n2 = hedges_g(sym_eda, relaxed_eda)
print(f"Sympathetic vs Relaxed: g={g2e:.4f} (n1={n1}, n2={n2})")

g3e, n1, n2 = hedges_g(para_eda, relaxed_eda)
print(f"Parasympathetic vs Relaxed: g={g3e:.4f} (n1={n1}, n2={n2})")


import numpy as np
import matplotlib.pyplot as plt

def get_rr_by_subjects(subject_list, posture):
    subset = raw_df[
        (raw_df['subject'].isin(subject_list)) & 
        (raw_df['state'] == posture)
    ]
    return subset['rr_interval_milliseconds'].dropna().values

# Get subject lists from feature file
sym_subjects = df[df['state']=='sympathetic']['subject'].unique()
para_subjects = df[df['state']=='parasympathetic']['subject'].unique()
relaxed_subjects = df[df['state']=='relaxed']['subject'].unique()

rr_sym = get_rr_by_subjects(sym_subjects, 'postural')
rr_para = get_rr_by_subjects(para_subjects, 'postural')
rr_relaxed = get_rr_by_subjects(relaxed_subjects, 'baseline')

print("Sympathetic RR samples:", len(rr_sym))
print("Parasympathetic RR samples:", len(rr_para))
print("Relaxed RR samples:", len(rr_relaxed))


import shutil

shutil.copy('extracted_features_3label.csv', 
            '/content/drive/MyDrive/extracted_features_3label.csv')
shutil.copy('_all_data_ (1).csv.xls', 
            '/content/drive/MyDrive/reseach_data.csv')

print("Files saved to Drive permanently")

def poincare_sd(rr_signal):
    rr_n = rr_signal[:-1]
    rr_n1 = rr_signal[1:]
    
    diff = rr_n1 - rr_n
    sd1 = np.std(diff) / np.sqrt(2)
    sd2 = np.sqrt(2 * np.var(rr_signal) - (np.std(diff)**2) / 2)
    
    return sd1, sd2, rr_n, rr_n1

# Combined plot, all 3 states overlapped
plt.figure(figsize=(8,8))

sd1_sym, sd2_sym, rr_n_sym, rr_n1_sym = poincare_sd(rr_sym)
plt.scatter(rr_n_sym, rr_n1_sym, s=5, alpha=0.4, color='red', 
            label=f'Sympathetic (SD1={sd1_sym:.1f}, SD2={sd2_sym:.1f})')

sd1_para, sd2_para, rr_n_para, rr_n1_para = poincare_sd(rr_para)
plt.scatter(rr_n_para, rr_n1_para, s=5, alpha=0.4, color='blue', 
            label=f'Parasympathetic (SD1={sd1_para:.1f}, SD2={sd2_para:.1f})')

sd1_relaxed, sd2_relaxed, rr_n_relaxed, rr_n1_relaxed = poincare_sd(rr_relaxed)
plt.scatter(rr_n_relaxed, rr_n1_relaxed, s=5, alpha=0.4, color='green', 
            label=f'Relaxed (SD1={sd1_relaxed:.1f}, SD2={sd2_relaxed:.1f})')

plt.xlabel('RR(n) ms')
plt.ylabel('RR(n+1) ms')
plt.title('Poincaré Plot — RR Interval by State')
plt.legend()
plt.savefig('poincare_all_states.png')
plt.show()

print(f"\nSympathetic:     SD1={sd1_sym:.2f}, SD2={sd2_sym:.2f}, ratio={sd1_sym/sd2_sym:.3f}")
print(f"Parasympathetic: SD1={sd1_para:.2f}, SD2={sd2_para:.2f}, ratio={sd1_para/sd2_para:.3f}")
print(f"Relaxed:         SD1={sd1_relaxed:.2f}, SD2={sd2_relaxed:.2f}, ratio={sd1_relaxed/sd2_relaxed:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(18,6))

for ax, (rr, label, color) in zip(axes, [
    (rr_sym, 'Sympathetic', 'red'),
    (rr_para, 'Parasympathetic', 'blue'),
    (rr_relaxed, 'Relaxed', 'green')
]):
    sd1, sd2, rr_n, rr_n1 = poincare_sd(rr)
    ax.scatter(rr_n, rr_n1, s=5, alpha=0.4, color=color)
    ax.set_title(f'{label}\nSD1={sd1:.1f}, SD2={sd2:.1f}')
    ax.set_xlabel('RR(n) ms')
    ax.set_ylabel('RR(n+1) ms')

plt.tight_layout()
plt.savefig('poincare_separated.png')
plt.show()
