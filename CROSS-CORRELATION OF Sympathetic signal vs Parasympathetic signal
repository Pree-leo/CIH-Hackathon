def get_signal_by_subjects(subject_list, posture, signal_col):
    subset = raw_df[
        (raw_df['subject'].isin(subject_list)) &
        (raw_df['state'] == posture)
    ]
    return subset[signal_col].dropna().values

sym_subjects  = df[df['state']=='sympathetic']['subject'].unique()
para_subjects = df[df['state']=='parasympathetic']['subject'].unique()
relaxed_subjects = df[df['state']=='relaxed']['subject'].unique()

# EDA signals
eda_sym     = get_signal_by_subjects(sym_subjects,  'postural', 'eda_microsiemens')
eda_para    = get_signal_by_subjects(para_subjects, 'postural', 'eda_microsiemens')

# RR interval signals
rr_sym      = get_signal_by_subjects(sym_subjects,  'postural', 'rr_interval_milliseconds')
rr_para     = get_signal_by_subjects(para_subjects, 'postural', 'rr_interval_milliseconds')

print("EDA  Sym:", len(eda_sym),  "| EDA  Para:", len(eda_para))
print("RR   Sym:", len(rr_sym),   "| RR   Para:", len(rr_para))

def normalize(sig):
    return (sig - np.mean(sig)) / np.std(sig)

def pearson_correlation(sig1, sig2):
    min_len = min(len(sig1), len(sig2))
    r, p = pearsonr(sig1[:min_len], sig2[:min_len])
    return r, p

def self_correlation(sig):
    half = len(sig) // 2
    r, p = pearsonr(sig[:half], sig[half:half*2])
    return r, p

print("===== CROSS-CORRELATION =====")
print("(Sympathetic signal vs Parasympathetic signal)")
print()

# EDA Cross-correlation
eda_sym_n  = normalize(eda_sym)
eda_para_n = normalize(eda_para)
r_eda_cross, p_eda_cross = pearson_correlation(eda_sym_n, eda_para_n)
print(f"EDA  Cross-correlation (Sym vs Para): r={r_eda_cross:.4f}, p={p_eda_cross:.6f}")

# RR Cross-correlation
rr_sym_n   = normalize(rr_sym)
rr_para_n  = normalize(rr_para)
r_rr_cross, p_rr_cross = pearsonr(rr_sym_n[:min(len(rr_sym_n),len(rr_para_n))],
                                    rr_para_n[:min(len(rr_sym_n),len(rr_para_n))])
print(f"RR   Cross-correlation (Sym vs Para): r={r_rr_cross:.4f}, p={p_rr_cross:.6f}")

print()
print("===== SELF-CORRELATION (control) =====")
print("(Sympathetic signal vs itself, split in half)")
print()

r_eda_self_sym, p_eda_self_sym = self_correlation(eda_sym_n)
print(f"EDA  Self-correlation (Sym halves):   r={r_eda_self_sym:.4f}, p={p_eda_self_sym:.6f}")

r_rr_self_sym, p_rr_self_sym = self_correlation(rr_sym_n)
print(f"RR   Self-correlation (Sym halves):   r={r_rr_self_sym:.4f}, p={p_rr_self_sym:.6f}")

r_eda_self_para, p_eda_self_para = self_correlation(eda_para_n)
print(f"EDA  Self-correlation (Para halves):  r={r_eda_self_para:.4f}, p={p_eda_self_para:.6f}")

r_rr_self_para, p_rr_self_para = self_correlation(rr_para_n)
print(f"RR   Self-correlation (Para halves):  r={r_rr_self_para:.4f}, p={p_rr_self_para:.6f}")

print()
print("===== SUMMARY TABLE =====")
print(f"{'Comparison':<40} {'r':>8} {'p':>12}")
print("-" * 62)
print(f"{'EDA Cross (Sym vs Para)':<40} {r_eda_cross:>8.4f} {p_eda_cross:>12.6f}")
print(f"{'RR  Cross (Sym vs Para)':<40} {r_rr_cross:>8.4f} {p_rr_cross:>12.6f}")
print(f"{'EDA Self (Sym halves)':<40} {r_eda_self_sym:>8.4f} {p_eda_self_sym:>12.6f}")
print(f"{'RR  Self (Sym halves)':<40} {r_rr_self_sym:>8.4f} {p_rr_self_sym:>12.6f}")
print(f"{'EDA Self (Para halves)':<40} {r_eda_self_para:>8.4f} {p_eda_self_para:>12.6f}")
print(f"{'RR  Self (Para halves)':<40} {r_rr_self_para:>8.4f} {p_rr_self_para:>12.6f}")

def lag_cross_correlation(sig1, sig2, label, color1, color2):
    min_len = min(len(sig1), len(sig2))
    s1 = sig1[:min_len]
    s2 = sig2[:min_len]
    correlation = np.correlate(s1, s2, mode='full')
    lags = np.arange(-min_len+1, min_len)
    return lags, correlation

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# EDA cross-correlation lag plot
lags_eda, corr_eda = lag_cross_correlation(eda_sym_n, eda_para_n,
                      'EDA', 'red', 'blue')
axes[0].plot(lags_eda, corr_eda, color='purple')
axes[0].axvline(0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_title('EDA Cross-Correlation: Sympathetic vs Parasympathetic')
axes[0].set_xlabel('Lag (samples)')
axes[0].set_ylabel('Correlation')

# RR cross-correlation lag plot
lags_rr, corr_rr = lag_cross_correlation(rr_sym_n, rr_para_n,
                    'RR', 'red', 'blue')
axes[1].plot(lags_rr, corr_rr, color='darkgreen')
axes[1].axvline(0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_title('RR Interval Cross-Correlation: Sympathetic vs Parasympathetic')
axes[1].set_xlabel('Lag (samples)')
axes[1].set_ylabel('Correlation')

plt.tight_layout()
plt.savefig('cross_correlation_plots.png')
plt.show()
print("Cross-correlation plot saved")
