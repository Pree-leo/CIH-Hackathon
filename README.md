# MIT-Internship
yoga dataset signal processing
# YogaSync — Real-Time Autonomic State Detection During Yoga

## Problem
537 million diabetic and yoga therapy patients receive zero objective 
physiological feedback during practice. Instructors guess. Devices react 
too late. We detect the body's autonomic state in real time — before 
damage is done.

## Solution
A wrist-worn wearable pipeline that classifies Sympathetic, Parasympathetic, 
and Relaxed autonomic nervous system states in real time using signal 
processing — no machine learning black box, no clinical equipment required.

## What We Built
- FFT-based frequency domain analysis on EDA and RR Interval signals
- Welch's PSD and Burg AR Model for spectral analysis
- Poincaré Plot with SD1/SD2 HRV analysis
- Cross-correlation between sympathetic and parasympathetic states
- Hedges' G effect size validation

## Results (Validated on 16 Subjects, 14,716 Data Points)
| Metric | Result |
|--------|--------|
| EDA significance (Welch's t-test) | p = 0.0122 |
| RR Interval significance | p < 0.000001 |
| Hedges' G (Sympathetic vs Parasympathetic) | g = 0.56 (Medium effect) |
| RR Cross-correlation | r = −0.23 (p < 0.001) |
| Relaxed SD1 (beat-to-beat variability) | 185.40 ms |

poincare_separated.png

## Tech Stack
- Python 3, NumPy, SciPy, Pandas, Matplotlib
- Google Colab
- Empatica EmbracePlus wearable sensor

## Team
NMAM Institute of Technology, Karkala
