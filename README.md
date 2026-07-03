# AutoInsulin

## Predictive Insulin Decision Support Using Real-Time Autonomic Stress Signals

AutoInsulin is an AI-powered decision support system that predicts changes in insulin demand by analyzing real-time autonomic nervous system activity from wearable physiological signals. Instead of reacting only after blood glucose changes, AutoInsulin aims to provide proactive insulin dose recommendations by identifying physiological stress before significant glucose fluctuations occur.

---

# Problem Statement

Current insulin pumps and Continuous Glucose Monitoring (CGM) systems primarily rely on blood glucose measurements to guide insulin therapy. However, insulin requirements are influenced by more than glucose levels alone.

Physiological stress, autonomic nervous system activity, and metabolic state can alter insulin sensitivity before noticeable glucose changes occur. Existing systems largely ignore these predictive physiological signals, resulting in reactive rather than proactive insulin management.

---

# Our Solution

AutoInsulin introduces a wearable-driven machine learning framework that continuously monitors:

- Electrodermal Activity (EDA)
- Heart Rate
- Skin Temperature

These physiological signals are processed in real time to classify the user's autonomic state into one of three categories:

- Sympathetic
- Relaxed
- Parasympathetic

The detected autonomic state is then used as an additional decision-support input to recommend personalized insulin dose adjustments before significant glucose excursions occur.

---

# System Workflow

```text
Wearable Device
       │
       ▼
Real-Time Signal Acquisition
(EDA • Heart Rate • Skin Temperature)
       │
       ▼
Signal Processing
Filtering • Normalization • Segmentation
       │
       ▼
Feature Extraction
       │
       ▼
Machine Learning Model
       │
       ▼
Autonomic State Classification
(Sympathetic | Relaxed | Parasympathetic)
       │
       ▼
Insulin Dose Recommendation
```

---

# Key Features

- Real-time wearable physiological monitoring
- Machine learning-based autonomic state classification
- Multimodal analysis using EDA, heart rate, and skin temperature
- Predictive insulin decision support
- User-friendly mobile application
- Personalized insulin recommendations based on physiological state

---

# Machine Learning Pipeline

### Input Signals

- Electrodermal Activity (EDA)
- Heart Rate
- Skin Temperature

### Signal Processing

- Noise Removal
- Filtering
- Normalization
- Segmentation
- Feature Extraction

### Model Output

- Sympathetic State
- Relaxed State
- Parasympathetic State

---

# Mobile Application

The companion mobile application provides:

- Live physiological monitoring
- Current autonomic state
- Stress detection alerts
- Insulin dose recommendations
- Historical physiological trends
- Personalized insights

---

# Expected Impact

AutoInsulin aims to:

- Enable proactive diabetes management
- Improve personalization of insulin recommendations
- Reduce the likelihood of glucose excursions
- Support informed insulin therapy decisions
- Enhance the quality of life for insulin-dependent individuals

---

# Technology Stack

- Python
- Machine Learning
- Signal Processing
- Wearable Biosensors
- Mobile Application
- GitHub

---

# Future Scope

- Integration with Continuous Glucose Monitoring (CGM) systems
- Smart insulin pump compatibility
- Personalized adaptive machine learning models
- Cloud-based remote monitoring
- Clinical validation through pilot studies

---

# Disclaimer

AutoInsulin is a research prototype developed for educational, research, and innovation purposes. It is not intended for clinical use or medical decision-making without appropriate clinical validation, regulatory approval, and healthcare supervision.

---

# Team

- Preethika
- Team Member 2
- Team Member 3
- Team Member 4

---

## License

This project is developed for research and hackathon purposes. All rights reserved by the project team.
