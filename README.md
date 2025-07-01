# mz_pna_cure_2025

## Objective
This project aims to develop a machine learning model to predict successful cure in critically ill patients with community-acquired pneumonia (CAP) enrolled in the SCRIPT study.  

[Halm's criteria](https://doi.org/10.1001/jama.279.18.1452) provide a set of clinical measures to help clinicians evaluate the stability of patients with CAP. Our goal is to use features derived from Halm’s criteria, based on patient data from Episode Day 1 to 3, to predict cure status on Day 7.

| Parameter             | Criterion                          |
|------------------------|------------------------------------|
| Temperature            | < 37.8 °C (100°F)                  |
| Heart Rate             | < 100 beats per minute             |
| Respiratory Rate       | <> 24 breaths per minute            |
| Systolic Blood Pressure| > 90 mm Hg                         |
| Oxygenation            | SpO₂ > 90% or PaO₂ > 60 mm Hg on room air |
| Oral Intake            | Able to maintain oral intake       |
| Mental Status          | Normal mental status               |

## Dataset
The analysis uses the [CarpeDiem](https://physionet.org/content/script-carpediem-dataset/1.1.0/), which is a cohort of critically ill patients enrolled in the SCRIPT study.

## Expected Output
Results, including the Table 1 summary and plots, will be saved in the `output` folder.

## Environment Setup

The environment setup code is provided in the `setup.sh` file for macOS.

**For macOS:**

1. Make the script executable: 
```bash
chmod +x setup.sh
```

2. Run the script:
```bash
./setup.sh
```
