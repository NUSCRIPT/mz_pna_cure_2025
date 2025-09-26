# mz_pna_cure_2025

Code for the study:  
**"Development of a Pilot Machine Learning Model to Predict Successful Cure in Critically Ill Patients with Community-Acquired Pneumonia"**  
📄 [Preprint DOI: 10.1101/2025.07.14.25331407](https://doi.org/10.1101/2025.07.14.25331407)

---

## 👥 Contributors

Mengou Zhu, Wan-Ting Liao, Alec Peltekian, Nikolay S. Markov, Mengjia Kang, Luke V. Rasmussen, Thomas Stoeger, Theresa L. Walunas, Alexander V. Misharin, Benjamin D. Singer, G. R. Scott Budinger, Richard G. Wunderink, Ankit Agrawal, and Catherine A. Gao.

---

## 🎯 Objective

To develop a machine learning model that predicts successful cure in critically ill patients with community-acquired pneumonia (CAP), using data from the SCRIPT study.

---

## ⚙️ Environment Setup

Environment dependencies are listed in `requirements.txt`.  
A setup script is provided for macOS users in `setup.sh`.

### macOS Setup

```bash
chmod +x setup.sh
./setup.sh
```

07.cap_pred_deidentified_version.ipynb - main analysis file 

## 📊 Data Availability

Data are partially available via PhysioNet:  
🔗 [SCRIPT CarpeDiem Dataset v1.1.0](https://physionet.org/content/script-carpediem-dataset/1.1.0/)

When using this dataset, please cite:  
Markov N., Gao C. A., Stoeger T., Pawlowski A., Kang M., Nannapaneni P., Grant R., Rasmussen L., Schneider D., Starren J., Wunderink R., Budinger G. S., Misharin A., Singer B., and the Northwestern SCRIPT Study Investigators (2023). *SCRIPT CarpeDiem Dataset: demographics, outcomes, and per-day clinical parameters for critically ill patients with suspected pneumonia* (version 1.1.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/5phr-4r89

An upcoming release will include patients enrolled since the publication of the original dataset.


---

## 🛠️ Questions or Issues?

If you have questions, suggestions, or encounter any problems, please feel free to open an issue. 

This repository has been reviewed and processed through de-identification checks. However, if you notice any information that appears concerning or potentially identifiable, please contact us immediately or open an issue to let us know.
