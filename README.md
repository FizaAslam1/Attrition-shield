# Attrition-shield
🛡️ AI-powered Employee Attrition Prediction using Deep Learning (ANN)

Attrition-shield is a Streamlit-based application and research codebase that predicts employee attrition using a feedforward Artificial Neural Network (ANN). The project includes data preprocessing (SMOTE balancing), model training with TensorFlow / Keras, automated hyperparameter tuning with Keras Tuner, interactive visualization (gauge chart), and a risk-scoring mechanism to help HR teams identify employees at high risk of leaving.

Badges
- Build / CI: (add badge)
- Streamlit Deploy: (add badge)
- License: (add badge)

Table of Contents
- About
- Key Features
- Tech Stack
- Demo / Screenshots
- Quickstart
  - Prerequisites
  - Local setup
  - Run the app
- Project structure
- Usage
  - Exploring the dashboard
  - Training / Retraining the model
- Model & Methodology
  - Data preprocessing (SMOTE)
  - Model architecture (ANN)
  - Hyperparameter tuning (Keras Tuner)
  - Risk scoring & Gauge Chart
- Evaluation & Metrics
- Deployment
- Contributing
- License
- Contact

About
----
Attrition-shield provides an end-to-end pipeline for building, tuning, and deploying an employee attrition prediction model. It pairs reproducible Jupyter notebooks for data exploration and training with an interactive Streamlit dashboard for inference and visualization.

Key Features
----
- End-to-end pipeline: data preprocessing → model training → deployment
- Imbalanced-class handling using SMOTE
- ANN model implemented with TensorFlow / Keras
- Automated hyperparameter search using Keras Tuner
- Interactive Streamlit dashboard with a gauge chart to visualize risk
- Risk scoring (probability → normalized risk score)
- Notebooks for experimentation and reproducibility

Tech Stack
----
- Python (core)
- Jupyter Notebooks (analysis & training)
- TensorFlow / Keras (modeling)
- Keras Tuner (hyperparameter search)
- imbalanced-learn (SMOTE)
- Streamlit (dashboard / deployment)
- common libs: pandas, scikit-learn, matplotlib / seaborn, plotly

Demo / Screenshots
----
(Replace with actual screenshots / GIFs)
- Dashboard screenshot: docs/screenshot-dashboard.png
- Gauge chart: docs/screenshot-gauge.png

Quickstart
----
Prerequisites
- Python 3.8+ (recommended)
- Git
- Optional: Conda

1) Clone the repo
```bash
git clone https://github.com/FizaAslam1/Attrition-shield.git
cd Attrition-shield
```

2) Create & activate a virtual environment
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

3) Install dependencies
```bash
pip install -r requirements.txt
```
If there is no requirements.txt, install the main packages:
```bash
pip install streamlit tensorflow scikit-learn imbalanced-learn pandas matplotlib seaborn plotly keras-tuner
```

4) Run the Streamlit app
Replace `<APP_FILE>` below with the actual Streamlit entrypoint (e.g., `app.py`, `streamlit_app.py`, or `main.py`).
```bash
streamlit run <APP_FILE>.py
```

Project structure (example)
----
- notebooks/                # Jupyter notebooks for EDA and training
- app/ or src/              # Streamlit app and supporting modules
- models/                   # Trained model artifacts / checkpoints
- data/                     # Sample datasets / preprocessing artifacts (if included)
- requirements.txt
- README.md

Usage
----
- Dashboard: Launch the Streamlit app and enter employee attributes via the UI; the app returns an attrition probability, a normalized risk score (0–100), and a gauge visualization.
- Batch predictions: Use provided inference script or notebook to score multiple employees from a CSV.
- Training: Re-run the training notebooks to retrain models on updated data or tune hyperparameters.

Model & Methodology
----
Data preprocessing
- Categorical encoding, missing-value handling, scaling/normalization are applied as appropriate.
- SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the minority attrition class prior to model training.

Model architecture
- Feedforward Artificial Neural Network (ANN) implemented in TensorFlow/Keras.
- Typical pipeline: input layer → hidden dense layers with activation and dropout → output sigmoid for attrition probability.

Hyperparameter tuning
- Keras Tuner is used to search important hyperparameters (e.g., number of layers, units per layer, learning rate, dropout).
- The tuning workflow is available in the notebooks: run the tuning cell/block to start the search and save the best model.

Risk scoring & Gauge Chart
- The model outputs a probability p (0–1) of attrition. The risk score displayed in the UI is a scaled mapping of this probability to a 0–100 range for easier interpretation:
  risk_score = round(p * 100)
- The gauge chart visualizes the risk score and uses color bands (low / medium / high) to highlight risk levels.

Evaluation & Metrics
----
The project evaluates model performance using industry-standard classification metrics:
- Accuracy
- Precision, Recall
- F1-score
- ROC AUC
- Confusion matrix

Cross-validation and a separate hold-out test set are recommended to ensure robust performance estimates.

Deployment
----
- The app is designed for deployment on Streamlit Cloud (Streamlit sharing).
- Ensure `requirements.txt` is present and the Streamlit entrypoint file is specified in the Streamlit Cloud app settings.
- For production-level deployment, consider containerizing with Docker, and set environment variables / secrets in your hosting environment.

Contributing
----
Contributions are welcome. Suggested workflow:
1. Fork the repository
2. Create a feature branch: git checkout -b feat/your-feature
3. Commit changes and push to your fork
4. Open a PR describing the change

Please include:
- Clear description of the change
- Steps to reproduce or test
- Any performance or data considerations

License
----
Specify your license here (e.g., MIT License). Add LICENSE file to repository.

Contact
----
Maintainer: Fiza Aslam
GitHub: https://github.com/FizaAslam1

Acknowledgements
----
- TensorFlow & Keras teams
- Streamlit for the UI framework
- scikit-learn and imbalanced-learn for preprocessing utilities

Notes & Next steps
----
- Add the actual Streamlit entrypoint filename in the Quickstart section.
- Replace placeholders for screenshots and badges with real images and links.
- If you want, I can:
  - Commit this README to the repository,
  - Generate a ready-to-use requirements.txt from the codebase,
  - Inspect the repo and update the Quickstart with exact filenames and commands.
