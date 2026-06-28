# Attrition-shield

🛡️ AI-powered Employee Attrition Prediction using Deep Learning (ANN)

Live demo: https://fizaaslam1-attrition-shield.hf.space/

Overview
--------
Attrition-shield is an end-to-end solution for predicting employee attrition using a feedforward Artificial Neural Network (ANN). The project includes reproducible Jupyter notebooks for data exploration and model training, a Streamlit-based interactive dashboard for inference and visualization, SMOTE-based handling of class imbalance, and an interface deployed on Hugging Face Spaces.

Key Features
------------
- End-to-end pipeline: data preprocessing → model training → evaluation → deployment
- Imbalanced-class handling with SMOTE
- ANN implemented with TensorFlow / Keras
- Automated hyperparameter tuning with Keras Tuner
- Interactive Streamlit dashboard with a gauge visualization for risk
- Risk scoring (probability → normalized 0–100 score)
- Reproducible notebooks for EDA, training, and inference

Live Demo
---------
Access the deployed app on Hugging Face Spaces:
https://fizaaslam1-attrition-shield.hf.space/

Tech Stack
----------
- Python 3.8+
- Jupyter Notebook
- TensorFlow / Keras
- Keras Tuner
- imbalanced-learn (SMOTE)
- Streamlit
- pandas, scikit-learn, matplotlib / seaborn, plotly

Quickstart (Local)
------------------
1. Clone the repository:
   ```bash
   git clone https://github.com/FizaAslam1/Attrition-shield.git
   cd Attrition-shield
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # macOS / Linux
   source .venv/bin/activate
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If a requirements file is not available:
   ```bash
   pip install streamlit tensorflow scikit-learn imbalanced-learn pandas matplotlib seaborn plotly keras-tuner
   ```

4. Run the Streamlit app:
   Replace `<APP_FILE>` with the repository's Streamlit entrypoint (e.g., `app`, `streamlit_app`, or `main`).
   ```bash
   streamlit run <APP_FILE>.py
   ```

Project Layout (example)
------------------------
- notebooks/                — Jupyter notebooks for EDA and model training
- app/ or src/              — Streamlit app and helper modules
- models/                   — Saved model artifacts / checkpoints
- data/                     — Sample datasets or preprocessing artifacts (if included)
- requirements.txt
- README.md

Usage
-----
- Interactive dashboard: launch the Streamlit app and input employee features to get an attrition probability, a 0–100 risk score, and a gauge visualization.
- Batch scoring: use the provided inference script or notebook to score multiple employees from a CSV.
- Retraining: re-run training notebooks to retrain or fine-tune models on updated data.

Model & Methodology
-------------------
- Data preprocessing: categorical encoding, missing-value handling, and feature scaling/normalization as appropriate.
- Class balancing: SMOTE to address minority-class imbalance prior to training.
- Architecture: feedforward ANN with dense layers, activations, dropout, and a sigmoid output for binary attrition probability.
- Hyperparameter tuning: Keras Tuner explores layer depth, units, learning rate, and dropout; best models are saved for inference.
- Risk scoring: model-predicted probability p is scaled into a user-friendly score:
  ```
  risk_score = round(p * 100)
  ```

Evaluation
----------
Recommended metrics and validation strategies:
- Accuracy, Precision, Recall, F1-score
- ROC AUC
- Confusion matrix
- Cross-validation and a separate hold-out test set for robust estimates

Deployment
----------
The app is already deployed on Hugging Face Spaces (link above). For alternative production deployment:
- Use Streamlit Cloud, Heroku, or Docker containers.
- Ensure `requirements.txt` is present and specify the Streamlit entrypoint in your host settings.
- Store any credentials or secrets in environment variables on the hosting platform.

Contributing
------------
Contributions are welcome. Suggested workflow:
1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit and push your changes to your fork
4. Open a pull request with a clear description of the change and testing steps

Please include test steps, expected behavior, and any performance or data considerations in your PR description.

License
-------
Add a LICENSE file to the repository (e.g., MIT License) and update this section accordingly.

Maintainer & Contact
--------------------
Maintainer: Fiza Aslam  
GitHub: https://github.com/FizaAslam1

Acknowledgements
----------------
- TensorFlow & Keras
- Streamlit
- scikit-learn and imbalanced-learn
