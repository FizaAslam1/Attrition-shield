import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# ⚙️ Page Config
st.set_page_config(page_title="Attrition Shield | HR Analytics", page_icon="🛡️", layout="wide")

# 🎨 Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); }
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }
    p, label, span { color: #e0e0e0 !important; }
    .stButton > button {
        background: linear-gradient(90deg, #6c5ce7, #a855f7);
        color: white !important; border: none; border-radius: 12px;
        padding: 1rem 2rem; font-size: 1.2rem; font-weight: 700;
        box-shadow: 0 4px 20px rgba(108, 92, 231, 0.5); width: 100%;
    }
    .risk-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.05));
        border: 2px solid #ef4444; border-radius: 20px; padding: 2rem;
    }
    .risk-low {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(34, 197, 94, 0.05));
        border: 2px solid #22c55e; border-radius: 20px; padding: 2rem;
    }
    .title-text {
        font-size: 3.5rem; font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 📋 Feature Order
feature_order = [
    'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
    'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
    'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
    'OverTime', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

# 🔧 Train Model On-the-Fly (Cached Resource)
@st.cache_resource
def get_model_and_scaler():
    # Multiple dataset sources
    urls = [
        "https://raw.githubusercontent.com/rshahid77/Employee-Attrition-Prediction/main/WA_Fn-UseC_-HR-Employee-Attrition.csv",
        "https://raw.githubusercontent.com/pavansubhasht/ibm-hr-analytics-attrition-dataset/main/WA_Fn-UseC_-HR-Employee-Attrition.csv",
        "https://raw.githubusercontent.com/tejshvij/IBM-HR-Analytics-Employee-Attrition-Performance/main/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    ]
    
    df = None
    for url in urls:
        try:
            df = pd.read_csv(url)
            break
        except:
            continue
    
    if df is None:
        raise Exception("Could not load dataset")
    
    # Rest same as before...
    df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)
    # ... continue with rest of function
    
    # Encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        if col != 'Attrition':
            df[col] = le.fit_transform(df[col])
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Split features and labels
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # Build the ANN Model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0, validation_split=0.1)
    
    return model, scaler

# Load/Cache model and scaler
try:
    model, scaler = get_model_and_scaler()
except Exception as e:
    st.error(f"Failed to load or train the model: {e}")
    st.stop()

# 🎯 Main UI Header
col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
with col2:
    st.markdown('<h1 class="title-text">🛡️ Attrition Shield</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#a0a0a0; font-size:1.2rem;">AI-Powered Employee Retention Intelligence</p>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

# 📋 Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.markdown("## 🎯 Risk Assessment")
    st.markdown("---")
    
    st.markdown("### 👤 Personal Info")
    age = st.slider("Age", 18, 65, 35)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=0)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    distance_from_home = st.slider("Distance from Home (km)", 1, 50, 10)
    
    st.markdown("---")
    st.markdown("### 💼 Job Details")
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)
    job_role = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    ], index=0)
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"], index=0)
    overtime = st.radio("Over Time", ["No", "Yes"], horizontal=True)
    
    st.markdown("---")
    st.markdown("### 😊 Satisfaction")
    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 2, help="1=Low, 4=Very High")
    work_life_balance = st.slider("Work-Life Balance", 1, 4, 3, help="1=Bad, 4=Best")
    environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
    relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
    job_involvement = st.slider("Job Involvement", 1, 4, 3)
    
    st.markdown("---")
    st.markdown("### 💰 Compensation")
    monthly_income = st.number_input("Monthly Income ($)", 1000, 60000, 5000, step=500)
    percent_salary_hike = st.slider("Last Salary Hike (%)", 1, 30, 15)
    stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3], index=0)
    
    st.markdown("---")
    st.markdown("### 📈 Career")
    years_at_company = st.slider("Years at Company", 0, 40, 5)
    years_in_role = st.slider("Years in Current Role", 0, 20, 3)
    years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, 2)
    years_with_manager = st.slider("Years with Current Manager", 0, 20, 3)
    total_working_years = st.slider("Total Working Years", 1, 50, 10)
    num_companies_worked = st.number_input("Number of Companies Worked", 0, 20, 2)

# 🎯 Predict
col1, col2, col3 = st.columns([0.05, 0.9, 0.05])
with col2:
    if st.button("🔍 Analyze Attrition Risk", use_container_width=True):
        with st.spinner("🧠 AI analyzing employee profile..."):
            
            input_dict = {
                'Age': age,
                'BusinessTravel': 1,  # Default: Travel_Rarely
                'DailyRate': 777.22,  # Average
                'Department': {'Sales': 0, 'Research & Development': 1, 'Human Resources': 2}.get(department, 0),
                'DistanceFromHome': distance_from_home,
                'Education': 3,  # Default: Bachelor's
                'EducationField': 0,  # Default: Life Sciences
                'EnvironmentSatisfaction': environment_satisfaction,
                'Gender': 1 if gender == 'Male' else 0,
                'HourlyRate': 65.00,
                'JobInvolvement': job_involvement,
                'JobLevel': job_level,
                'JobRole': {
                    'Sales Executive': 0, 'Research Scientist': 1, 'Laboratory Technician': 2,
                    'Manufacturing Director': 3, 'Healthcare Representative': 4, 'Manager': 5,
                    'Sales Representative': 6, 'Research Director': 7, 'Human Resources': 8
                }.get(job_role, 0),
                'JobSatisfaction': job_satisfaction,
                'MaritalStatus': {'Single': 0, 'Married': 1, 'Divorced': 2}.get(marital_status, 0),
                'MonthlyIncome': monthly_income,
                'MonthlyRate': 14645.31,
                'NumCompaniesWorked': num_companies_worked,
                'OverTime': 1 if overtime == 'Yes' else 0,
                'PercentSalaryHike': percent_salary_hike,
                'PerformanceRating': 3,  # Default: Meets Expectations
                'RelationshipSatisfaction': relationship_satisfaction,
                'StockOptionLevel': stock_option_level,
                'TotalWorkingYears': total_working_years,
                'TrainingTimesLastYear': 3,
                'WorkLifeBalance': work_life_balance,
                'YearsAtCompany': years_at_company,
                'YearsInCurrentRole': years_in_role,
                'YearsSinceLastPromotion': years_since_promotion,
                'YearsWithCurrManager': years_with_manager
            }
            
            input_array = np.array([[input_dict[feat] for feat in feature_order]])
            input_scaled = scaler.transform(input_array)
            
            probability = model.predict(input_scaled, verbose=0)[0][0]
            risk_percentage = probability * 100
            
            threshold = 0.38
            
            st.markdown('<br>', unsafe_allow_html=True)
            
            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_percentage,
                title={'text': "Attrition Risk Score", 'font': {'color': 'white', 'size': 20}},
                delta={'reference': threshold * 100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(46, 213, 115, 0.3)"},
                        {'range': [30, threshold * 100], 'color': "rgba(255, 165, 0, 0.3)"},
                        {'range': [threshold * 100, 100], 'color': "rgba(255, 71, 87, 0.4)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 3},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': 'white'}, height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<br>', unsafe_allow_html=True)
            
            if probability > threshold:
                st.markdown(f"""
                <div class="risk-high">
                    <h2 style="color:#ff4757; text-align:center;">⚠️ HIGH ATTRITION RISK</h2>
                    <h3 style="text-align:center; color:#ff6b6b;">{risk_percentage:.1f}% Probability of Resignation</h3>
                    <br>
                    <h4 style="color:#ffa502;">📋 Recommended Actions:</h4>
                    <ul style="color:#ddd; font-size:1.1rem;">
                        <li>Schedule immediate 1-on-1 meeting with employee</li>
                        <li>Review compensation package & consider salary adjustment</li>
                        <li>Reduce overtime burden if applicable</li>
                        <li>Create clear career progression roadmap</li>
                        <li>Assign mentorship or leadership opportunities</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h2 style="color:#2ed573; text-align:center;">✅ LOW ATTRITION RISK</h2>
                    <h3 style="text-align:center; color:#7bed9f;">{risk_percentage:.1f}% Probability of Resignation</h3>
                    <br>
                    <h4 style="color:#2ed573;">💡 Retention Tips:</h4>
                    <ul style="color:#ddd; font-size:1.1rem;">
                        <li>Continue engagement through regular feedback</li>
                        <li>Provide growth opportunities to maintain satisfaction</li>
                        <li>Recognize contributions publicly</li>
                        <li>Maintain current work-life balance support</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown('<br><br>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#666; padding:1rem; border-top:1px solid rgba(255,255,255,0.1);">
    <p>🛡️ Attrition Shield — Powered by Deep Learning | Built with Streamlit & TensorFlow</p>
    <p style="font-size:0.8rem;">© 2026 Fiza Aslam | Data Scientist</p>
</div>
""", unsafe_allow_html=True)
