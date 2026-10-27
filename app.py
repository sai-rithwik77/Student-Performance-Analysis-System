import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF

st.set_page_config(page_title="Student Performance Analysis System", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for UI Cards & Badges
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border-left: 5px solid #1E88E5;
    }
    .metric-card-high { border-left-color: #E53935; }
    .metric-card-mod { border-left-color: #FB8C00; }
    .metric-card-track { border-left-color: #43A047; }
    .badge-high {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 6px 14px;
        border-radius: 15px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-mod {
        background-color: #FFF3E0;
        color: #EF6C00;
        padding: 6px 14px;
        border-radius: 15px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-track {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 6px 14px;
        border-radius: 15px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_default_data(n=250):
    np.random.seed(42)
    student_ids = [f"STU-{1000+i}" for i in range(n)]
    attendance = np.random.uniform(45, 100, n).round(1)
    study_hours = np.random.uniform(2, 25, n).round(1)
    assignments = np.random.uniform(40, 100, n).round(1)
    previous_score = np.random.uniform(35, 100, n).round(1)
    
    final_score = (
        0.30 * attendance + 
        0.25 * (study_hours * 4) + 
        0.20 * assignments + 
        0.25 * previous_score + 
        np.random.normal(0, 3, n)
    ).round(1)
    final_score = np.clip(final_score, 0, 100)
    
    return pd.DataFrame({
        'Student_ID': student_ids,
        'Attendance': attendance,
        'Study_Hours': study_hours,
        'Assignments_Score': assignments,
        'Previous_Score': previous_score,
        'Final_Score': final_score
    })

@st.cache_resource
def train_models(df):
    X = df[['Attendance', 'Study_Hours', 'Assignments_Score', 'Previous_Score']]
    y = df['Final_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    rf_pred = rf.predict(X_test)
    lr_pred = lr.predict(X_test)
    
    metrics = {
        'RF': {
            'MAE': mean_absolute_error(y_test, rf_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'R2': r2_score(y_test, rf_pred)
        },
        'LR': {
            'MAE': mean_absolute_error(y_test, lr_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
            'R2': r2_score(y_test, lr_pred)
        }
    }
    return rf, lr, metrics, X_train, X_test, y_train, y_test, rf_pred, lr_pred

def classify_risk(predicted_score, attendance, study_hours):
    risk_score = max(0, min(100, 100 - predicted_score))
    if predicted_score < 50 or attendance < 60:
        tier = "High Risk"
    elif predicted_score < 70 or attendance < 75:
        tier = "Moderate Risk"
    else:
        tier = "On Track"
    return tier, round(risk_score, 1)

def get_recommendations(attendance, study_hours, assignments, previous_score):
    recs = []
    if attendance < 75:
        recs.append("⚠️ Low Attendance Alert (<75%): Mandatory academic counseling and daily attendance tracking required.")
    if study_hours < 10:
        recs.append("📚 Insufficient Weekly Study Time (<10 hrs): Recommend structured peer-study groups and time management workshops.")
    if assignments < 60:
        recs.append("📝 Low Assignment Completion (<60%): Implement weekly progress check-ins and assignment extension support.")
    if previous_score < 50:
        recs.append("🎯 Foundational Knowledge Gap (<50% prior score): Provide targeted remedial tutoring in core concepts.")
    if not recs:
        recs.append("🌟 Excellent Performance: Student is performing consistently. Maintain current habits and explore enrichment opportunities.")
    return recs

def generate_pdf_report(df_summary, metrics):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Student Performance Analytics - Executive Summary", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Overall System Metrics", ln=True)
    pdf.set_font("Arial", '', 10)
    total_students = len(df_summary)
    avg_score = df_summary['Predicted_Score'].mean() if 'Predicted_Score' in df_summary.columns else df_summary['Final_Score'].mean()
    pdf.cell(0, 8, f"Total Students Analyzed: {total_students}", ln=True)
    pdf.cell(0, 8, f"Average Performance Score: {avg_score:.2f}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. Model Performance Comparison", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 8, f"Random Forest -> MAE: {metrics['RF']['MAE']:.2f}, RMSE: {metrics['RF']['RMSE']:.2f}, R2: {metrics['RF']['R2']:.3f}", ln=True)
    pdf.cell(0, 8, f"Linear Regression -> MAE: {metrics['LR']['MAE']:.2f}, RMSE: {metrics['LR']['RMSE']:.2f}, R2: {metrics['LR']['R2']:.3f}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "3. Risk Tier Breakdown", ln=True)
    pdf.set_font("Arial", '', 10)
    if 'Risk_Tier' in df_summary.columns:
        counts = df_summary['Risk_Tier'].value_counts()
        for tier, cnt in counts.items():
            pdf.cell(0, 8, f"- {tier}: {cnt} students ({(cnt/total_students)*100:.1f}%)", ln=True)
            
    return pdf.output(dest='S').encode('latin-1', 'replace')

df_default = load_default_data()
rf_model, lr_model, model_metrics, X_train, X_test, y_train, y_test, rf_test_pred, lr_test_pred = train_models(df_default)

st.title("🎓 Student Performance Analysis & Early-Warning System")
st.markdown("An AI/ML-powered analytical platform to evaluate student performance, compare predictive models, and deliver explainable early-warning recommendations.")

st.sidebar.header("⚙️ Configuration & Navigation")
selected_model_name = st.sidebar.radio("Primary Model Selection:", ["Random Forest Regressor", "Linear Regression"])
primary_model = rf_model if selected_model_name == "Random Forest Regressor" else lr_model

page = st.sidebar.selectbox("Navigation:", [
    "📊 Analytics Dashboard",
    "🎯 Single Student Risk Analysis",
    "📁 CSV Upload & Batch Reports",
    "📈 Dedicated Model Evaluation"
])

if page == "📊 Analytics Dashboard":
    st.header("📊 Interactive Analytics Dashboard")
    df_dash = df_default.copy()
    df_dash['Predicted_Score'] = primary_model.predict(df_dash[['Attendance', 'Study_Hours', 'Assignments_Score', 'Previous_Score']]).round(1)
    
    risks = [classify_risk(row['Predicted_Score'], row['Attendance'], row['Study_Hours']) for _, row in df_dash.iterrows()]
    df_dash['Risk_Tier'] = [r[0] for r in risks]
    df_dash['Risk_Score'] = [r[1] for r in risks]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><h4>Total Students</h4><h2>{len(df_dash)}</h2></div>", unsafe_allow_html=True)
    with col2:
        high_cnt = (df_dash['Risk_Tier'] == 'High Risk').sum()
        st.markdown(f"<div class='metric-card metric-card-high'><h4>High Risk Students</h4><h2 style='color:#C62828;'>{high_cnt}</h2></div>", unsafe_allow_html=True)
    with col3:
        mod_cnt = (df_dash['Risk_Tier'] == 'Moderate Risk').sum()
        st.markdown(f"<div class='metric-card metric-card-mod'><h4>Moderate Risk</h4><h2 style='color:#EF6C00;'>{mod_cnt}</h2></div>", unsafe_allow_html=True)
    with col4:
        track_cnt = (df_dash['Risk_Tier'] == 'On Track').sum()
        st.markdown(f"<div class='metric-card metric-card-track'><h4>On Track</h4><h2 style='color:#2E7D32;'>{track_cnt}</h2></div>", unsafe_allow_html=True)
        
    st.markdown("---")
    
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        fig_dist = px.histogram(df_dash, x="Final_Score", nbins=20, title="Grade & Performance Distribution", color_discrete_sequence=['#1E88E5'])
        st.plotly_chart(fig_dist, use_container_width=True)
    with vcol2:
        fig_scatter = px.scatter(df_dash, x="Attendance", y="Final_Score", color="Risk_Tier",
                                 title="Attendance vs. Marks (Color-Coded by Risk)",
                                 color_discrete_map={"High Risk": "#E53935", "Moderate Risk": "#FB8C00", "On Track": "#43A047"})
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    vcol3, vcol4 = st.columns(2)
    with vcol3:
        fig_risk = px.pie(df_dash, names="Risk_Tier", title="Student Risk Tier Breakdown",
                          color="Risk_Tier",
                          color_discrete_map={"High Risk": "#E53935", "Moderate Risk": "#FB8C00", "On Track": "#43A047"})
        st.plotly_chart(fig_risk, use_container_width=True)
    with vcol4:
        corr = df_dash[['Attendance', 'Study_Hours', 'Assignments_Score', 'Previous_Score', 'Final_Score']].corr().round(2)
        fig_corr = px.imshow(corr, text_auto=True, title="Feature Correlation Heatmap", color_continuous_scale="Blues")
        st.plotly_chart(fig_corr, use_container_width=True)

elif page == "🎯 Single Student Risk Analysis":
    st.header("🎯 Single Student Early-Warning & Prediction")
    st.markdown("Adjust parameters below to analyze student risk, compare model predictions, and review support recommendations.")
    
    sc1, sc2 = st.columns(2)
    with sc1:
        att = st.slider("Attendance Rate (%)", 0.0, 100.0, 75.0, 0.5)
        study = st.slider("Weekly Study Hours", 0.0, 40.0, 12.0, 0.5)
    with sc2:
        assign = st.slider("Assignments Score (%)", 0.0, 100.0, 70.0, 0.5)
        prev = st.slider("Previous Term Score (%)", 0.0, 100.0, 65.0, 0.5)
        
    input_data = pd.DataFrame([[att, study, assign, prev]], columns=['Attendance', 'Study_Hours', 'Assignments_Score', 'Previous_Score'])
    
    rf_pred_val = rf_model.predict(input_data)[0]
    lr_pred_val = lr_model.predict(input_data)[0]
    active_pred = rf_pred_val if selected_model_name == "Random Forest Regressor" else lr_pred_val
    tier, r_score = classify_risk(active_pred, att, study)
    
    st.markdown("### 🤖 Model Comparison & Prediction Output")
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        st.metric("Random Forest Prediction", f"{rf_pred_val:.1f} / 100")
    with pcol2:
        st.metric("Linear Regression Prediction", f"{lr_pred_val:.1f} / 100")
    with pcol3:
        st.metric(f"Active Model ({selected_model_name.split()[0]})", f"{active_pred:.1f} / 100")
        
    st.markdown("### 🚦 Early-Warning Alert Level")
    if tier == "High Risk":
        st.markdown(f"<div class='badge-high'>🚨 {tier} (Risk Score: {r_score})</div>", unsafe_allow_html=True)
    elif tier == "Moderate Risk":
        st.markdown(f"<div class='badge-mod'>⚠️ {tier} (Risk Score: {r_score})</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='badge-track'>✅ {tier} (Risk Score: {r_score})</div>", unsafe_allow_html=True)
        
    st.markdown("### 💡 Explainable Root-Cause Support Recommendations")
    recommendations = get_recommendations(att, study, assign, prev)
    for rec in recommendations:
        st.info(rec)

elif page == "📁 CSV Upload & Batch Reports":
    st.header("📁 CSV Schema Validation & Batch Reports")
    st.markdown("Upload student data CSV files for automated schema validation, predictions, and executive report downloads.")
    
    sample_df = pd.DataFrame({
        'Student_ID': ['STU-2001', 'STU-2002', 'STU-2003'],
        'Attendance': [85.0, 55.0, 92.0],
        'Study_Hours': [14.0, 6.0, 18.0],
        'Assignments_Score': [78.0, 50.0, 88.0],
        'Previous_Score': [72.0, 45.0, 85.0]
    })
    st.download_button("📥 Download Sample CSV Template", data=sample_df.to_csv(index=False), file_name="sample_student_data.csv", mime="text/csv")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    target_df = None
    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            req_cols = ['Attendance', 'Study_Hours', 'Assignments_Score', 'Previous_Score']
            missing_cols = [c for c in req_cols if c not in user_df.columns]
            if missing_cols:
                st.error(f"❌ Schema Validation Error: CSV is missing required columns: {missing_cols}")
                st.warning("Please align your CSV headers with the required schema or download the sample CSV template.")
            else:
                st.success("✅ CSV Schema Validation Passed!")
                user_df[req_cols] = user_df[req_cols].fillna(user_df[req_cols].mean())
                target_df = user_df
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")
    else:
        st.info("Using default dataset for report preview.")
        target_df = df_default.copy()
        
    if target_df is not None:
        X_batch = target_df[['Attendance', 'Study_Hours', 'Assignments_Score', 'Previous_Score']]
        target_df['RF_Predicted_Score'] = rf_model.predict(X_batch).round(1)
        target_df['LR_Predicted_Score'] = lr_model.predict(X_batch).round(1)
        target_df['Predicted_Score'] = target_df['RF_Predicted_Score'] if selected_model_name == "Random Forest Regressor" else target_df['LR_Predicted_Score']
        
        batch_risks = [classify_risk(r['Predicted_Score'], r['Attendance'], r['Study_Hours']) for _, r in target_df.iterrows()]
        target_df['Risk_Tier'] = [r[0] for r in batch_risks]
        target_df['Risk_Score'] = [r[1] for r in batch_risks]
        
        st.subheader("📋 Processed Student Batch Results")
        st.dataframe(target_df, use_container_width=True)
        
        st.subheader("📥 Executive Report Exports")
        ex1, ex2 = st.columns(2)
        with ex1:
            st.download_button("📄 Export Processed CSV Data", data=target_df.to_csv(index=False), file_name="processed_student_performance.csv", mime="text/csv")
        with ex2:
            try:
                pdf_bytes = generate_pdf_report(target_df, model_metrics)
                st.download_button("📑 Export Executive Summary (PDF)", data=pdf_bytes, file_name="executive_summary_report.pdf", mime="application/pdf")
            except Exception:
                html_rep = target_df.describe().to_html()
                st.download_button("📑 Export Executive Summary (HTML)", data=html_rep, file_name="executive_summary_report.html", mime="text/html")

elif page == "📈 Dedicated Model Evaluation":
    st.header("📈 Dedicated Model Evaluation & Comparison")
    st.markdown("Side-by-side performance evaluation of Random Forest Regressor and Linear Regression models.")
    
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.subheader("🌲 Random Forest Regressor")
        st.metric("Mean Absolute Error (MAE)", f"{model_metrics['RF']['MAE']:.3f}")
        st.metric("Root Mean Squared Error (RMSE)", f"{model_metrics['RF']['RMSE']:.3f}")
        st.metric("R² Score", f"{model_metrics['RF']['R2']:.3f}")
    with mcol2:
        st.subheader("📈 Linear Regression")
        st.metric("Mean Absolute Error (MAE)", f"{model_metrics['LR']['MAE']:.3f}")
        st.metric("Root Mean Squared Error (RMSE)", f"{model_metrics['LR']['RMSE']:.3f}")
        st.metric("R² Score", f"{model_metrics['LR']['R2']:.3f}")
        
    st.markdown("---")
    
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        fig_act_pred = go.Figure()
        fig_act_pred.add_trace(go.Scatter(x=y_test, y=rf_test_pred, mode='markers', name='Random Forest'))
        fig_act_pred.add_trace(go.Scatter(x=y_test, y=lr_test_pred, mode='markers', name='Linear Regression'))
        fig_act_pred.add_trace(go.Scatter(x=[30, 100], y=[30, 100], mode='lines', name='Perfect Fit', line=dict(dash='dash', color='gray')))
        fig_act_pred.update_layout(title="Actual vs. Predicted Scores", xaxis_title="Actual Final Score", yaxis_title="Predicted Final Score")
        st.plotly_chart(fig_act_pred, use_container_width=True)
    with pcol2:
        rf_residuals = y_test - rf_test_pred
        lr_residuals = y_test - lr_test_pred
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=rf_test_pred, y=rf_residuals, mode='markers', name='RF Residuals'))
        fig_res.add_trace(go.Scatter(x=lr_test_pred, y=lr_residuals, mode='markers', name='LR Residuals'))
        fig_res.add_shape(type="line", x0=min(rf_test_pred), y0=0, x1=max(rf_test_pred), y1=0, line=dict(color="red", dash="dash"))
        fig_res.update_layout(title="Residual Analysis (Errors vs. Predicted)", xaxis_title="Predicted Score", yaxis_title="Residuals")
        st.plotly_chart(fig_res, use_container_width=True)
        
    st.subheader("🔑 Feature Importance & Model Coefficients")
    features = ['Attendance', 'Study_Hours', 'Assignments_Score', 'Previous_Score']
    imp_df = pd.DataFrame({
        'Feature': features * 2,
        'Importance': np.concatenate([rf_model.feature_importances_, np.abs(lr_model.coef_)]),
        'Model': ['Random Forest'] * len(features) + ['Linear Regression'] * len(features)
    })
    fig_imp = px.bar(imp_df, x='Feature', y='Importance', color='Model', barmode='group', title="Interactive Feature Importance Comparison")
    st.plotly_chart(fig_imp, use_container_width=True)
