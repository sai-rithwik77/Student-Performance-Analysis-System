import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
from fpdf import FPDF

st.set_page_config(page_title="Student Performance Analysis System", layout="wide")
st.title("Student Performance Analysis System")
st.markdown("An AI/ML-powered analytical system to evaluate, predict, and track student academic performance.")

@st.cache_data
def load_default_data(n=250):
    np.random.seed(42)
    student_ids = [f"STU-{1000+i}" for i in range(n)]
    attendance = np.random.uniform(45, 100, n).round(1)
    study_hours = np.random.uniform(2, 25, n).round(1)
    assignments = np.random.uniform(40, 100, n).round(1)
    prev_marks = np.random.uniform(35, 95, n).round(1)
    final_marks = (0.25 * attendance + 1.2 * study_hours + 0.25 * assignments + 0.30 * prev_marks + np.random.normal(0, 3.5, n)).clip(0, 100).round(1)
    return pd.DataFrame({
        "Student_ID": student_ids,
        "Attendance_Pct": attendance,
        "Study_Hours_Week": study_hours,
        "Assignment_Score": assignments,
        "Previous_Marks": prev_marks,
        "Final_Marks": final_marks
    })

st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Student CSV (Optional)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom CSV loaded successfully!")
else:
    df = load_default_data()
    st.sidebar.info("Using default synthetic student dataset (250 records).")

features = ["Attendance_Pct", "Study_Hours_Week", "Assignment_Score", "Previous_Marks"]
target = "Final_Marks"

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

df["Predicted_Marks"] = rf_model.predict(X).round(1)

def get_alert_status(row):
    if row["Predicted_Marks"] < 50 or row["Attendance_Pct"] < 65:
        return "High Risk (Immediate Support)"
    elif row["Predicted_Marks"] < 65 or row["Attendance_Pct"] < 75:
        return "Moderate Risk (Monitor)"
    return "On Track"

def get_recommendations(row):
    recs = []
    if row["Attendance_Pct"] < 75:
        recs.append("Boost class attendance above 75%")
    if row["Study_Hours_Week"] < 10:
        recs.append("Increase self-study to at least 10-12 hrs/week")
    if row["Assignment_Score"] < 70:
        recs.append("Focus on assignment completion and rubric review")
    if not recs:
        recs.append("Maintain current study schedule and peer mentoring")
    return "; ".join(recs)

df["Alert_Status"] = df.apply(get_alert_status, axis=1)
df["Recommendations"] = df.apply(get_recommendations, axis=1)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard & Stats", "AI Marks Predictor", "Early-Warning & Recommendations", "Research & Methodology", "Export Reports"])

with tab1:
    st.subheader("System Overview & Model Evaluation")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", len(df))
    col2.metric("Average Marks", f"{df['Final_Marks'].mean():.1f}%")
    col3.metric("High Risk Students", len(df[df["Alert_Status"].str.contains("High Risk")]))
    col4.metric("Model R2 Score", f"{r2:.3f}")
    m1, m2, m3 = st.columns(3)
    m1.info(f"MAE: {mae:.2f}")
    m2.info(f"RMSE: {rmse:.2f}")
    m3.info(f"R2: {r2:.2f}")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(df, x="Final_Marks", nbins=20, title="Distribution of Final Marks"), use_container_width=True)
    with c2:
        st.plotly_chart(px.scatter(df, x="Attendance_Pct", y="Final_Marks", color="Alert_Status", title="Attendance vs. Final Marks"), use_container_width=True)

with tab2:
    st.subheader("Predict Individual Student Performance")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        in_att = st.slider("Attendance (%)", 0.0, 100.0, 72.0)
        in_study = st.slider("Weekly Study Hours", 0.0, 40.0, 9.0)
        in_assign = st.slider("Assignment Score (%)", 0.0, 100.0, 68.0)
        in_prev = st.slider("Previous Exam Marks (%)", 0.0, 100.0, 65.0)
    with p_col2:
        input_df = pd.DataFrame([[in_att, in_study, in_assign, in_prev]], columns=features)
        pred_score = rf_model.predict(input_df)[0]
        st.metric("Predicted Final Score", f"{pred_score:.1f} / 100")
        sample_row = {"Predicted_Marks": pred_score, "Attendance_Pct": in_att, "Study_Hours_Week": in_study, "Assignment_Score": in_assign}
        st.write("**Early-Warning Status:**", get_alert_status(sample_row))
        st.write("**Personalized Action Plan:**", get_recommendations(sample_row))

with tab3:
    st.subheader("Early-Warning Alert System & Actionable Recommendations")
    risk_filter = st.selectbox("Filter by Risk Level", ["All", "High Risk (Immediate Support)", "Moderate Risk (Monitor)", "On Track"])
    filtered_df = df if risk_filter == "All" else df[df["Alert_Status"] == risk_filter]
    st.dataframe(filtered_df[["Student_ID", "Attendance_Pct", "Study_Hours_Week", "Assignment_Score", "Predicted_Marks", "Alert_Status", "Recommendations"]], use_container_width=True)

with tab4:
    st.subheader("Research Base Paper & System Methodology")
    st.markdown("### 1. Abstract & Base Paper Context\nAcademic performance modeling leverages Educational Data Mining (EDM) to identify at-risk students before final assessments using Random Forest ensemble learning.\n\n### 2. Proposed Methodology\n1. Data Collection & Preprocessing\n2. Feature Engineering\n3. Ensemble Regression (Random Forest)\n4. Evaluation Metrics (MAE, RMSE, R2)\n5. Automated Risk Stratification & Prescriptive Analytics")

with tab5:
    st.subheader("Download Analyzed Reports")
    st.download_button("Download Analyzed Dataset (CSV)", data=df.to_csv(index=False).encode("utf-8"), file_name="analyzed_student_performance.csv", mime="text/csv")
    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Student Performance Analysis Report", ln=True, align="C")
        pdf.ln(5)
        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 8, f"Total Students: {len(df)} | Avg Marks: {df['Final_Marks'].mean():.1f}%", ln=True)
        pdf.cell(0, 8, f"Metrics -> MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.2f}", ln=True)
        return pdf.output(dest="S").encode("latin-1")
    st.download_button("Download Summary Report (PDF)", data=generate_pdf(), file_name="Student_Performance_Report.pdf", mime="application/pdf")
