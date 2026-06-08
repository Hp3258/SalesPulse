
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="SalesPulse",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = pickle.load(open('model.pkl', 'rb'))
    features = pickle.load(open('features.pkl', 'rb'))
    return model, features

model, features = load_model()

st.sidebar.title("SalesPulse")
st.sidebar.markdown("Employee Attrition Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔍 Predict", "📁 Bulk Upload",
     "📈 Explainability", "🎯 Recommendations",
     "🔄 What-If Simulator", "📊 Model Performance",
     "🏢 Department Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Built with XGBoost + SHAP")

# ============ HOME PAGE ============
if "Home" in page:
    st.title("SalesPulse — Employee Attrition Intelligence")
    st.markdown("Predict which employees are at risk of leaving using Machine Learning.")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Employees", "1,470")
    with col2:
        st.metric("Attrition Rate", "16.1%")
    with col3:
        st.metric("Model", "XGBoost")
    with col4:
        st.metric("F1 Score", "0.444")

    st.markdown("---")
    st.subheader("How it works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1 — Predict**\nEnter employee details and get instant attrition risk score")
    with col2:
        st.info("**Step 2 — Explain**\nSHAP values show exactly why the model made that prediction")
    with col3:
        st.info("**Step 3 — Act**\nUse insights to retain high risk employees before they leave")

    st.markdown("---")
    st.subheader("Key findings from data analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.warning("Employees who work **overtime** are significantly more likely to leave")
        st.warning("Employees with **no stock options** show highest attrition risk")
        st.warning("**Low job satisfaction** is a strong predictor of attrition")
    with col2:
        st.success("Employees with **high stock options** tend to stay longer")
        st.success("**Senior level** employees show lower attrition rates")
        st.success("Good **work life balance** strongly reduces attrition risk")

# ============ PREDICT PAGE ============
elif "Predict" in page:
    st.title("Employee Attrition Predictor")
    st.markdown("Fill in the employee details and click **Predict** to get risk assessment.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Employee Profile")
        age = st.slider("Age", 18, 60, 35)
        monthly_income = st.slider("Monthly Income", 1000, 20000, 6000)
        job_satisfaction = st.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3)
        work_life_balance = st.slider("Work Life Balance (1=Low, 4=High)", 1, 4, 3)
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        overtime = st.selectbox("OverTime", ["No", "Yes"])
        stock_option = st.slider("Stock Option Level (0=None, 3=High)", 0, 3, 1)
        job_level = st.slider("Job Level (1=Junior, 5=Senior)", 1, 5, 2)
        distance = st.slider("Distance From Home (km)", 1, 29, 5)
        business_travel = st.selectbox("Business Travel",
                                        ["Non-Travel",
                                         "Travel_Rarely",
                                         "Travel_Frequently"])
        predict_btn = st.button("🔍 Predict Attrition Risk",
                                use_container_width=True)

    with col2:
        st.subheader("Risk Assessment")

        if not predict_btn:
            st.info("👈 Fill in the employee profile and click **Predict** to see results.")

        else:
            df_clean = pd.read_csv('cleaned_data.csv')
            input_data = df_clean.drop('Attrition', axis=1).mean().to_dict()

            input_data['Age'] = age
            input_data['MonthlyIncome'] = monthly_income
            input_data['JobSatisfaction'] = job_satisfaction
            input_data['WorkLifeBalance'] = work_life_balance
            input_data['YearsAtCompany'] = years_at_company
            input_data['OverTime'] = 1 if overtime == "Yes" else 0
            input_data['StockOptionLevel'] = stock_option
            input_data['JobLevel'] = job_level
            input_data['DistanceFromHome'] = distance
            input_data['BusinessTravel'] = (2 if business_travel == "Travel_Frequently"
                                            else 1 if business_travel == "Travel_Rarely"
                                            else 0)

            input_df = pd.DataFrame([input_data])[features]
            probability = model.predict_proba(input_df)[0][1]
            prob_percent = round(float(probability) * 100, 1)

            if probability >= 0.7:
                st.error(f"🔴 HIGH RISK — {prob_percent}% probability of leaving")
            elif probability >= 0.4:
                st.warning(f"🟡 MEDIUM RISK — {prob_percent}% probability of leaving")
            else:
                st.success(f"🟢 LOW RISK — {prob_percent}% probability of leaving")

            st.progress(float(probability))
            st.markdown("---")

            st.subheader("Why this prediction?")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            feature_shap = pd.DataFrame({
                'Feature': features,
                'Impact': shap_values[0]
            }).sort_values('Impact', ascending=False)

            top_risk = feature_shap.head(3)
            top_protect = feature_shap.tail(3)

            st.markdown("**Top risk factors:**")
            for _, row in top_risk.iterrows():
                st.markdown(f"🔴 **{row['Feature']}** — pushing towards attrition (+{round(row['Impact'],3)})")

            st.markdown("**Protective factors:**")
            for _, row in top_protect.iterrows():
                st.markdown(f"🟢 **{row['Feature']}** — reducing attrition risk ({round(row['Impact'],3)})")

            # SHAP Waterfall Chart
            st.markdown("---")
            st.subheader("SHAP Waterfall Chart")
            st.markdown("Visual breakdown of every feature's contribution to this prediction:")

            shap_sorted = pd.DataFrame({
                'Feature': features,
                'Impact': shap_values[0]
            })
            shap_sorted['Abs'] = shap_sorted['Impact'].abs()
            shap_sorted = shap_sorted.sort_values('Abs', ascending=True).tail(10)

            colors = ['#D85A30' if x > 0 else '#1D9E75'
                      for x in shap_sorted['Impact']]

            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.barh(shap_sorted['Feature'],
                          shap_sorted['Impact'],
                          color=colors)
            ax.axvline(x=0, color='gray', linewidth=0.8)
            ax.set_xlabel('SHAP Value')
            ax.set_title('Feature Contributions — Red = increases risk, Green = reduces risk')

            for bar, val in zip(bars, shap_sorted['Impact']):
                ax.text(
                    bar.get_width() + (0.01 if val > 0 else -0.01),
                    bar.get_y() + bar.get_height()/2,
                    f'{round(val, 3)}',
                    va='center',
                    ha='left' if val > 0 else 'right',
                    fontsize=9
                )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("---")
            risk_label = ("high" if probability >= 0.7
                         else "medium" if probability >= 0.4
                         else "low")
            top_factor = feature_shap.iloc[0]['Feature']
            st.info(f"**Summary:** This employee has a **{risk_label} attrition risk** of {prob_percent}%. The strongest driver is **{top_factor}**.")

# ============ BULK UPLOAD PAGE ============
elif "Bulk" in page:
    st.title("Bulk Employee Risk Assessment")
    st.markdown("Upload a CSV file to get attrition risk scores for multiple employees.")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload employee CSV file", type=['csv'])

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.success(f"File uploaded — {len(df_upload)} employees found")

        try:
            input_bulk = df_upload[features]
            probabilities = model.predict_proba(input_bulk)[:, 1]

            df_upload['Attrition Probability %'] = [round(float(p)*100, 1)
                                                     for p in probabilities]
            df_upload['Risk Level'] = ['High' if p >= 0.7
                                       else 'Medium' if p >= 0.4
                                       else 'Low'
                                       for p in probabilities]

            col1, col2, col3 = st.columns(3)
            with col1:
                high = len(df_upload[df_upload['Risk Level'] == 'High'])
                st.metric("High Risk", high)
            with col2:
                med = len(df_upload[df_upload['Risk Level'] == 'Medium'])
                st.metric("Medium Risk", med)
            with col3:
                low = len(df_upload[df_upload['Risk Level'] == 'Low'])
                st.metric("Low Risk", low)

            st.markdown("---")
            st.subheader("Results")
            st.dataframe(
                df_upload[['Attrition Probability %', 'Risk Level']],
                use_container_width=True
            )

            csv = df_upload.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Results CSV",
                csv,
                "attrition_results.csv",
                "text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Error: {e}. Make sure your CSV has the correct columns.")
    else:
        st.info("Please upload a CSV file with employee data.")
        st.markdown("**Note:** CSV must have the same columns as the IBM HR dataset.")

# ============ EXPLAINABILITY PAGE ============
elif "Explain" in page:
    st.title("Model Explainability — SHAP Analysis")
    st.markdown("Understand what drives attrition predictions across all employees.")
    st.markdown("---")

    st.subheader("Global Feature Importance")

    if st.button("Generate SHAP Analysis", use_container_width=True):
        with st.spinner("Calculating SHAP values — this takes 10-15 seconds..."):
            df_clean = pd.read_csv('cleaned_data.csv')
            X_sample = (df_clean.drop('Attrition', axis=1)[features]
                       .sample(100, random_state=42))

            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_sample)

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_vals, X_sample,
                            feature_names=features,
                            plot_type="bar",
                            show=False)
            st.pyplot(fig)
            plt.close()
            st.success("SHAP analysis complete.")

    st.markdown("---")
    st.subheader("Saved SHAP charts")
    col1, col2 = st.columns(2)
    with col1:
        try:
            st.image('shap_global.png',
                    caption='Global Feature Importance',
                    use_container_width=True)
        except:
            st.info("Generate SHAP analysis above first")
    with col2:
        try:
            st.image('shap_dot.png',
                    caption='Feature Impact Direction',
                    use_container_width=True)
        except:
            st.info("Generate SHAP analysis above first")

# ============ RECOMMENDATIONS PAGE ============
elif "Recommendations" in page:
    st.title("🎯 Retention Recommendation Engine")
    st.markdown("Get specific HR action recommendations based on employee risk factors.")
    st.markdown("---")

    recommendations = {
        'StockOptionLevel': {
            'action': 'Offer Stock Option Plan',
            'detail': 'Employees with no stock options have significantly higher attrition. Offering even a basic stock option plan creates financial retention incentive.',
            'priority': 'High'
        },
        'OverTime': {
            'action': 'Reduce Overtime Hours',
            'detail': 'Overtime is one of the strongest attrition drivers. Consider flexible scheduling, workload redistribution, or hiring additional staff.',
            'priority': 'High'
        },
        'JobSatisfaction': {
            'action': 'Schedule Satisfaction Review',
            'detail': 'Low job satisfaction requires immediate manager intervention. Schedule a 1-on-1 review to identify specific pain points and improvement areas.',
            'priority': 'High'
        },
        'EnvironmentSatisfaction': {
            'action': 'Improve Work Environment',
            'detail': 'Poor environment satisfaction affects daily experience. Consider workspace improvements, team culture initiatives, or role reassignment.',
            'priority': 'Medium'
        },
        'WorkLifeBalance': {
            'action': 'Implement Work-Life Balance Policy',
            'detail': 'Poor work-life balance drives burnout. Consider remote work options, flexible hours, or mandatory time-off policies.',
            'priority': 'High'
        },
        'JobLevel': {
            'action': 'Create Clear Promotion Path',
            'detail': 'Junior employees leave more frequently. Define a clear career progression path with timeline and milestones.',
            'priority': 'Medium'
        },
        'MonthlyIncome': {
            'action': 'Review Compensation Package',
            'detail': 'Below-market salary is a key attrition driver. Conduct market salary benchmarking and adjust compensation accordingly.',
            'priority': 'High'
        },
        'YearsAtCompany': {
            'action': 'Implement Early Retention Program',
            'detail': 'Employees in first 2-3 years are highest risk. Create onboarding mentorship and early career development programs.',
            'priority': 'Medium'
        },
        'BusinessTravel': {
            'action': 'Reduce Travel Requirements',
            'detail': 'Frequent business travel increases attrition risk. Consider virtual meetings, travel rotation, or travel compensation policies.',
            'priority': 'Medium'
        },
        'JobInvolvement': {
            'action': 'Increase Role Engagement',
            'detail': 'Low job involvement signals disengagement. Assign meaningful projects, involve employee in decision making, recognize contributions.',
            'priority': 'Medium'
        },
        'MaritalStatus': {
            'action': 'Offer Family Support Benefits',
            'detail': 'Single employees show higher attrition. Consider social engagement programs, team building activities, and community building.',
            'priority': 'Low'
        },
        'DistanceFromHome': {
            'action': 'Offer Remote Work or Transport Support',
            'detail': 'Long commute increases attrition risk. Consider work from home options, flexible hours, or transport reimbursement.',
            'priority': 'Medium'
        },
        'RelationshipSatisfaction': {
            'action': 'Improve Team Relationships',
            'detail': 'Poor relationship satisfaction affects retention. Consider team building activities, conflict resolution support, or team reassignment.',
            'priority': 'Medium'
        },
        'TrainingTimesLastYear': {
            'action': 'Increase Learning Opportunities',
            'detail': 'Employees with less training feel stagnant. Create learning budgets, training programs, and skill development opportunities.',
            'priority': 'Medium'
        },
        'YearsSinceLastPromotion': {
            'action': 'Accelerate Promotion Review',
            'detail': 'Long time without promotion signals career stagnation. Conduct immediate promotion eligibility review.',
            'priority': 'High'
        }
    }

    st.subheader("Enter Employee Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 60, 30)
        monthly_income = st.slider("Monthly Income", 1000, 20000, 4000)
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 1)
        overtime = st.selectbox("OverTime", ["Yes", "No"])

    with col2:
        work_life_balance = st.slider("Work Life Balance", 1, 4, 2)
        years_at_company = st.slider("Years at Company", 0, 40, 2)
        stock_option = st.slider("Stock Option Level", 0, 3, 0)
        job_level = st.slider("Job Level", 1, 5, 1)

    with col3:
        distance = st.slider("Distance From Home", 1, 29, 15)
        business_travel = st.selectbox("Business Travel",
                                        ["Non-Travel", "Travel_Rarely",
                                         "Travel_Frequently"])
        years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, 5)

    analyze_btn = st.button("🎯 Generate Recommendations",
                            use_container_width=True)

    if analyze_btn:
        df_clean = pd.read_csv('cleaned_data.csv')
        input_data = df_clean.drop('Attrition', axis=1).mean().to_dict()

        input_data['Age'] = age
        input_data['MonthlyIncome'] = monthly_income
        input_data['JobSatisfaction'] = job_satisfaction
        input_data['WorkLifeBalance'] = work_life_balance
        input_data['YearsAtCompany'] = years_at_company
        input_data['OverTime'] = 1 if overtime == "Yes" else 0
        input_data['StockOptionLevel'] = stock_option
        input_data['JobLevel'] = job_level
        input_data['DistanceFromHome'] = distance
        input_data['YearsSinceLastPromotion'] = years_since_promotion
        input_data['BusinessTravel'] = (2 if business_travel == "Travel_Frequently"
                                        else 1 if business_travel == "Travel_Rarely"
                                        else 0)

        input_df = pd.DataFrame([input_data])[features]
        probability = model.predict_proba(input_df)[0][1]
        prob_percent = round(float(probability) * 100, 1)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            if probability >= 0.7:
                st.error(f"🔴 HIGH RISK\n\n**{prob_percent}%** probability")
            elif probability >= 0.4:
                st.warning(f"🟡 MEDIUM RISK\n\n**{prob_percent}%** probability")
            else:
                st.success(f"🟢 LOW RISK\n\n**{prob_percent}%** probability")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        feature_shap = pd.DataFrame({
            'Feature': features,
            'Impact': shap_values[0]
        }).sort_values('Impact', ascending=False)

        top_risk_features = feature_shap[feature_shap['Impact'] > 0].head(5)

        st.markdown("---")
        st.subheader("📋 Recommended HR Actions")
        st.markdown("Based on top risk factors for this employee:")

        action_count = 0
        for _, row in top_risk_features.iterrows():
            feature = row['Feature']
            if feature in recommendations:
                rec = recommendations[feature]
                action_count += 1
                priority_color = (
                    "🔴" if rec['priority'] == 'High'
                    else "🟡" if rec['priority'] == 'Medium'
                    else "🟢"
                )
                with st.expander(
                    f"{priority_color} Action {action_count}: {rec['action']} "
                    f"— driven by **{feature}**",
                    expanded=True
                ):
                    st.markdown(f"**Why:** SHAP impact score: +{round(row['Impact'], 3)}")
                    st.markdown(f"**Recommendation:** {rec['detail']}")
                    st.markdown(f"**Priority:** {rec['priority']}")

        if action_count == 0:
            st.success("No critical risk factors identified. Employee appears stable.")

        st.markdown("---")
        st.info(f"**Action Summary:** {action_count} HR interventions recommended. Addressing high priority actions first could significantly reduce the {prob_percent}% attrition risk.")

# ============ WHAT-IF SIMULATOR PAGE ============
elif "What-If" in page:
    st.title("🔄 What-If Retention Simulator")
    st.markdown("Compare current vs proposed employee profile to see impact of HR interventions.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Current Profile")
        st.markdown("*Employee as they are now*")
        age = st.slider("Age", 18, 60, 30, key="age_curr")
        income_curr = st.slider("Monthly Income", 1000, 20000, 4000, key="inc_curr")
        sat_curr = st.slider("Job Satisfaction", 1, 4, 1, key="sat_curr")
        wlb_curr = st.slider("Work Life Balance", 1, 4, 1, key="wlb_curr")
        yrs_curr = st.slider("Years at Company", 0, 40, 2, key="yrs_curr")
        ot_curr = st.selectbox("OverTime", ["Yes", "No"], key="ot_curr")
        stock_curr = st.slider("Stock Option Level", 0, 3, 0, key="stk_curr")
        jl_curr = st.slider("Job Level", 1, 5, 1, key="jl_curr")

    with col2:
        st.subheader("✅ Proposed Profile")
        st.markdown("*After HR intervention*")
        st.slider("Age", 18, 60, 30, key="age_prop", disabled=True)
        income_prop = st.slider("Monthly Income", 1000, 20000, 6000, key="inc_prop")
        sat_prop = st.slider("Job Satisfaction", 1, 4, 3, key="sat_prop")
        wlb_prop = st.slider("Work Life Balance", 1, 4, 3, key="wlb_prop")
        st.slider("Years at Company", 0, 40, 2, key="yrs_prop", disabled=True)
        ot_prop = st.selectbox("OverTime", ["No", "Yes"], key="ot_prop")
        stock_prop = st.slider("Stock Option Level", 0, 3, 2, key="stk_prop")
        jl_prop = st.slider("Job Level", 1, 5, 2, key="jl_prop")

    simulate_btn = st.button("🔄 Run Simulation", use_container_width=True)

    if simulate_btn:
        df_clean = pd.read_csv('cleaned_data.csv')
        base = df_clean.drop('Attrition', axis=1).mean().to_dict()

        curr = base.copy()
        curr['Age'] = age
        curr['MonthlyIncome'] = income_curr
        curr['JobSatisfaction'] = sat_curr
        curr['WorkLifeBalance'] = wlb_curr
        curr['YearsAtCompany'] = yrs_curr
        curr['OverTime'] = 1 if ot_curr == "Yes" else 0
        curr['StockOptionLevel'] = stock_curr
        curr['JobLevel'] = jl_curr

        prop = base.copy()
        prop['Age'] = age
        prop['MonthlyIncome'] = income_prop
        prop['JobSatisfaction'] = sat_prop
        prop['WorkLifeBalance'] = wlb_prop
        prop['YearsAtCompany'] = yrs_curr
        prop['OverTime'] = 1 if ot_prop == "Yes" else 0
        prop['StockOptionLevel'] = stock_prop
        prop['JobLevel'] = jl_prop

        curr_df = pd.DataFrame([curr])[features]
        prop_df = pd.DataFrame([prop])[features]

        prob_curr = round(float(model.predict_proba(curr_df)[0][1]) * 100, 1)
        prob_prop = round(float(model.predict_proba(prop_df)[0][1]) * 100, 1)
        delta = round(prob_curr - prob_prop, 1)

        st.markdown("---")
        st.subheader("📊 Simulation Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Risk", f"{prob_curr}%")
            if prob_curr >= 70:
                st.error("🔴 High Risk")
            elif prob_curr >= 40:
                st.warning("🟡 Medium Risk")
            else:
                st.success("🟢 Low Risk")

        with col2:
            st.metric(
                "Proposed Risk",
                f"{prob_prop}%",
                delta=f"-{delta}%" if delta > 0 else f"+{abs(delta)}%",
                delta_color="inverse"
            )
            if prob_prop >= 70:
                st.error("🔴 High Risk")
            elif prob_prop >= 40:
                st.warning("🟡 Medium Risk")
            else:
                st.success("🟢 Low Risk")

        with col3:
            st.metric("Risk Reduction", f"{delta}%")
            if delta > 30:
                st.success("✅ High impact intervention")
            elif delta > 10:
                st.warning("⚠️ Moderate impact")
            else:
                st.error("❌ Low impact — reconsider changes")

        st.markdown("---")
        st.subheader("📝 Intervention Summary")
        changes = []
        if income_prop != income_curr:
            changes.append(f"Monthly income: ₹{income_curr:,} → ₹{income_prop:,}")
        if sat_prop != sat_curr:
            changes.append(f"Job satisfaction: {sat_curr} → {sat_prop}")
        if wlb_prop != wlb_curr:
            changes.append(f"Work-life balance: {wlb_curr} → {wlb_prop}")
        if ot_prop != ot_curr:
            changes.append(f"Overtime: {ot_curr} → {ot_prop}")
        if stock_prop != stock_curr:
            changes.append(f"Stock options: Level {stock_curr} → Level {stock_prop}")
        if jl_prop != jl_curr:
            changes.append(f"Job level: {jl_curr} → {jl_prop}")

        if changes:
            st.markdown("**Proposed interventions:**")
            for change in changes:
                st.markdown(f"✅ {change}")

        st.info(f"**Conclusion:** Implementing these {len(changes)} HR intervention(s) reduces attrition risk from **{prob_curr}%** to **{prob_prop}%** — a reduction of **{delta} percentage points**.")

# ============ MODEL PERFORMANCE PAGE ============
elif "Model Performance" in page:
    st.title("📊 Model Performance Dashboard")
    st.markdown("Comprehensive evaluation of the XGBoost attrition prediction model.")
    st.markdown("---")

    if st.button("📊 Generate Performance Report", use_container_width=True):
        with st.spinner("Calculating model performance metrics..."):

            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (confusion_matrix, roc_curve,
                                        auc, precision_recall_curve,
                                        f1_score, precision_score,
                                        recall_score, accuracy_score)
            from imblearn.over_sampling import SMOTE
            import seaborn as sns

            df_clean = pd.read_csv('cleaned_data.csv')
            X = df_clean.drop('Attrition', axis=1)[features]
            y = df_clean['Attrition']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

            smote = SMOTE(random_state=42)
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_train_sm, y_train_sm)
            lr_pred = lr.predict(X_test)
            lr_prob = lr.predict_proba(X_test)[:, 1]

            st.success("Performance report generated successfully.")
            st.markdown("---")

            st.subheader("Key Metrics — XGBoost")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy",
                         f"{round(accuracy_score(y_test, y_pred)*100, 1)}%")
            with col2:
                st.metric("F1 Score",
                         round(f1_score(y_test, y_pred), 3))
            with col3:
                st.metric("Precision",
                         round(precision_score(y_test, y_pred), 3))
            with col4:
                st.metric("Recall",
                         round(recall_score(y_test, y_pred), 3))

            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['Predicted Stay', 'Predicted Leave'],
                           yticklabels=['Actual Stay', 'Actual Leave'],
                           ax=ax)
                ax.set_title('Confusion Matrix — XGBoost')
                st.pyplot(fig)
                plt.close()

                tn, fp, fn, tp = cm.ravel()
                st.markdown(f"""
                - ✅ **True Negatives:** {tn} — correctly predicted staying
                - ✅ **True Positives:** {tp} — correctly predicted leaving
                - ❌ **False Positives:** {fp} — predicted leaving but stayed
                - ❌ **False Negatives:** {fn} — predicted staying but left
                """)

            with col2:
                st.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
                roc_auc_lr = auc(fpr_lr, tpr_lr)

                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(fpr, tpr, color='#1D9E75', lw=2,
                       label=f'XGBoost (AUC = {round(roc_auc, 3)})')
                ax.plot(fpr_lr, tpr_lr, color='#7F77DD', lw=2,
                       label=f'Logistic Regression (AUC = {round(roc_auc_lr, 3)})')
                ax.plot([0, 1], [0, 1], color='gray',
                       linestyle='--', label='Random classifier')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve Comparison')
                ax.legend()
                st.pyplot(fig)
                plt.close()

                st.markdown(f"""
                - **XGBoost AUC:** {round(roc_auc, 3)}
                - **Logistic Regression AUC:** {round(roc_auc_lr, 3)}
                - AUC of 1.0 = perfect model
                - AUC of 0.5 = random guessing
                """)

            st.markdown("---")
            st.subheader("Precision-Recall Curve")
            col1, col2 = st.columns(2)

            with col1:
                precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(recall, precision, color='#D85A30', lw=2)
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Precision-Recall Curve — XGBoost')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

            with col2:
                st.markdown("### Understanding the tradeoff")
                st.markdown("""
                **Precision** = Out of all employees flagged as leaving,
                how many actually left?

                **Recall** = Out of all employees who actually left,
                how many did we catch?

                **In HR context:** Recall matters more. Missing an
                employee who is about to leave is more costly than
                a false alarm.
                """)

            st.markdown("---")
            st.subheader("Model Comparison")
            comparison_data = {
                'Model': ['Logistic Regression', 'XGBoost'],
                'Accuracy': [
                    f"{round(accuracy_score(y_test, lr_pred)*100, 1)}%",
                    f"{round(accuracy_score(y_test, y_pred)*100, 1)}%"
                ],
                'F1 Score': [
                    round(f1_score(y_test, lr_pred), 3),
                    round(f1_score(y_test, y_pred), 3)
                ],
                'Precision': [
                    round(precision_score(y_test, lr_pred), 3),
                    round(precision_score(y_test, y_pred), 3)
                ],
                'Recall': [
                    round(recall_score(y_test, lr_pred), 3),
                    round(recall_score(y_test, y_pred), 3)
                ],
                'AUC': [
                    round(roc_auc_lr, 3),
                    round(roc_auc, 3)
                ]
            }
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("Cross Validation — Model Stability")
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean F1 Score", round(cv_scores.mean(), 3))
            with col2:
                st.metric("Std Deviation", round(cv_scores.std(), 3))
            with col3:
                st.metric("Stability",
                         "High" if cv_scores.std() < 0.05 else "Medium")

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(range(1, 6), cv_scores, color='#1D9E75', alpha=0.8)
            ax.axhline(y=cv_scores.mean(), color='#D85A30',
                      linestyle='--', label=f'Mean: {round(cv_scores.mean(),3)}')
            ax.set_xlabel('Fold')
            ax.set_ylabel('F1 Score')
            ax.set_title('Cross Validation F1 Scores')
            ax.legend()
            st.pyplot(fig)
            plt.close()

    else:
        st.info("Click the button above to generate the complete model performance report.")

# ============ DEPARTMENT ANALYTICS PAGE ============
elif "Department" in page:
    st.title("🏢 Department Risk Analytics")
    st.markdown("Attrition risk breakdown across departments, job roles, and age groups.")
    st.markdown("---")

    df_clean = pd.read_csv('cleaned_data.csv')
    X_all = df_clean.drop('Attrition', axis=1)[features]
    all_probs = model.predict_proba(X_all)[:, 1]
    df_clean['Risk Score'] = all_probs
    df_clean['Risk Level'] = ['High' if p >= 0.7
                               else 'Medium' if p >= 0.4
                               else 'Low'
                               for p in all_probs]

    # Load original data for department names
    df_orig = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    df_orig['Risk Score'] = all_probs
    df_orig['Risk Level'] = df_clean['Risk Level']

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Employees", len(df_orig))
    with col2:
        high = len(df_orig[df_orig['Risk Level'] == 'High'])
        st.metric("High Risk", high)
    with col3:
        avg_risk = round(all_probs.mean() * 100, 1)
        st.metric("Avg Risk Score", f"{avg_risk}%")
    with col4:
        dept_count = df_orig['Department'].nunique()
        st.metric("Departments", dept_count)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk by Department")
        dept_risk = df_orig.groupby('Department')['Risk Score'].mean() * 100
        dept_risk = dept_risk.sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ['#D85A30' if v > 50 else '#BA7517' if v > 30 else '#1D9E75'
                  for v in dept_risk.values]
        bars = ax.barh(dept_risk.index, dept_risk.values, color=colors)
        ax.set_xlabel('Average Risk Score %')
        ax.set_title('Average Attrition Risk by Department')
        for bar, val in zip(bars, dept_risk.values):
            ax.text(bar.get_width() + 0.5,
                   bar.get_y() + bar.get_height()/2,
                   f'{round(val, 1)}%',
                   va='center', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Risk by Job Role")
        role_risk = df_orig.groupby('JobRole')['Risk Score'].mean() * 100
        role_risk = role_risk.sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ['#D85A30' if v > 50 else '#BA7517' if v > 30 else '#1D9E75'
                  for v in role_risk.values]
        ax.barh(role_risk.index, role_risk.values, color=colors)
        ax.set_xlabel('Average Risk Score %')
        ax.set_title('Average Attrition Risk by Job Role')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk by Age Group")
        df_orig['Age Group'] = pd.cut(df_orig['Age'],
                                       bins=[18, 25, 35, 45, 60],
                                       labels=['18-25', '26-35', '36-45', '46-60'])
        age_risk = df_orig.groupby('Age Group', observed=True)['Risk Score'].mean() * 100

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(age_risk.index, age_risk.values,
                     color=['#D85A30', '#BA7517', '#1D9E75', '#1D9E75'])
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Average Risk Score %')
        ax.set_title('Attrition Risk by Age Group')
        for bar, val in zip(bars, age_risk.values):
            ax.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.5,
                   f'{round(val, 1)}%',
                   ha='center', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Risk Distribution by Department")
        dept_counts = df_orig.groupby(['Department', 'Risk Level']).size().unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=(6, 4))
        dept_counts.plot(kind='bar',
                        color=['#D85A30', '#1D9E75', '#BA7517'],
                        ax=ax)
        ax.set_xlabel('Department')
        ax.set_ylabel('Number of Employees')
        ax.set_title('Risk Level Distribution by Department')
        ax.legend(title='Risk Level')
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Department Summary Table")
    summary = df_orig.groupby('Department').agg(
        Total_Employees=('Risk Score', 'count'),
        Avg_Risk_Score=('Risk Score', lambda x: f"{round(float(x.mean())*100, 1)}%"),,
        High_Risk_Count=('Risk Level', lambda x: (x == 'High').sum()),
        Actual_Attrition=('Attrition', lambda x: (x == 'Yes').sum())
    ).reset_index()
    st.dataframe(summary, use_container_width=True, hide_index=True)
