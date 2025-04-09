import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE, VarianceThreshold, mutual_info_classif, chi2
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
)
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Set Streamlit page config
st.set_page_config(page_title="Feature Explorer", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Feature Visualizer", "Feature Selection", "ML Classification", "DL Models", "Federated Learning"])

# Function to load data
def load_data():
    url = st.text_input("Enter CSV URL from GitHub or upload a file:",
                        value="https://raw.githubusercontent.com/sufian-utm/real-time_monitoring/main/data/cwru/12k_DE_td_features.csv")
    if url:
        try:
            df = pd.read_csv(url)
            return df
        except:
            st.error("Could not load from URL.")
    return None

uploaded_file = st.file_uploader("Or upload a local CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data()

if df is not None:
    df['fault_type'] = df['fault'].astype(str).str.extract(r'^(IR|OR|B|Normal)')
    df['fault_size'] = df['fault'].astype(str).str.extract(r'(\d{3})')
    df['fault_size'] = df['fault_size'].fillna('000')

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop(['fault_size'], errors='ignore')

    if page == "Feature Visualizer":
        st.title("Feature Visualization")
        st.subheader("Preview of Dataset")
        st.dataframe(df.head(), use_container_width=True)

        selected_feature = st.selectbox("Select a Feature to Visualize", options=numeric_cols)

        st.subheader(f"Histogram of {selected_feature}")
        fig1 = px.histogram(df, x=selected_feature, color="fault_type", marginal="box", nbins=40)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader(f"Boxplot of {selected_feature} by Fault Type")
        fig2 = px.box(df, x="fault_type", y=selected_feature, color="fault_type", points="all")
        st.plotly_chart(fig2, use_container_width=True)

        if st.checkbox("Show Violin Plot"):
            st.subheader(f"Violin Plot of {selected_feature} by Fault Type")
            fig4 = px.violin(df, x="fault_type", y=selected_feature, color="fault_type", box=True, points="all")
            st.plotly_chart(fig4, use_container_width=True)

        if st.checkbox("Show Scatter Plot by Fault Size"):
            st.subheader(f"Scatter Plot of {selected_feature} vs Fault Size")
            fig5 = px.scatter(df, x="fault_size", y=selected_feature, color="fault_type")
            st.plotly_chart(fig5, use_container_width=True)

        if st.checkbox("Show Correlation Heatmap"):
            st.subheader("Feature Correlation Heatmap")
            fig3, ax = plt.subplots(figsize=(12, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig3)

        if st.checkbox("Compare Feature Distributions"):
            st.subheader("Compare Feature Distributions Across Fault Types")
            fig6 = go.Figure()
            for f_type in df['fault_type'].unique():
                subset = df[df['fault_type'] == f_type]
                fig6.add_trace(go.Histogram(x=subset[selected_feature], name=str(f_type), opacity=0.5))
            fig6.update_layout(barmode='overlay', xaxis_title=selected_feature, yaxis_title="Count")
            st.plotly_chart(fig6, use_container_width=True)

        if st.checkbox("Show Time Series Plot"):
            st.subheader("Time Series Plot of Feature by Index")
            for f_type in df['fault_type'].unique():
                subset = df[df['fault_type'] == f_type].reset_index()
                fig_ts = px.line(subset, x=subset.index, y=selected_feature,
                                 title=f"{selected_feature} over Samples - {f_type}",
                                 labels={'index': 'Sample Index'})
                fig_ts.update_layout(showlegend=False)
                st.plotly_chart(fig_ts, use_container_width=True)

    elif page == "Feature Selection":
        st.title("Feature Selection Methods")

        # Prepare Data
        X = df[numeric_cols]
        y = LabelEncoder().fit_transform(df['fault_type'])

        # Feature Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        st.subheader("Choose Feature Selection Method")
        selected_method = st.selectbox("Select Feature Selection Method", options=[
            "SelectKBest (ANOVA F-statistic)", "Recursive Feature Elimination (RFE)",
            "VarianceThreshold", "Random Forest Feature Importance", "L1-based (Lasso)",
            "Mutual Information", "Chi-Square", "ANOVA F-statistic", "Univariate Feature Selection",
            "Correlation-based Feature Selection"
        ])

        # Method 1: SelectKBest (ANOVA F-statistic)
        if selected_method == "SelectKBest (ANOVA F-statistic)":
            st.write("Selecting features using SelectKBest with ANOVA F-statistic...")
            selector = SelectKBest(f_classif, k=10)
            X_selected = selector.fit_transform(X_scaled, y)
            selected_features = X.columns[selector.get_support()]
            st.write("Top 10 Selected Features:", selected_features)

        # Method 2: Recursive Feature Elimination (RFE)
        elif selected_method == "Recursive Feature Elimination (RFE)":
            st.write("Selecting features using Recursive Feature Elimination (RFE)...")
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rfe = RFE(rf, n_features_to_select=10)
            X_selected = rfe.fit_transform(X_scaled, y)
            selected_features = X.columns[rfe.support_]
            st.write("Top 10 Selected Features:", selected_features)

        # Method 3: VarianceThreshold
        elif selected_method == "VarianceThreshold":
            st.write("Selecting features using VarianceThreshold...")
            selector = VarianceThreshold(threshold=0.1)
            X_selected = selector.fit_transform(X_scaled)
            selected_features = X.columns[selector.get_support()]
            st.write("Top Features with Variance Threshold:", selected_features)

        # Method 4: Random Forest Feature Importance
        elif selected_method == "Random Forest Feature Importance":
            st.write("Selecting features using Random Forest Feature Importance...")
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_scaled, y)
            importance = rf.feature_importances_
            indices = np.argsort(importance)[::-1]
            selected_features = X.columns[indices][:10]
            st.write("Top 10 Important Features:", selected_features)

        # Method 5: L1-based (Lasso) Regularization
        elif selected_method == "L1-based (Lasso)":
            st.write("Selecting features using Lasso (L1-based Regularization)...")
            lasso = Lasso(alpha=0.01)
            lasso.fit(X_scaled, y)
            selected_features = X.columns[np.abs(lasso.coef_) > 0]
            st.write("Top Selected Features with Lasso Regularization:", selected_features)

        # Method 6: Mutual Information
        elif selected_method == "Mutual Information":
            st.write("Selecting features using Mutual Information...")
            mutual_info = mutual_info_classif(X_scaled, y)
            indices = np.argsort(mutual_info)[::-1]
            selected_features = X.columns[indices][:10]
            st.write("Top 10 Features by Mutual Information:", selected_features)

        # Method 7: Chi-Square
        elif selected_method == "Chi-Square":
            st.write("Selecting features using Chi-Square test...")
            chi2_selector = SelectKBest(chi2, k=10)
            X_selected = chi2_selector.fit_transform(X_scaled, y)
            selected_features = X.columns[chi2_selector.get_support()]
            st.write("Top 10 Selected Features by Chi-Square:", selected_features)

        # Method 8: ANOVA F-statistic
        elif selected_method == "ANOVA F-statistic":
            st.write("Selecting features using ANOVA F-statistic...")
            selector = SelectKBest(f_classif, k=10)
            X_selected = selector.fit_transform(X_scaled, y)
            selected_features = X.columns[selector.get_support()]
            st.write("Top 10 Features by ANOVA F-statistic:", selected_features)

        # Method 9: Univariate Feature Selection
        elif selected_method == "Univariate Feature Selection":
            st.write("Selecting features using Univariate Feature Selection...")
            selector = SelectPercentile(f_classif, percentile=10)
            X_selected = selector.fit_transform(X_scaled, y)
            selected_features = X.columns[selector.get_support()]
            st.write("Top 10 Features by Univariate Feature Selection:", selected_features)

        # Method 10: Correlation-based Feature Selection
        elif selected_method == "Correlation-based Feature Selection":
            st.write("Selecting features based on Correlation Threshold...")
            corr_matrix = pd.DataFrame(X_scaled).corr()
            threshold = st.slider("Set Correlation Threshold", 0.0, 1.0, 0.9)
            selected_features = [column for column in X.columns if corr_matrix[column].abs().max() < threshold]
            st.write(f"Features with correlation below {threshold}:", selected_features)

    elif page == "ML Classification":
        st.title("Machine Learning Classification Models")

        st.subheader("Data Preparation")
        target_col = "fault_type"
        df = df.dropna(subset=[target_col])
        X = df[numeric_cols]
        y = LabelEncoder().fit_transform(df[target_col])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(probability=True),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Extra Trees": ExtraTreesClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "Linear SVC": LinearSVC(),
            "Passive Aggressive": PassiveAggressiveClassifier(),
        }

        selected_model = st.selectbox("Choose ML Model", options=list(models.keys()))
        model = models[selected_model]

        if st.button("Train and Evaluate"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.text("Classification Report")
            st.text(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            st.subheader("Confusion Matrix")
            fig_cm, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig_cm)

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
                st.subheader("ROC Curve")
                fig_roc, ax = plt.subplots()
                for i in range(y_prob.shape[1]):
                    fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
                    ax.plot(fpr, tpr, label=f"Class {i} (AUC: {auc(fpr, tpr):.2f})")
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.legend()
                st.pyplot(fig_roc)

    elif page == "DL Models":
        st.title("Deep Learning Models")
        st.write("Coming soon: 20 Deep Learning models with training visualization")

    elif page == "Federated Learning":
        st.title("Federated Learning Setup")
        st.write("Coming soon: 20 FL simulations using selected models")
else:
    st.info("Please enter a valid GitHub URL or upload a file to begin.")
