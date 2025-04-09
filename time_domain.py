import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
)
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

st.set_page_config(page_title="Feature Explorer", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Feature Visualizer", "ML Classification", "DL Models", "Federated Learning"])

# Load data from GitHub or local file
# @st.cache_data
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
