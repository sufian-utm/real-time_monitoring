import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectKBest, f_classif, RFE, VarianceThreshold, mutual_info_classif, chi2, SelectFromModel, SelectPercentile
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
)
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.inspection import permutation_importance
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Set Streamlit page config
st.set_page_config(page_title="Feature Explorer", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Feature Visualizer", "Feature Selection", "ML Classification", "DL Models"])

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

def plot_feature_selection_scores(scores, feature_names, title="Feature Selection Scores", num=10):
    sorted_idx = np.argsort(scores)[::-1]
    sorted_scores = np.array(scores)[sorted_idx]
    sorted_features = np.array(feature_names)[sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features[:num][::-1], sorted_scores[:num][::-1], color='skyblue')
    plt.xlabel("Score")
    plt.title(title)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()
    
uploaded_file = st.file_uploader("Or upload a local CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data()

if df is not None:
    df['fault_type'] = df['fault'].astype(str).str.extract(r'^(IR|OR|B|Normal)')
    df['fault_size'] = df['fault'].astype(str).str.extract(r'(\d{3})')
    df['fault_size'] = df['fault_size'].fillna('000')

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop(['fault_size', 'Unnamed: 0'], errors='ignore')

    if page == "ML Classification" or page == "DL Models":
        target_col_type = "fault_type"
        target_col_size = "fault_size"

        df = df.dropna(subset=[target_col_type, target_col_size])
        X = df[numeric_cols]
        y_type = LabelEncoder().fit_transform(df[target_col_type])
        y_size = LabelEncoder().fit_transform(df[target_col_size])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train_type, y_test_type, y_train_size, y_test_size = train_test_split(
            X_scaled, y_type, y_size, test_size=0.2, random_state=42
        )

        # Add feature selection dropdown
        st.sidebar.header("🔍 Feature Selection")
        feature_selection_method = st.sidebar.selectbox("Select Feature Selection Method", [
            "Recursive Feature Elimination (RFE)", "Pearson Correlation",
            "VarianceThreshold", "Random Forest Feature Importance", "L1-based (Lasso)","Mutual Information", 
            "Chi-Square", "ANOVA F-statistic",  "K-Nearest Neighbors (KNN)", "GaussianNB"
        ])
        num_features = st.sidebar.slider("Number of Features", 5, 25, 10)
        
        # Feature Selection Method
        if feature_selection_method == "Recursive Feature Elimination (RFE)":
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(rf, n_features_to_select=num_features).fit(X, y_type)
            X_selected = selector.transform(X)
        elif feature_selection_method == "VarianceThreshold":
            selector = VarianceThreshold(threshold=0.1).fit(X)
            X_selected = selector.transform(X)
        elif feature_selection_method == "Random Forest Feature Importance":
            rf = RandomForestClassifier()
            rf.fit(X, y_type)
            importances = rf.feature_importances_
            top_indices = np.argsort(importances)[::-1][:num_features]
            X_selected = X.iloc[:, top_indices]
            selected_features = X.columns[top_indices].tolist()
        elif feature_selection_method == "L1-based (Lasso)":
            lasso = Lasso(alpha=0.01)
            lasso.fit(X, y_type)
            mask = np.abs(lasso.coef_) > 0
            selected_features = X.columns[mask].tolist()
            X_selected = X[selected_features]
        elif feature_selection_method == "Mutual Information":
            selector = SelectKBest(score_func=mutual_info_classif, k=num_features).fit(X, y_type)
            X_selected = selector.transform(X)
        elif feature_selection_method == "Chi-Square":
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            selector = SelectKBest(score_func=chi2, k=num_features)
            X_selected = selector.fit_transform(X_scaled, y_type)
        elif feature_selection_method == "ANOVA F-statistic":
            selector = SelectKBest(score_func=f_classif, k=num_features).fit(X, y_type)
            X_selected = selector.transform(X)
        elif feature_selection_method == "K-Nearest Neighbors (KNN)":
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X, y_type)
            from sklearn.inspection import permutation_importance
            result = permutation_importance(knn, X, y_type, n_repeats=10, random_state=42)
            importances = result.importances_mean
            top_indices = np.argsort(importances)[::-1][:num_features]
            X_selected = X.iloc[:, top_indices]
            selected_features = X.columns[top_indices].tolist()
        elif feature_selection_method == "GaussianNB":
            gnb = GaussianNB()
            gnb.fit(X, y_type)
            # No built-in feature ranking; use variance as a proxy
            variances = X.var()
            top_indices = np.argsort(variances)[::-1][:num_features]
            X_selected = X.iloc[:, top_indices]
            selected_features = X.columns[top_indices].tolist()
        elif feature_selection_method == "Pearson Correlation":
            # Convert y back to categorical labels if needed
            # if len(np.unique(y_type)) > 2:
            #    st.warning("Pearson correlation is best for binary targets. Proceed with caution.")
        
            correlations = []
            for i, feature in enumerate(X.columns):
                corr = np.corrcoef(X_scaled[:, i], y_type)[0, 1]
                correlations.append(abs(corr))

            top_indices = np.argsort(correlations)[::-1][:num_features]
            X_selected = X.iloc[:, top_indices]
            selected_features = X.columns[top_indices].tolist()
        
        # Determine selected features if not already assigned
        if 'selected_features' not in locals():
            if isinstance(selector, TransformerMixin) and hasattr(selector, 'get_support'):
                selected_features = X.columns[selector.get_support()].tolist()
            else:
                selected_features = X.columns[:X_selected.shape[1]].tolist()
    
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
            "Recursive Feature Elimination (RFE)", "Pearson Correlation",
            "VarianceThreshold", "Random Forest Feature Importance", "L1-based (Lasso)",
            "Mutual Information", "Chi-Square", "ANOVA F-statistic",  
            "K-Nearest Neighbors (KNN)", "GaussianNB"
        ])
        
        # Add numbr of features selection bar
        num_features = st.sidebar.slider("Number of Features", 5, 25, 10)
        
        # Method 1: Pearson Correlation
        if selected_method == "Pearson Correlation":
            st.write("Selecting features based on Pearson correlation with the target...")
        
            # Convert y back to categorical labels if needed
            # if len(np.unique(y)) > 2:
                # st.warning("Pearson correlation is best for binary targets. Proceed with caution.")
        
            correlations = []
            for i, feature in enumerate(X.columns):
                corr = np.corrcoef(X_scaled[:, i], y)[0, 1]
                correlations.append(abs(corr))
        
            plot_feature_selection_scores(correlations, X.columns, title="Pearson Correlation with Target", num=num_features)
            top_indices = np.argsort(correlations)[::-1][:num_features]
            selected_features = X.columns[top_indices]
            st.write(f"Top {num_features} Features by Pearson Correlation:", selected_features)

        # Method 2: Recursive Feature Elimination (RFE)
        elif selected_method == "Recursive Feature Elimination (RFE)":
            st.write("Selecting features using Recursive Feature Elimination (RFE)...")

            estimator = LogisticRegression(solver="liblinear")
            num_features_to_select = min(num_features, X.shape[1])  # Select top 10 or fewer if less features
            selector = RFE(estimator, n_features_to_select=num_features_to_select)
            selector.fit(X, y)
            selected_features = X.columns[selector.support_]
            
            # Plot rankings
            rankings = selector.ranking_
            ranking_df = pd.DataFrame({
                "Feature": X.columns,
                "Ranking": rankings,
                "Selected": selector.support_
            }).sort_values("Ranking")
        
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(ranking_df["Feature"], ranking_df["Ranking"],
                           color=["green" if sel else "red" for sel in ranking_df["Selected"]])
            ax.set_xlabel("RFE Ranking (1 = Best)")
            ax.set_title("RFE Feature Ranking")
            st.pyplot(fig)
            
            st.write(f"Top {num_features} Selected Features:", selected_features)

        # Method 3: VarianceThreshold
        elif selected_method == "VarianceThreshold":
            st.write("Selecting features using VarianceThreshold...")

            threshold = st.slider("Select Variance Threshold", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
            selector = VarianceThreshold(threshold=threshold)
        
            X_selected = selector.fit_transform(X)
            selected_mask = selector.get_support()
            selected_features = X.columns[selected_mask]
            
            st.write(f"Top {num_features} Features with Variance Threshold:", selected_features)

            # Plotting variances
            feature_variances = X.var()
            fig, ax = plt.subplots(figsize=(10, 5))
            feature_variances.plot(kind="bar", ax=ax, color=["green" if val else "red" for val in selected_mask])
            ax.axhline(y=threshold, color="r", linestyle="--", label=f"Threshold = {threshold}")
            ax.set_title("Feature Variances")
            ax.set_ylabel("Variance")
            ax.legend()
            st.pyplot(fig)
        
        # Method 4: Random Forest Feature Importance
        elif selected_method == "Random Forest Feature Importance":
            st.write("Selecting features using Random Forest Feature Importance...")
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_scaled, y)
            importance = rf.feature_importances_
            plot_feature_selection_scores(importance, X.columns, title="Random Forest Feature Importance", num=num_features)
            indices = np.argsort(importance)[::-1]
            selected_features = X.columns[indices][:num_features]
            st.write(f"Top {num_features} Important Features:", selected_features)
            
        # Method 5: L1-based (Lasso) Regularization
        elif selected_method == "L1-based (Lasso)":
            st.write("Selecting features using Lasso (L1-based Regularization)...")
            lasso = Lasso(alpha=0.01)
            lasso.fit(X_scaled, y)
            coef = np.abs(lasso.coef_)
            plot_feature_selection_scores(coef, X.columns, title="Lasso Feature Coefficients", num=num_features)
            selected_features = X.columns[coef > 0][:num_features]
            st.write(f"Top {num_features} Selected Features with Lasso Regularization:", selected_features)

        # Method 6: Mutual Information
        elif selected_method == "Mutual Information":
            st.write("Selecting features using Mutual Information...")
            mutual_info = mutual_info_classif(X_scaled, y)
            plot_feature_selection_scores(mutual_info, X.columns, title="Mutual Information Scores", num=num_features)
            indices = np.argsort(mutual_info)[::-1]
            selected_features = X.columns[indices][:num_features]
            st.write(f"Top {num_features} Features by Mutual Information:", selected_features)

        # Method 7: Chi-Square
        elif selected_method == "Chi-Square":
            st.write("Selecting features using Chi-Square test...")
    
            # Ensure X is non-negative for Chi-Square
            X_chi2 = np.maximum(X_scaled, 0)  # Set any negative values to zero
            
            # Apply SelectKBest with chi2
            selector = SelectKBest(chi2, k=10)
            selector.fit(X_chi2, y)
            
            # Get the scores for each feature
            scores = selector.scores_
            
            # Plot the feature selection scores
            plot_feature_selection_scores(scores, X.columns, title="Chi-Square Scores", num=num_features)
            
            # Get the top selected features
            top_indices = np.argsort(scores)[::-1][:num_features]
            selected_features = X.columns[top_indices]
            
            st.write(f"Top {num_features} Selected Features by Chi-Square:", selected_features)

        # Method 8: ANOVA F-statistic
        elif selected_method == "ANOVA F-statistic":
            st.write("Selecting features using ANOVA F-statistic...")
            selector = SelectKBest(f_classif, k='all')
            selector.fit(X_scaled, y)
            scores = selector.scores_
            plot_feature_selection_scores(scores, X.columns, title="ANOVA F-statistic Scores", num=num_features)
            top_indices = np.argsort(scores)[::-1][:num_features]
            selected_features = X.columns[top_indices]
            st.write(f"Top {num_features} Features by ANOVA F-statistic:", selected_features)
            
        # Method 11: K-Nearest Neighbors (KNN)
        elif selected_method == "K-Nearest Neighbors (KNN)":
            st.write("Selecting features using K-Nearest Neighbors (KNN)...")
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_scaled, y)
            result = permutation_importance(knn, X_scaled, y, n_repeats=10, random_state=42)
            importances = result.importances_mean
            plot_feature_selection_scores(importances, X.columns, title="Permutation Importance (KNN)", num=num_features)
            top_features = np.argsort(importances)[::-1][:num_features]
            selected_features = X.columns[top_features]
            st.write(f"Top {num_features} Features based on KNN:", selected_features)
    
        # Method 12: GaussianNB
        elif selected_method == "GaussianNB":
            st.write("Selecting features using Gaussian Naive Bayes...")

            # Initialize Gaussian Naive Bayes and fit the model
            gnb = GaussianNB()
            gnb.fit(X_scaled, y)
        
            # Get the coefficients (log probabilities for each class)
            coefficients = gnb.theta_
        
            # Compute the absolute value of the coefficients for each feature
            importance = np.abs(coefficients).sum(axis=0)
        
            # Plot the feature importance scores
            plot_feature_selection_scores(importance, X.columns, title="GaussianNB Feature Importance Scores", num=num_features)
        
            # Get top 10 features based on importance
            top_features = np.argsort(importance)[::-1][:num_features]
            selected_features = X.columns[top_features]

            st.write(f"Top {num_features} Features by GaussianNB:", selected_features)
            
    elif page == "ML Classification":
        st.title("Machine Learning Classification Models")
        st.subheader("Data Preparation")
        st.subheader(f"Top {num_features} Selected Features - {feature_selection_method}")
        st.write(selected_features)
        
        # ML Models
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

        selected_model = st.selectbox("Choose ML Model", options=list(models.keys()), key="ml_model_selectbox")
        model = models[selected_model]

        if st.button("Train and Evaluate"):
            model.fit(X_train, y_train_type)  # Train on fault type
            y_pred_type = model.predict(X_test)
            
            model.fit(X_train, y_train_size)  # Train on fault size
            y_pred_size = model.predict(X_test)

            st.text("Classification Report for Fault Type")
            st.text(classification_report(y_test_type, y_pred_type))

            st.text("Classification Report for Fault Size")
            st.text(classification_report(y_test_size, y_pred_size))

            # Confusion Matrices for Fault Type and Fault Size
            cm_type = confusion_matrix(y_test_type, y_pred_type)
            cm_size = confusion_matrix(y_test_size, y_pred_size)

            st.subheader("Confusion Matrices")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            sns.heatmap(cm_type, annot=True, fmt="d", cmap="Blues", ax=ax1)
            ax1.set_title("Fault Type Confusion Matrix")
            ax1.set_xlabel("Predicted")
            ax1.set_ylabel("Actual")

            sns.heatmap(cm_size, annot=True, fmt="d", cmap="Blues", ax=ax2)
            ax2.set_title("Fault Size Confusion Matrix")
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")

            st.pyplot(fig)

            if hasattr(model, "predict_proba"):
                y_prob_type = model.predict_proba(X_test)
                y_prob_size = model.predict_proba(X_test)
                
                st.subheader("ROC Curve for Fault Type")
                fig_roc_type, ax_roc_type = plt.subplots()
                for i in range(y_prob_type.shape[1]):
                    fpr, tpr, _ = roc_curve(y_test_type == i, y_prob_type[:, i])
                    ax_roc_type.plot(fpr, tpr, label=f"Class {i} (AUC: {auc(fpr, tpr):.2f})")
                ax_roc_type.plot([0, 1], [0, 1], 'k--')
                ax_roc_type.set_xlabel("False Positive Rate")
                ax_roc_type.set_ylabel("True Positive Rate")
                ax_roc_type.legend()
                st.pyplot(fig_roc_type)

                st.subheader("ROC Curve for Fault Size")
                fig_roc_size, ax_roc_size = plt.subplots()
                for i in range(y_prob_size.shape[1]):
                    fpr, tpr, _ = roc_curve(y_test_size == i, y_prob_size[:, i])
                    ax_roc_size.plot(fpr, tpr, label=f"Class {i} (AUC: {auc(fpr, tpr):.2f})")
                ax_roc_size.plot([0, 1], [0, 1], 'k--')
                ax_roc_size.set_xlabel("False Positive Rate")
                ax_roc_size.set_ylabel("True Positive Rate")
                ax_roc_size.legend()
                st.pyplot(fig_roc_size)

    elif page == "DL Models":
        st.subheader(f"Top {num_features} Selected Features - {feature_selection_method}")
        st.write(selected_features)
             
        # Preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        X = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    
        fault_type_encoded = LabelEncoder().fit_transform(df['fault_type'])
        fault_size_encoded = LabelEncoder().fit_transform(df['fault_size'])
    
        ohe = OneHotEncoder(sparse_output=False)
        y_type = ohe.fit_transform(fault_type_encoded.reshape(-1, 1))
        y_size = ohe.fit_transform(fault_size_encoded.reshape(-1, 1))
    
        X_train, X_test, y_type_train, y_type_test, y_size_train, y_size_test = train_test_split(
            X, y_type, y_size, test_size=0.2, random_state=42
        )
    
        # Model Selection
        st.header("🧠 Deep Learning Model")
        model_type = st.selectbox("Select DL Model", [
            "MLP", "CNN1D", "LSTM1D", "GRU1D", "BiLSTM1D", "ResNet1D",
            "Transformer1D", "DenseNet1D", "CNN+BiGRU", "CNN+Attention"
            ],
            key="dl_model_selectbox"
        )
    
        # Custom Attention Layer
        class Attention1D(tf.keras.layers.Layer):
            def call(self, inputs):
                score = tf.keras.layers.Dense(1, activation='tanh')(inputs)
                weights = tf.nn.softmax(score, axis=1)
                output = tf.reduce_sum(inputs * weights, axis=1)
                return output
        
        # Build the Model
        def build_model(input_shape, type_output, size_output, model_type):
            inputs = Input(shape=input_shape)  # Use Functional API to define inputs
            x = inputs  # Start with the input layer
            
            if model_type == "MLP":
                x = Flatten()(x)
                x = Dense(128, activation='relu')(x)
            elif model_type == "CNN1D":
                x = Conv1D(64, 3, activation='relu')(x)
                x = MaxPooling1D()(x)
                x = Flatten()(x)
            elif model_type == "LSTM1D":
                x = LSTM(64)(x)
            elif model_type == "GRU1D":
                x = GRU(64)(x)
            elif model_type == "BiLSTM1D":
                x = Bidirectional(LSTM(64))(x)
            elif model_type == "ResNet1D":
                x = Conv1D(64, 3, activation='relu', padding='same')(x)
                x = Conv1D(64, 3, activation='relu', padding='same')(x)
                x = tf.keras.layers.Add()([inputs, x])  # Skip connection
                x = Flatten()(x)
            elif model_type == "CNN+BiGRU":
                x = Conv1D(32, 3, activation='relu')(x)
                x = Bidirectional(GRU(64))(x)
            elif model_type == "CNN+Attention":
                x = Conv1D(64, 3, activation='relu')(x)
                x = Attention1D()(x)  # Apply custom attention layer
            else:
                x = Flatten()(x)
            
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.3)(x)
            
            # Outputs
            type_out = Dense(type_output.shape[1], activation='softmax', name='type_output')(x)
            size_out = Dense(size_output.shape[1], activation='softmax', name='size_output')(x)
            
            # Define model
            model = Model(inputs=inputs, outputs=[type_out, size_out])
            
            return compile_model(model, type_output, size_output)
        
        # Compile Model
        def compile_model(model, type_output, size_output):
            model.compile(
                optimizer=Adam(),
                loss={"type_output": "categorical_crossentropy", "size_output": "categorical_crossentropy"},
                metrics={"type_output": "accuracy", "size_output": "accuracy"}
            )
            return model

        # Model checkpoint callback
        checkpoint_cb = ModelCheckpoint('best_model.weights.h5', save_best_only=True, save_weights_only=True)
        
        # Training Button
        if st.button("🚀 Train Model"):
            model = build_model(X_train.shape[1:], y_type_train, y_size_train, model_type)
            
            # Train with progress bar
            progress_bar = st.progress(0)  # Streamlit progress bar
            
            history = model.fit(
                X_train, {"type_output": y_type_train, "size_output": y_size_train},
                validation_data=(X_test, {"type_output": y_type_test, "size_output": y_size_test}),
                epochs=10, batch_size=32, verbose=1,
                callbacks=[checkpoint_cb],
                steps_per_epoch=len(X_train) // 32
            )
            
            st.success("✅ Training complete.")
            st.subheader("📊 Evaluation")
            
            # Evaluate the model
            results = model.evaluate(X_test, {"type_output": y_type_test, "size_output": y_size_test}, verbose=0)
            
            # The results will be in the form: [loss, type_loss, size_loss, type_acc, size_acc]
            loss = results[0]  # The overall loss
            type_loss = results[1]
            size_loss = results[2]
            type_acc = results[3]
            size_acc = results[4]
            
            # Display results
            st.write(f"**Overall Loss:** {loss:.4f}")
            st.write(f"**Type Loss:** {type_loss:.4f}")
            st.write(f"**Size Loss:** {size_loss:.4f}")
            st.write(f"**Fault Type Accuracy:** {type_acc:.2f}")
            st.write(f"**Fault Size Accuracy:** {size_acc:.2f}")
            
            # Detailed classification report
            y_type_pred = model.predict(X_test)[0]
            y_size_pred = model.predict(X_test)[1]
            y_type_pred_labels = np.argmax(y_type_pred, axis=1)
            y_size_pred_labels = np.argmax(y_size_pred, axis=1)
            
            type_report = classification_report(np.argmax(y_type_test, axis=1), y_type_pred_labels, target_names=ohe.categories_[0])
            size_report = classification_report(np.argmax(y_size_test, axis=1), y_size_pred_labels, target_names=ohe.categories_[0])
            
            st.text("Fault Type Classification Report:\n" + type_report)
            st.text("Fault Size Classification Report:\n" + size_report)
            
            st.subheader("📈 Training History")
            fig, ax = plt.subplots(2, 2, figsize=(10, 6))
            ax[0, 0].plot(history.history['type_output_accuracy'], label='Train')
            ax[0, 0].plot(history.history['val_type_output_accuracy'], label='Val')
            ax[0, 0].legend(); ax[0, 0].set_title("Type Accuracy")
            
            ax[0, 1].plot(history.history['size_output_accuracy'], label='Train')
            ax[0, 1].plot(history.history['val_size_output_accuracy'], label='Val')
            ax[0, 1].legend(); ax[0, 1].set_title("Size Accuracy")
            
            ax[1, 0].plot(history.history['type_output_loss'], label='Train')
            ax[1, 0].plot(history.history['val_type_output_loss'], label='Val')
            ax[1, 0].legend(); ax[1, 0].set_title("Type Loss")
            
            ax[1, 1].plot(history.history['size_output_loss'], label='Train')
            ax[1, 1].plot(history.history['val_size_output_loss'], label='Val')
            ax[1, 1].legend(); ax[1, 1].set_title("Size Loss")
            
            st.pyplot(fig)
            
else:
    st.info("Please enter a valid GitHub URL or upload a file to begin.")
