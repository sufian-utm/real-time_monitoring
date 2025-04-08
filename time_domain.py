import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import flwr as fl  # Federated Learning

st.set_page_config(page_title="Feature Explorer", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Feature Visualizer", "ML Classification", "DL Models", "Federated Learning"])

# Load data from GitHub or local file
@st.cache_data
def load_data():
    url = st.text_input("Enter CSV URL from GitHub or upload a file:",
                        value="https://raw.githubusercontent.com/username/repo/main/cwru_features.csv")
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

        # Machine Learning Models (Random Forest, SVM)
        st.subheader("Random Forest Classifier")
        X = df[numeric_cols]
        y = df['fault_type']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_model = RandomForestClassifier(n_estimators=100)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        st.write(f"Random Forest Classifier Accuracy: {accuracy_rf:.2f}")

        st.subheader("Support Vector Machine")
        svm_model = SVC(kernel='linear')
        svm_model.fit(X_train, y_train)
        y_pred_svm = svm_model.predict(X_test)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        st.write(f"SVM Accuracy: {accuracy_svm:.2f}")

    elif page == "DL Models":
        st.title("Deep Learning Models")
        st.write("Training deep learning models with time-domain features...")

        # Deep Learning Model (e.g., Simple MLP)
        st.subheader("Feedforward Neural Network (MLP)")
        X = df[numeric_cols].values
        y = pd.get_dummies(df['fault_type']).values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = Sequential()
        model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(y_train.shape[1], activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Model training
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

        # Plot training history
        st.subheader("Training History")
        fig, ax = plt.subplots()
        ax.plot(history.history['accuracy'], label='Train Accuracy')
        ax.plot(history.history['val_accuracy'], label='Val Accuracy')
        ax.set_title("Accuracy Over Epochs")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.legend()
        st.pyplot(fig)

    elif page == "Federated Learning":
        st.title("Federated Learning Setup")
        st.write("Setting up Federated Learning models...")

        # Example Federated Learning using Flower
        def get_model():
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
            ])
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        # Federated Learning strategy
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=0.5,  # 50% of clients participate in each round
            fraction_evaluate=0.5,
            min_fit_clients=10,
            min_eval_clients=5,
            min_available_clients=10
        )

        # Setup federated learning simulation (details omitted)
        st.write("Federated learning setup is in progress...")

else:
    st.info("Please enter a valid GitHub URL or upload a file to begin.")
