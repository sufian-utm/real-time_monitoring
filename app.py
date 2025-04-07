import os
import streamlit as st
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.inspection import permutation_importance
import torch.nn.functional as F
from models.mlp import MLP
from models.cnn1d import CNN1D
from models.resnet1d import ResNet1D
from models.lstm1d import LSTM1D
from models.gru1d import GRU1D
from models.bilstm1d import BiLSTM1D
from models.transformer1d import Transformer1D
from models.densenet1d import DenseNet1D
from models.vgg16_1d import VGG16_1D
from models.inception1d import Inception1D
from models.xception1d import Xception1D
from models.mobilenet1d import MobileNet1D
from models.efficientnet1d import EfficientNet1D
from models.deepcnn1d import DeepCNN1D

# Function to calculate and plot SHAP values
def plot_shap_summary(model, X):
    # SHAP Explainer for the model
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    
    # Plotting the SHAP summary plot
    shap.summary_plot(shap_values, X)

# Function to calculate and plot permutation feature importance
def plot_permutation_importance(model, X, y):
    # Performing permutation importance
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    
    # Create a dataframe for importance
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': result.importances_mean})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    # Plotting permutation feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    st.pyplot()

# Function to show model prediction confidence
def show_confidence(model, X):
    with torch.no_grad():
        outputs = model(X)
        probs = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
        confidence, predicted = torch.max(probs, 1)
    
    # Display prediction confidence and predicted class
    confidence = confidence.cpu().numpy()
    predicted = predicted.cpu().numpy()
    
    st.write(f"Prediction Confidence: {confidence[0]:.4f}")
    st.write(f"Predicted Class: {predicted[0]}")

# Sidebar for model selection and visualization options
st.sidebar.title("Model Visualizations")

model_dict = {
    "MLP": MLP,
    "CNN1D": CNN1D,
    "ResNet1D": ResNet1D,
    "LSTM1D": LSTM1D,
    "GRU1D": GRU1D,
    "BiLSTM1D": BiLSTM1D,
    "Transformer1D": Transformer1D,
    "DenseNet1D": DenseNet1D,
    "VGG16_1D": VGG16_1D,
    "Inception1D": Inception1D,
    "Xception1D": Xception1D,
    "MobileNet1D": MobileNet1D,
    "EfficientNet1D": EfficientNet1D,
    "DeepCNN1D": DeepCNN1D
}

# Model selection (Here, you need to replace with actual models)
model_selection = st.sidebar.selectbox("Select a model", list(model_dict.keys()))  # Replace with actual model names

# Visualization options
feature_selection = st.sidebar.checkbox("Show Feature Importance", value=True)
confidence_selection = st.sidebar.checkbox("Show Prediction Confidence", value=True)

# Assuming you have pre-trained models saved or loaded here
# Load the model (replace with actual model loading function)
def load_model(model_name, input_size=1024, model_path_dir="models/"):
    assert model_name in model_dict, f"{model_name} is not a valid model name."
    model_class = model_dict[model_name]
    model = model_class(input_size=input_size)  # Adjust if your model takes different args
    model_path = os.path.join(model_path_dir, f"{model_name}.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# Example of how test data might be loaded
def load_test_data():
    # Example test data placeholder
    # Replace with actual test dataset loading
    X_test = pd.DataFrame(np.random.rand(10, 5), columns=[f"Feature {i}" for i in range(5)])
    y_test = pd.Series(np.random.randint(0, 4, size=10))  # Assuming 4 classes
    return X_test, y_test

# Load the selected model
model = load_model(model_selection)

# Load test data
X_test, y_test = load_test_data()

# Show selected visualizations
if feature_selection:
    st.subheader("SHAP Feature Importance")
    plot_shap_summary(model, X_test)

    st.subheader("Permutation Feature Importance")
    plot_permutation_importance(model, X_test, y_test)

if confidence_selection:
    st.subheader("Prediction Confidence")
    show_confidence(model, X_test)
