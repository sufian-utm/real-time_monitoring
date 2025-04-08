import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="CWRU Time-Domain Feature Explorer", layout="wide")

# Load data from GitHub or local file
@st.cache_data
def load_data():
    url = st.text_input("Enter CSV URL from GitHub or upload a file:",
                        value="https://github.com/sufian-utm/real-time_monitoring/blob/main/data/cwru/12k_DE_td_features.csv")
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
    st.title("CWRU Time-Domain Feature Visualization")
    st.subheader("Preview of Dataset")
    st.dataframe(df.head(), use_container_width=True)

    if 'fault' not in df.columns:
        st.warning("Dataset must contain a 'fault' column encoding fault type and size.")
    else:
        # Extract fault type and size from the 'fault' column
        df['fault_type'] = df['fault'].astype(str).str.extract(r'^(IR|OR|B|Normal)')
        df['fault_size'] = df['fault'].astype(str).str.extract(r'(\d{3})')
        df['fault_size'] = df['fault_size'].fillna('000')  # Normal cases get 0 fault size

        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop(['fault_size'], errors='ignore')
        selected_feature = st.selectbox("Select a Feature to Visualize", options=numeric_cols)

        # Histogram
        st.subheader(f"Histogram of {selected_feature}")
        fig1 = px.histogram(df, x=selected_feature, color="fault_type", marginal="box", nbins=40)
        st.plotly_chart(fig1, use_container_width=True)

        # Boxplot
        st.subheader(f"Boxplot of {selected_feature} by Fault Type")
        fig2 = px.box(df, x="fault_type", y=selected_feature, color="fault_type", points="all")
        st.plotly_chart(fig2, use_container_width=True)

        # Correlation Heatmap
        if st.checkbox("Show Correlation Heatmap"):
            st.subheader("Feature Correlation Heatmap")
            fig3, ax = plt.subplots(figsize=(12, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig3)
else:
    st.info("Please enter a valid GitHub URL or upload a file to begin.")
