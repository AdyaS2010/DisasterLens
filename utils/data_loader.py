import os

import streamlit as st
import pandas as pd

SAMPLE_DATA_PATH = os.path.join("data", "sample_disaster_tweets.csv")


@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with Streamlit caching."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Data file not found: {file_path}\n"
            f"Make sure the file exists in the project directory."
        )
    return pd.read_csv(file_path)


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    """Load the built-in sample disaster tweets dataset."""
    return load_data(SAMPLE_DATA_PATH)


def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    """Load data from a Streamlit UploadedFile object."""
    if uploaded_file is None:
        raise ValueError("No file was uploaded. Please upload a CSV file.")
    return pd.read_csv(uploaded_file)
