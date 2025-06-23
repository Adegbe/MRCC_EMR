import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from data_cleaner import DataCleaner  # Assuming your DataCleaner is in data_cleaner.py
import tempfile

st.title("Data Cleaning Dashboard")
st.write("Upload your dataset for automatic cleaning and profiling")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "json"])

if uploaded_file:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name
    
    cleaner = DataCleaner()
    
    try:
        # Load file
        df = cleaner.load_file(file_path)
        
        # Set PII and duplicate columns (customize as needed)
        cleaner.set_pii_columns([col for col in df.columns if "name" in col.lower() or "email" in col.lower()])
        cleaner.set_duplicate_key_columns(["Patient ID", "Record ID"])
        
        # Clean data
        with st.spinner("Cleaning data..."):
            cleaned_df = cleaner.clean_data(df)
        
        # Show results
        st.success(f"Data cleaned successfully! Original: {df.shape[0]} rows, {df.shape[1]} columns → Cleaned: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")
        
        tab1, tab2 = st.tabs(["Cleaned Data", "Profiling Report"])
        
        with tab1:
            st.dataframe(cleaned_df)
            
        with tab2:
            with st.spinner("Generating report..."):
                report_path = "profile_report.html"
                cleaner.generate_report(cleaned_df, report_path)
            
            st.components.v1.html(open(report_path, "r").read(), height=1000, scrolling=True)
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
