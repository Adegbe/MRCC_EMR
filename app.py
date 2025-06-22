import streamlit as st
import pandas as pd
import os
from mrcc_emr.data_cleaner import DataCleaner
from io import BytesIO

# --- App Title ---
st.set_page_config(page_title="MRCC EMR Preprocessing Tool", layout="wide")
st.title("🧹 MRCC EMR Preprocessing Tool")

# --- Sidebar Options ---
st.sidebar.header("Preprocessing Options")
normalize_cols = st.sidebar.checkbox("Normalize Column Names", value=True)
standardize_gender = st.sidebar.checkbox("Standardize Gender Field", value=True)
handle_missing = st.sidebar.checkbox("Handle Missing Values", value=True)
remove_duplicates = st.sidebar.checkbox("Detect and Remove Duplicates", value=True)
mask_pii = st.sidebar.checkbox("Mask PII (Personally Identifiable Information)", value=False)
correct_invalid = st.sidebar.checkbox("Correct Invalid Entries", value=True)
validate_types = st.sidebar.checkbox("Validate Data Types", value=True)

# --- File Upload ---
uploaded_file = st.file_uploader("Upload CSV, Excel, or JSON file", type=["csv", "xls", "xlsx", "json"])

# --- PII + Duplicate columns ---
pii_columns = st.text_input("Columns to mask as PII (comma-separated)", value="name,email,phone")
dup_columns = st.text_input("Columns to detect duplicates by (comma-separated)", value="id")

# --- Notes Section ---
st.markdown("""
#### ℹ️ Notes:
This tool is designed for internal use only. Ensure all data handling complies with privacy regulations.
""")

# --- Main Logic ---
if uploaded_file:
    cleaner = DataCleaner()
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    temp_path = f"temp_uploaded{ext}"
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getvalue())

    df = cleaner.load_file(temp_path)

    # Configure options
    cleaner.set_pii_columns([col.strip() for col in pii_columns.split(",")])
    cleaner.set_duplicate_key_columns([col.strip() for col in dup_columns.split(",")])

    # Apply steps based on UI toggles
    if normalize_cols:
        df = cleaner._normalize_column_names(df)
    if standardize_gender:
        df = cleaner._clean_string_data(df)
    if handle_missing:
        df = cleaner._handle_missing_data(df)
    if correct_invalid:
        df = cleaner._correct_wrong_entries(df)
    if remove_duplicates:
        df = cleaner._handle_duplicates(df)
    if mask_pii:
        df = cleaner._mask_pii(df)

    st.success("✅ Data processing complete")
    st.dataframe(df.head(50))

    # Download cleaned file
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Cleaned CSV", csv_buffer.getvalue(),
                       file_name="cleaned_data.csv", mime="text/csv")

    # Optionally render profiling report
    if st.checkbox("Generate and show data profiling report"):
        report_dict = cleaner.generate_report(df)
        st.json(report_dict)
else:
    st.info("Upload a file to begin preprocessing.")
