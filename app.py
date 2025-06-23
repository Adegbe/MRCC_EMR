import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from data_cleaner import DataCleaner
import tempfile
import os
import json
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="🧹 Data Cleaning Dashboard",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #1E90FF;
    }
    .st-bb {
        background-color: #f0f2f6;
    }
    .st-at {
        background-color: #ffffff;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stDownloadButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
    }
    .stDownloadButton button:hover {
        background-color: #45a049;
    }
    .stAlert {
        border-radius: 8px;
    }
    .stSpinner > div > div {
        border-top-color: #1E90FF !important;
    }
</style>
""", unsafe_allow_html=True)

# Main app
st.title("🧹 Data Cleaning Dashboard")
st.markdown("Upload your dataset for automatic cleaning and profiling")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "json"])

if uploaded_file:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name
    
    cleaner = DataCleaner()
    
    try:
        # Load file
        with st.spinner("📥 Loading file..."):
            df = cleaner.load_file(file_path)
        
        # Show original data preview
        st.subheader("📄 Original Data Preview")
        st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        st.dataframe(df.head(5))
        
        # Settings sidebar
        with st.sidebar:
            st.subheader("⚙️ Cleaning Settings")
            
            # PII columns
            st.info("Select columns containing personally identifiable information")
            pii_cols = st.multiselect(
                "PII columns to mask:",
                options=df.columns,
                default=[col for col in df.columns if any(kw in col.lower() for kw in ['name', 'email', 'phone', 'address'])]
            )
            cleaner.set_pii_columns(pii_cols)
            
            # Duplicate detection columns
            st.info("Select columns to use for duplicate detection")
            dup_cols = st.multiselect(
                "Duplicate detection columns:",
                options=df.columns,
                default=[col for col in df.columns if 'id' in col.lower() or 'code' in col.lower()]
            )
            cleaner.set_duplicate_key_columns(dup_cols)
            
            # Custom rules
            st.subheader("🔧 Custom Rules")
            st.info("Add custom rules in the code editor")
        
        # Clean data
        with st.spinner("🧼 Cleaning data..."):
            cleaned_df = cleaner.clean_data(df)
        
        # Show cleaning results
        st.subheader("✅ Cleaning Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Original Rows", df.shape[0])
        col2.metric("Cleaned Rows", cleaned_df.shape[0])
        col3.metric("Duplicates Removed", cleaner.report_data.get('duplicate_count', 0))
        
        # Show cleaned data
        st.subheader("🧽 Cleaned Data Preview")
        st.dataframe(cleaned_df.head(10))
        
        # Generate and show report
        st.subheader("📊 Data Summary Report")
        
        # Basic stats
        with st.expander("Data Types"):
            if 'data_types' in cleaner.report_data:
                st.dataframe(pd.DataFrame.from_dict(cleaner.report_data['data_types'], 
                                                   orient='index', columns=['Data Type']))
        
        with st.expander("Missing Values"):
            if 'missing_values' in cleaner.report_data:
                st.dataframe(pd.DataFrame.from_dict(cleaner.report_data['missing_values'], 
                                                   orient='index', columns=['Missing Values']))
        
        with st.expander("Unique Values"):
            if 'unique_values' in cleaner.report_data:
                st.dataframe(pd.DataFrame.from_dict(cleaner.report_data['unique_values'], 
                                                   orient='index', columns=['Unique Values']))
        
        with st.expander("Numeric Statistics"):
            if 'numeric_stats' in cleaner.report_data:
                st.dataframe(pd.DataFrame(cleaner.report_data['numeric_stats']))
        
        # Download options
        st.subheader("💾 Download Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = cleaned_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name='cleaned_data.csv',
                mime='text/csv'
            )
        
        with col2:
            # Create Excel file in memory
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                cleaned_df.to_excel(writer, index=False)
            excel_data = output.getvalue()
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name='cleaned_data.xlsx',
                mime='application/vnd.ms-excel'
            )
        
        with col3:
            report_json = json.dumps(cleaner.report_data, indent=2)
            st.download_button(
                label="Download Report",
                data=report_json,
                file_name='cleaning_report.json',
                mime='application/json'
            )
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.error("Please check the file format and try again.")
    
    finally:
        # Clean up temp file
        if os.path.exists(file_path):
            os.unlink(file_path)
else:
    st.info("ℹ️ Please upload a CSV, Excel, or JSON file to get started")
    st.markdown("""
    **Example datasets:**
    - [Sample CSV: Titanic Dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv)
    - [Sample CSV: Iris Dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv)
    """)

# Add footer
st.markdown("---")
st.markdown("### 🧹 Data Cleaning Dashboard v1.1")
st.markdown("Built with Streamlit and Pandas")
