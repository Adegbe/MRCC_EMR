
import pandas as pd
import numpy as np
import json
import os
import zipfile
import re
from datetime import datetime
from pandas_profiling import ProfileReport
from typing import Union, Dict, List, Optional, Callable

class DataCleaner:
    def __init__(self):
        self.report_data = {}
        self.warning_log = []
        self.pii_columns = []
        self.duplicate_key_columns = []
        self.custom_rules = {}
        self.column_index_map = {}
        self.validated_columns = []
        
    def load_file(self, file_path: str, low_memory: bool = False) -> pd.DataFrame:
        """Load data from various file formats with auto-detection"""
        self.report_data['original_file'] = os.path.basename(file_path)
        self.report_data['load_time'] = datetime.now().isoformat()
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                df = pd.read_csv(file_path, low_memory=low_memory)
            elif file_ext in ('.xls', '.xlsx'):
                df = pd.read_excel(file_path)
            elif file_ext == '.json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        except Exception as e:
            self._log_warning(f"File loading failed: {str(e)}")
            raise
            
        # Create column index mapping (1-indexed)
        self.column_index_map = {col: idx+1 for idx, col in enumerate(df.columns)}
            
        self.report_data['original_rows'] = len(df)
        self.report_data['original_columns'] = len(df.columns)
        self.report_data['sample_data'] = df.head(5).to_dict('records')
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main cleaning pipeline"""
        # Store original columns for tracking
        original_columns = df.columns.tolist()
        
        # Identify columns that need special validation
        self.validated_columns = self._get_validated_columns(df)
        
        # Step 1: Normalize column names
        df = self._normalize_column_names(df)
        
        # Step 2: Clean string data
        df = self._clean_string_data(df)
        
        # Step 3: Handle missing data
        df = self._handle_missing_data(df)
        
        # Step 4: Detect and correct wrong entries
        df = self._correct_wrong_entries(df)
        
        # Step 5: Handle duplicates
        df = self._handle_duplicates(df)
        
        # Step 6: Mask PII data
        df = self._mask_pii(df)
        
        # Step 7: Apply custom rules if any
        if self.custom_rules:
            df = self._apply_custom_rules(df)
            
        # Track column changes
        self.report_data['column_changes'] = {
            'original_columns': original_columns,
            'final_columns': df.columns.tolist(),
            'added_columns': [col for col in df.columns if col not in original_columns],
            'removed_columns': [col for col in original_columns if col not in df.columns]
        }
        
        self.report_data['final_rows'] = len(df)
        self.report_data['final_columns'] = len(df.columns)
        self.report_data['warnings_count'] = len(self.warning_log)
        self.report_data['processing_time'] = datetime.now().isoformat()
        
        return df
    
    def _get_validated_columns(self, df: pd.DataFrame) -> list:
        """Identify columns that need special validation"""
        validated = []
        if 'age' in df.columns:
            validated.append('age')
        if 'blood_type' in df.columns:
            validated.append('blood_type')
        
        # Add date-related columns
        date_cols = [col for col in df.columns if any(kw in col.lower() 
                    for kw in ['date', 'dob', 'time'])]
        validated.extend(date_cols)
        
        return validated
    
    def generate_report(self, df: pd.DataFrame, report_path: str = None) -> dict:
        """Generate comprehensive data profile report"""
        profile = ProfileReport(df, explorative=True)
        
        # Basic statistics
        self.report_data['data_types'] = dict(df.dtypes.astype(str))
        self.report_data['missing_values'] = df.isnull().sum().to_dict()
        self.report_data['unique_values'] = df.nunique().to_dict()
        
        # Descriptive stats for numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            self.report_data['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        # Save report if path provided
        if report_path:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            profile.to_file(report_path)
            self.report_data['report_path'] = report_path
            
        return self.report_data
    
    def export_data(self, df: pd.DataFrame, output_path: str, format: str = 'csv', 
                   include_report: bool = False, zip_output: bool = False) -> Union[str, None]:
        """Export cleaned data with various options"""
        output_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Export main data file
        if format.lower() == 'csv':
            data_file = os.path.join(output_dir, f"{base_name}_cleaned.csv")
            df.to_csv(data_file, index=False)
        elif format.lower() in ('xls', 'xlsx'):
            data_file = os.path.join(output_dir, f"{base_name}_cleaned.xlsx")
            df.to_excel(data_file, index=False)
        else:
            raise ValueError("Unsupported export format. Use 'csv' or 'excel'.")
        
        # Export report if requested
        report_file = None
        if include_report:
            report_file = os.path.join(output_dir, f"{base_name}_report.json")
            with open(report_file, 'w') as f:
                json.dump(self.report_data, f, indent=2)
        
        # Create ZIP bundle if requested
        if zip_output:
            zip_file = os.path.join(output_dir, f"{base_name}_bundle.zip")
            with zipfile.ZipFile(zip_file, 'w') as zipf:
                zipf.write(data_file, os.path.basename(data_file))
                if report_file:
                    zipf.write(report_file, os.path.basename(report_file))
            
            # Clean up individual files if we created a zip
            os.remove(data_file)
            if report_file:
                os.remove(report_file)
                
            return zip_file
        
        return data_file
    
    def set_pii_columns(self, columns: List[str]):
        """Set which columns contain PII that should be masked"""
        self.pii_columns = columns
        
    def set_duplicate_key_columns(self, columns: List[str]):
        """Set which columns to use for duplicate detection"""
        self.duplicate_key_columns = columns
        
    def add_custom_rule(self, rule_name: str, rule_func: Callable[[pd.DataFrame], pd.DataFrame]):
        """Add a custom data cleaning rule"""
        self.custom_rules[rule_name] = rule_func
    
    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase with underscores"""
        new_columns = []
        for col in df.columns:
            new_col = (
                str(col).strip()
                .lower()
                .replace(' ', '_')
                .replace('-', '_')
                .replace(r'[^\w_]+', '', regex=True)
            )
            new_columns.append(new_col)
        df.columns = new_columns
        return df
    
    def _clean_string_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean string data in the dataframe"""
        string_cols = df.select_dtypes(include=['object', 'string']).columns
        
        for col in string_cols:
            # Trim whitespace
            df[col] = df[col].astype(str).str.strip()
            
            # Standardize gender-like columns
            if 'gender' in col or 'sex' in col:
                df[col] = (
                    df[col].str.lower()
                    .replace({
                        'm': 'male', 'male': 'male', 
                        'f': 'female', 'female': 'female',
                        '0': 'male', '1': 'female'
                    })
                )
                
        return df
    
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data - replace empty strings with 'N'"""
        if df.empty:
            return df
            
        # Convert all columns to string to handle missing values consistently
        for col in df.columns:
            # Skip validated columns - they will be handled in correction step
            if col in self.validated_columns:
                continue
                
            # Replace NaN with empty string
            df[col] = df[col].fillna('')
            # Convert to string and trim
            df[col] = df[col].astype(str).str.strip()
            # Replace empty strings with 'N'
            df.loc[df[col] == '', col] = 'N'
                    
        return df
    
    def _correct_wrong_entries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and correct wrong entries with special handling for validated columns"""
        if df.empty:
            return df
            
        # Age validation - replace invalid ages with "N"
        if 'age' in df.columns:
            try:
                # Convert to numeric, coerce errors
                df['age'] = pd.to_numeric(df['age'], errors='coerce')
                # Replace invalid ages (<0 or >120) with "N"
                invalid_age_mask = (df['age'] < 0) | (df['age'] > 120) | df['age'].isna()
                df.loc[invalid_age_mask, 'age'] = "N"
            except Exception as e:
                self._log_warning(f"Age validation failed: {str(e)}")
                df['age'] = "N"
                
        # Blood type normalization
        if 'blood_type' in df.columns:
            # Standardize blood types
            df['blood_type'] = df['blood_type'].str.upper().str.strip()
            # Replace invalid blood types with "N"
            valid_blood_types = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']
            invalid_blood_mask = ~df['blood_type'].isin(valid_blood_types)
            df.loc[invalid_blood_mask, 'blood_type'] = "N"
        
        # Date validation - replace invalid dates with "N"
        date_cols = [col for col in self.validated_columns 
                    if any(kw in col for kw in ['date', 'dob', 'time'])]
        current_date = datetime.now()
        
        for col in date_cols:
            try:
                # Convert to datetime
                date_series = pd.to_datetime(df[col], errors='coerce')
                
                # Replace invalid dates (future dates or unparseable) with "N"
                invalid_mask = (date_series > current_date) | date_series.isna()
                invalid_count = invalid_mask.sum()
                
                if invalid_count > 0:
                    df.loc[invalid_mask, col] = "N"
                    
                # Format valid dates to string
                valid_mask = ~invalid_mask
                df.loc[valid_mask, col] = date_series[valid_mask].dt.strftime('%Y-%m-%d')
            except Exception as e:
                self._log_warning(f"Date validation failed for {col}: {str(e)}")
                df[col] = "N"
                    
        # Replace any remaining invalid values with "N" for validated columns
        for col in self.validated_columns:
            if col in df.columns:
                # Replace empty strings and whitespace-only with "N"
                df[col] = df[col].astype(str)
                empty_mask = df[col].str.strip().eq('')
                df.loc[empty_mask, col] = "N"
        
        return df
    
    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate rows based on key columns"""
        if df.empty:
            return df
            
        if not self.duplicate_key_columns:
            # If no key columns specified, check all columns
            duplicate_mask = df.duplicated(keep='first')
        else:
            # Check for duplicates using specified key columns
            # Only use columns that exist in the DataFrame
            valid_columns = [col for col in self.duplicate_key_columns if col in df.columns]
            if not valid_columns:
                self._log_warning("No valid columns for duplicate detection")
                return df
                
            duplicate_mask = df.duplicated(subset=valid_columns, keep='first')
        
        duplicate_count = duplicate_mask.sum()
        self.report_data['duplicate_count'] = duplicate_count
        
        if duplicate_count > 0:
            self._log_warning(f"Found {duplicate_count} duplicate rows")
            df = df[~duplicate_mask]
            
        return df
    
    def _mask_pii(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mask PII data using MRCC + column number"""
        if not self.pii_columns or df.empty:
            return df
            
        for col in self.pii_columns:
            if col in df.columns:
                # Get original column index
                col_idx = self.column_index_map.get(col, 0)
                # Replace all values with MRCC + column number
                df[col] = f'MRCC{col_idx}'
                
        return df
    
    def _apply_custom_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply user-defined cleaning rules"""
        if df.empty:
            return df
            
        for rule_name, rule_func in self.custom_rules.items():
            try:
                df = rule_func(df)
                self._log_warning(f"Applied custom rule: {rule_name}")
            except Exception as e:
                self._log_warning(f"Failed to apply custom rule {rule_name}: {str(e)}")
                
        return df
    
    def _log_warning(self, message: str):
        """Log a warning message with timestamp"""
        timestamp = datetime.now().isoformat()
        self.warning_log.append({'timestamp': timestamp, 'message': message})
        print(f"[WARNING] {timestamp}: {message}")


if __name__ == "__main__":
    # Example usage when run directly
    print("EMR Data Cleaner - Command Line Interface")
    print("----------------------------------------")
    
    file_path = input("Enter path to data file: ").strip()
    output_dir = input("Enter output directory [./output]: ").strip() or "./output"
    output_name = input("Enter output base name [cleaned_data]: ").strip() or "cleaned_data"
    
    cleaner = DataCleaner()
    
    # Configure PII columns
    pii_input = input("Enter PII columns to mask (comma separated) [name,email,phone]: ").strip()
    pii_columns = [col.strip() for col in pii_input.split(",")] if pii_input else ['name', 'email', 'phone']
    cleaner.set_pii_columns(pii_columns)
    
    # Configure duplicate detection
    dup_input = input("Enter duplicate key columns (comma separated) [id]: ").strip()
    dup_columns = [col.strip() for col in dup_input.split(",")] if dup_input else ['id']
    cleaner.set_duplicate_key_columns(dup_columns)
    
    # Add custom rules
    add_custom = input("Add custom rules? (y/N): ").strip().lower()
    if add_custom == 'y':
        while True:
            rule_name = input("Rule name (or 'done' to finish): ").strip()
            if rule_name.lower() == 'done':
                break
            print(f"Define function for rule '{rule_name}':")
            print("Available template: def custom_rule(df):\n    # Your logic\n    return df")
            # In real implementation, you'd need a way to accept code
            print("Note: Code execution not implemented in CLI. Add rules programmatically.")
    
    try:
        print("\nLoading data...")
        df = cleaner.load_file(file_path)
        
        print("Cleaning data...")
        cleaned_df = cleaner.clean_data(df)
        
        print("Generating report...")
        report_path = os.path.join(output_dir, f"{output_name}_report.html")
        cleaner.generate_report(cleaned_df, report_path)
        
        print("Exporting data...")
        output_path = os.path.join(output_dir, output_name)
        result_file = cleaner.export_data(
            cleaned_df, 
            output_path=output_path,
            format='csv',
            include_report=True,
            zip_output=True
        )
        
        print(f"\nProcessing complete. Output saved to: {result_file}")
        print(f"Report generated at: {report_path}")
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
