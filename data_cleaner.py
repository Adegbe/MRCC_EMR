import pandas as pd
import numpy as np
import json
import os
import zipfile
import re
from datetime import datetime
from ydata_profiling import ProfileReport
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

        self.column_index_map = {col: idx+1 for idx, col in enumerate(df.columns)}
        self.report_data['original_rows'] = len(df)
        self.report_data['original_columns'] = len(df.columns)
        self.report_data['sample_data'] = df.head(5).to_dict('records')
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        original_columns = df.columns.tolist()
        self.validated_columns = self._get_validated_columns(df)
        df = self._normalize_column_names(df)
        df = self._clean_string_data(df)
        df = self._handle_missing_data(df)
        df = self._correct_wrong_entries(df)
        df = self._handle_duplicates(df)
        df = self._mask_pii(df)
        if self.custom_rules:
            df = self._apply_custom_rules(df)
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
        validated = []
        if 'age' in df.columns:
            validated.append('age')
        if 'blood_type' in df.columns:
            validated.append('blood_type')
        date_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['date', 'dob', 'time'])]
        validated.extend(date_cols)
        return validated

    def generate_report(self, df: pd.DataFrame, report_path: str = None) -> dict:
        profile = ProfileReport(df, explorative=True)
        self.report_data['data_types'] = dict(df.dtypes.astype(str))
        self.report_data['missing_values'] = df.isnull().sum().to_dict()
        self.report_data['unique_values'] = df.nunique().to_dict()
        numeric_cols = df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            self.report_data['numeric_stats'] = df[numeric_cols].describe().to_dict()
        if report_path:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            profile.to_file(report_path)
            self.report_data['report_path'] = report_path
        return self.report_data

    def export_data(self, df: pd.DataFrame, output_path: str, format: str = 'csv', 
                   include_report: bool = False, zip_output: bool = False) -> Union[str, None]:
        output_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        if format.lower() == 'csv':
            data_file = os.path.join(output_dir, f"{base_name}_cleaned.csv")
            df.to_csv(data_file, index=False)
        elif format.lower() in ('xls', 'xlsx'):
            data_file = os.path.join(output_dir, f"{base_name}_cleaned.xlsx")
            df.to_excel(data_file, index=False)
        else:
            raise ValueError("Unsupported export format. Use 'csv' or 'excel'.")
        report_file = None
        if include_report:
            report_file = os.path.join(output_dir, f"{base_name}_report.json")
            with open(report_file, 'w') as f:
                json.dump(self.report_data, f, indent=2)
        if zip_output:
            zip_file = os.path.join(output_dir, f"{base_name}_bundle.zip")
            with zipfile.ZipFile(zip_file, 'w') as zipf:
                zipf.write(data_file, os.path.basename(data_file))
                if report_file:
                    zipf.write(report_file, os.path.basename(report_file))
            os.remove(data_file)
            if report_file:
                os.remove(report_file)
            return zip_file
        return data_file

    def set_pii_columns(self, columns: List[str]):
        self.pii_columns = columns

    def set_duplicate_key_columns(self, columns: List[str]):
        self.duplicate_key_columns = columns

    def add_custom_rule(self, rule_name: str, rule_func: Callable[[pd.DataFrame], pd.DataFrame]):
        self.custom_rules[rule_name] = rule_func

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [str(col).strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        return df

    def _clean_string_data(self, df: pd.DataFrame) -> pd.DataFrame:
        string_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in string_cols:
            df[col] = df[col].astype(str).str.strip()
            if 'gender' in col or 'sex' in col:
                df[col] = df[col].str.lower().replace({
                    'm': 'male', 'male': 'male', 'f': 'female', 'female': 'female', '0': 'male', '1': 'female'
                })
        return df

    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        for col in df.columns:
            if col in self.validated_columns:
                continue
            df[col] = df[col].fillna('').astype(str).str.strip()
            df.loc[df[col] == '', col] = 'N'
        return df

    def _correct_wrong_entries(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if 'age' in df.columns:
            try:
                df['age'] = pd.to_numeric(df['age'], errors='coerce')
                df.loc[(df['age'] < 0) | (df['age'] > 120) | df['age'].isna(), 'age'] = "N"
            except Exception as e:
                self._log_warning(f"Age validation failed: {str(e)}")
                df['age'] = "N"
        if 'blood_type' in df.columns:
            df['blood_type'] = df['blood_type'].str.upper().str.strip()
            valid_blood_types = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']
            df.loc[~df['blood_type'].isin(valid_blood_types), 'blood_type'] = "N"
        date_cols = [col for col in self.validated_columns if any(kw in col for kw in ['date', 'dob', 'time'])]
        current_date = datetime.now()
        for col in date_cols:
            try:
                date_series = pd.to_datetime(df[col], errors='coerce')
                invalid_mask = (date_series > current_date) | date_series.isna()
                df.loc[invalid_mask, col] = "N"
                df.loc[~invalid_mask, col] = date_series[~invalid_mask].dt.strftime('%Y-%m-%d')
            except Exception as e:
                self._log_warning(f"Date validation failed for {col}: {str(e)}")
                df[col] = "N"
        for col in self.validated_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
                df.loc[df[col].str.strip() == '', col] = "N"
        return df

    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if not self.duplicate_key_columns:
            duplicate_mask = df.duplicated(keep='first')
        else:
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
        if not self.pii_columns or df.empty:
            return df
        for col in self.pii_columns:
            if col in df.columns:
                col_idx = self.column_index_map.get(col, 0)
                df[col] = f'MRCC{col_idx}'
        return df

    def _apply_custom_rules(self, df: pd.DataFrame) -> pd.DataFrame:
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
        timestamp = datetime.now().isoformat()
        self.warning_log.append({'timestamp': timestamp, 'message': message})
        print(f"[WARNING] {timestamp}: {message}")
