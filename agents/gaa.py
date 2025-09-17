# gaa.py

import pandas as pd
from sdv.tabular import CTGAN
import os

class GenerativeAugmentationAgent:
    def __init__(self):
        self.model = CTGAN()
        self.augmented_data_path = "data/augmented/augmented_data.csv"

    def is_imbalanced(self, df, target_col='true_label'):
        if target_col not in df.columns:
            return False
        counts = df[target_col].value_counts()
        if len(counts) < 2:
            return True
        imbalance_ratio = min(counts) / max(counts)
        return imbalance_ratio < 0.5

    def process(self, file_path):
        """Original method for processing CSV files"""
        df = pd.read_csv(file_path)
        if 'label' not in df.columns:
            raise ValueError("The file does not contain a 'label' column.")

        if self.is_imbalanced(df):
            print("Imbalance detected. Synthetic data generation in progress...")
            self.model.fit(df)
            synthetic_samples = self.model.sample(len(df))
            df_augmented = pd.concat([df, synthetic_samples], ignore_index=True)
        else:
            print("No significant imbalance. Lightweight generation of synthetic data...")
            self.model.fit(df)
            synthetic_samples = self.model.sample(int(0.3 * len(df)))
            df_augmented = pd.concat([df, synthetic_samples], ignore_index=True)

        os.makedirs(os.path.dirname(self.augmented_data_path), exist_ok=True)
        df_augmented.to_csv(self.augmented_data_path, index=False)
        print(f"Augmented data recorded at: {self.augmented_data_path}")
        return self.augmented_data_path

    def augment_if_needed(self, files_list):
        """New method to handle list of dictionaries from UIA"""
        print("[GAA] Checking dataset balance...")
        
        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(files_list)
        
        # Check if augmentation is needed
        if self.is_imbalanced(df, 'true_label'):
            print("[GAA] Imbalance detected. Generating synthetic data...")
            
            # Prepare data for CTGAN (only numeric features)
            numeric_cols = [col for col in df.columns if col.startswith('feature')]
            numeric_cols.append('true_label')  # Add target column
            df_for_training = df[numeric_cols].copy()
            
            try:
                # Fit and generate synthetic data
                self.model.fit(df_for_training)
                synthetic_samples = self.model.sample(len(df_for_training))
                
                # Add synthetic samples back to original format
                synthetic_files = []
                for idx, row in synthetic_samples.iterrows():
                    synthetic_file = files_list[idx % len(files_list)].copy()  # Use template
                    # Update with synthetic features
                    for col in numeric_cols:
                        if col in row:
                            synthetic_file[col] = row[col]
                    synthetic_file['file_id'] = f"synthetic_{idx}"
                    synthetic_files.append(synthetic_file)
                
                # Combine original and synthetic data
                balanced_files = files_list + synthetic_files
                print(f"[GAA] Generated {len(synthetic_files)} synthetic samples")
                
            except Exception as e:
                print(f"[GAA] Error during augmentation: {e}. Using original data.")
                balanced_files = files_list
        else:
            print("[GAA] Dataset is balanced. No augmentation needed.")
            balanced_files = files_list
            
        return balanced_files

    def generer_data_equilibrees(self, df):
        """Method for compatibility with PlanningAgent"""
        if self.is_imbalanced(df, 'true_label'):
            print("[GAA] Generating balanced data...")
            # Prepare numeric data for CTGAN
            numeric_cols = [col for col in df.columns if col.startswith('feature')]
            if 'true_label' in df.columns:
                numeric_cols.append('true_label')
            
            df_for_training = df[numeric_cols].copy()
            
            try:
                self.model.fit(df_for_training)
                synthetic_samples = self.model.sample(len(df_for_training))
                df_augmented = pd.concat([df_for_training, synthetic_samples], ignore_index=True)
                return df_augmented
            except Exception as e:
                print(f"[GAA] Error during augmentation: {e}. Using original data.")
                return df
        else:
            print("[GAA] Dataset is already balanced.")
            return df