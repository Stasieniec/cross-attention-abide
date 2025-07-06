import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

import urllib.request

class ImprovedSMRIExtractor:
    """Extract sMRI features."""
    
    def __init__(self, freesurfer_path: str, output_path: str = "processed_smri_data_improved"):
        """
        Initialize the sMRI extractor - inspired by Dong et al. paper.
        
        Args:
            freesurfer_path: Path to FreeSurfer stats directory
            output_path: Path to save processed data
        """
        self.freesurfer_path = Path(freesurfer_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True, parents=True)
        
        self.target_features = 800 
    
    def download_phenotypic_data(self) -> Path:
        """Download ABIDE phenotypic data if not present."""
        pheno_file = self.output_path / 'Phenotypic_V1_0b_preprocessed1.csv'
        
        if not pheno_file.exists():
            url = 'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv'
            urllib.request.urlretrieve(url, pheno_file)
        
        return pheno_file
    
    def parse_freesurfer_stats(self) -> dict:
        """
        Parse FreeSurfer stats files.
        
        Extracts features from:
        1. Cortical surface extraction (Desikan-Killiany atlas) - 9 features per region
        2. Subcortical parcellation - 7 features per region  
        3. White matter parcellation - similar features
        
        Returns:
            Dictionary mapping subject IDs to feature dictionaries
        """
        
        def parse_aseg_stats(file_path: Path) -> dict:
            """Parse aseg.stats file - subcortical and ventricular volumes."""
            features = {}
            
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            # Format: Index SegId NVoxels Volume_mm3 StructName ...
                            struct_name = parts[4]
                            nvoxels = float(parts[2])
                            volume = float(parts[3])
                            
                            # Additional features if available
                            norm_mean = float(parts[5]) if len(parts) > 5 else np.nan
                            norm_std = float(parts[6]) if len(parts) > 6 else np.nan
                            norm_min = float(parts[7]) if len(parts) > 7 else np.nan
                            norm_max = float(parts[8]) if len(parts) > 8 else np.nan
                            norm_range = float(parts[9]) if len(parts) > 9 else np.nan
                            
                            # Store all 7 features
                            features[f'aseg_{struct_name}_nvoxels'] = nvoxels
                            features[f'aseg_{struct_name}_volume'] = volume
                            features[f'aseg_{struct_name}_norm_mean'] = norm_mean
                            features[f'aseg_{struct_name}_norm_std'] = norm_std
                            features[f'aseg_{struct_name}_norm_min'] = norm_min
                            features[f'aseg_{struct_name}_norm_max'] = norm_max
                            features[f'aseg_{struct_name}_norm_range'] = norm_range
                            
                        except (ValueError, IndexError):
                            continue
            
            return features
        
        def parse_aparc_stats(file_path: Path, hemisphere: str) -> dict:
            """Parse aparc.stats file - cortical parcellation (Desikan-Killiany)."""
            features = {}
            
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    
                    parts = line.strip().split()
                    if len(parts) >= 10:
                        try:
                            # Format from paper: StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
                            struct_name = parts[0]
                            num_vert = float(parts[1])      # Number of vertices
                            surf_area = float(parts[2])     # Surface area
                            gray_vol = float(parts[3])      # Gray matter volume
                            thick_avg = float(parts[4])     # Average thickness
                            thick_std = float(parts[5])     # Thickness SD
                            mean_curv = float(parts[6])     # Integrated rectified mean curvature
                            gaus_curv = float(parts[7])     # Integrated rectified Gaussian curvature
                            fold_ind = float(parts[8])      # Folding index
                            curv_ind = float(parts[9])      # Intrinsic curvature index
                            
                            # Store all 9 features as specified in paper
                            prefix = f'{hemisphere}_{struct_name}'
                            features[f'{prefix}_numvert'] = num_vert
                            features[f'{prefix}_surfarea'] = surf_area
                            features[f'{prefix}_grayvol'] = gray_vol
                            features[f'{prefix}_thickavg'] = thick_avg
                            features[f'{prefix}_thickstd'] = thick_std
                            features[f'{prefix}_meancurv'] = mean_curv
                            features[f'{prefix}_gauscurv'] = gaus_curv
                            features[f'{prefix}_foldind'] = fold_ind
                            features[f'{prefix}_curvind'] = curv_ind
                            
                        except (ValueError, IndexError):
                            continue
            
            return features
        
        def parse_wmparc_stats(file_path: Path) -> dict:
            """Parse wmparc.stats file - white matter parcellation."""
            features = {}
            
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            # Similar format to aseg
                            struct_name = parts[4]
                            nvoxels = float(parts[2])
                            volume = float(parts[3])
                            
                            # Additional normalized intensity features
                            norm_mean = float(parts[5]) if len(parts) > 5 else np.nan
                            norm_std = float(parts[6]) if len(parts) > 6 else np.nan
                            norm_min = float(parts[7]) if len(parts) > 7 else np.nan
                            norm_max = float(parts[8]) if len(parts) > 8 else np.nan
                            norm_range = float(parts[9]) if len(parts) > 9 else np.nan
                            
                            features[f'wm_{struct_name}_nvoxels'] = nvoxels
                            features[f'wm_{struct_name}_volume'] = volume
                            features[f'wm_{struct_name}_norm_mean'] = norm_mean
                            features[f'wm_{struct_name}_norm_std'] = norm_std
                            features[f'wm_{struct_name}_norm_min'] = norm_min
                            features[f'wm_{struct_name}_norm_max'] = norm_max
                            features[f'wm_{struct_name}_norm_range'] = norm_range
                            
                        except (ValueError, IndexError):
                            continue
            
            return features
        
        def parse_subject_features(subject_id: str) -> dict:
            """Parse all relevant stats files for a subject."""
            subject_path = self.freesurfer_path / subject_id
            all_features = {}
            
            # Parse subcortical features (aseg.stats) - as in paper
            aseg_file = subject_path / 'aseg.stats'
            if aseg_file.exists():
                aseg_features = parse_aseg_stats(aseg_file)
                all_features.update(aseg_features)
            
            # Parse cortical features (aparc.stats for both hemispheres) - Desikan-Killiany atlas as in paper
            for hemisphere in ['lh', 'rh']:
                aparc_file = subject_path / f'{hemisphere}.aparc.stats'
                if aparc_file.exists():
                    aparc_features = parse_aparc_stats(aparc_file, hemisphere)
                    all_features.update(aparc_features)
            
            # Parse white matter features (wmparc.stats) - as mentioned in paper
            wmparc_file = subject_path / 'wmparc.stats'
            if wmparc_file.exists():
                wmparc_features = parse_wmparc_stats(wmparc_file)
                all_features.update(wmparc_features)
            
            return all_features
        
        # Get all subject directories
        subject_dirs = [d.name for d in self.freesurfer_path.iterdir() if d.is_dir()]
        subject_dirs = sorted(subject_dirs)
        
        # Process all subjects
        all_subject_features = {}
        failed_subjects = []
        
        for i, subject_id in enumerate(subject_dirs):
            try:
                features = parse_subject_features(subject_id)
                if features:  # Only add if we got some features
                    all_subject_features[subject_id] = features
                else:
                    failed_subjects.append(subject_id)
                
            except Exception as e:
                failed_subjects.append(subject_id)
        
        return all_subject_features
    
    def create_feature_matrix(self, subject_features: dict) -> tuple:
        """Convert feature dictionary to matrix format."""
        
        if not subject_features:
            raise ValueError("No subject features available!")
        
        # Get all unique feature names
        all_feature_names = set()
        for features in subject_features.values():
            all_feature_names.update(features.keys())
        
        feature_names = sorted(list(all_feature_names))
        subject_ids = sorted(list(subject_features.keys()))
        
        # Create matrix
        feature_matrix = np.zeros((len(subject_ids), len(feature_names)))
        
        for i, subject_id in enumerate(subject_ids):
            subject_data = subject_features[subject_id]
            for j, feature_name in enumerate(feature_names):
                if feature_name in subject_data:
                    feature_matrix[i, j] = subject_data[feature_name]
                else:
                    feature_matrix[i, j] = np.nan  # Missing feature
        
        # Handle missing values - replace with median
        nan_count = np.sum(np.isnan(feature_matrix))
        
        for j in range(feature_matrix.shape[1]):
            col = feature_matrix[:, j]
            if np.any(np.isnan(col)):
                median_val = np.nanmedian(col)
                feature_matrix[np.isnan(col), j] = median_val
        
        return feature_matrix, subject_ids, feature_names
    
    def match_with_phenotypic_data(self, feature_matrix: np.ndarray, fs_subject_ids: list, 
                                   phenotype_df: pd.DataFrame) -> tuple:
        """Match FreeSurfer subjects with phenotypic data."""
        
        # Convert FreeSurfer subject IDs to integers for matching
        fs_subject_ids_int = []
        for sid in fs_subject_ids:
            try:
                fs_subject_ids_int.append(int(sid))
            except ValueError:
                # Try to extract numbers from string
                import re
                match = re.search(r'\d+', str(sid))
                if match:
                    fs_subject_ids_int.append(int(match.group()))
                else:
                    fs_subject_ids_int.append(None)
        
        # Find subjects that exist in both datasets
        pheno_subjects = set(phenotype_df['SUB_ID'].astype(int))
        fs_subjects = set([sid for sid in fs_subject_ids_int if sid is not None])
        
        common_subjects = fs_subjects.intersection(pheno_subjects)
        
        # Create indices for common subjects
        fs_indices = []
        pheno_indices = []
        matched_subject_ids = []
        
        for i, fs_id in enumerate(fs_subject_ids_int):
            if fs_id in common_subjects:
                pheno_idx = phenotype_df[phenotype_df['SUB_ID'] == fs_id].index
                if len(pheno_idx) > 0:
                    fs_indices.append(i)
                    pheno_indices.append(pheno_idx[0])
                    matched_subject_ids.append(fs_id)
        
        # Extract matched data
        matched_features = feature_matrix[fs_indices]
        matched_phenotype = phenotype_df.iloc[pheno_indices]
        
        # Create labels (1=Autism, 2=Control -> 0=Control, 1=Autism) - consistent with existing code
        labels = np.where(matched_phenotype['DX_GROUP'].values == 2, 0, 1)
        
        return matched_features, labels, matched_subject_ids, matched_phenotype
    
    def apply_paper_feature_selection(self, X: np.ndarray, y: np.ndarray, 
                                      feature_names: list) -> tuple:
        """
        Apply feature selection following the paper's methodology:
        - Standardization
        - Recursive Feature Elimination with Ridge Classifier
        - Select 800 features as specified in paper
        """
        
        # Step 1: Standardization (as mentioned in paper)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Step 2: Recursive Feature Elimination with Ridge Classifier (as in paper)
        
        # Use same parameters as likely used in paper
        ridge_estimator = Ridge(alpha=1.0, random_state=42)
        
        # Apply RFE to select target number of features
        n_features_to_select = min(self.target_features, X_scaled.shape[1])
        
        rfe = RFE(
            estimator=ridge_estimator,
            n_features_to_select=n_features_to_select,
            step=0.1,  # Remove 10% of features at each step (reasonable for large feature sets)
            verbose=1
        )
        
        X_selected = rfe.fit_transform(X_scaled, y)
        
        # Get selected feature names
        selected_feature_mask = rfe.support_
        selected_feature_names = [feature_names[i] for i in range(len(feature_names)) 
                                  if selected_feature_mask[i]]
        
        return X_selected, selected_feature_names, scaler, rfe
    
    def evaluate_baseline_performance(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate baseline performance to verify feature quality."""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Test SVM classifier (same as paper)
        svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
        svm_model.fit(X_train, y_train)
        svm_pred = svm_model.predict(X_test)
        svm_acc = accuracy_score(y_test, svm_pred)
        
        return {
            'svm_accuracy': svm_acc,
            'test_size': len(y_test),
            'test_autism': int(np.sum(y_test)),
            'test_control': int(len(y_test) - np.sum(y_test))
        }
    
    def save_processed_data(self, features: np.ndarray, labels: np.ndarray, 
                            subject_ids: list, feature_names: list, 
                            scaler, rfe, performance_metrics: dict) -> None:
        """Save processed data in format compatible with existing codebase."""
        
        # Save main arrays
        np.save(self.output_path / 'features.npy', features)
        np.save(self.output_path / 'labels.npy', labels)
        np.save(self.output_path / 'subject_ids.npy', np.array(subject_ids))
        
        # Save feature names
        with open(self.output_path / 'feature_names.txt', 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        
        # Save metadata with paper methodology info
        metadata = {
            'extraction_method': 'paper_methodology',
            'paper_reference': 'A Framework for Comparison and Interpretation of Machine Learning Classifiers to Predict Autism on the ABIDE Dataset',
            'feature_selection': 'RFE_with_Ridge_classifier',
            'n_subjects': int(len(subject_ids)),
            'n_features_original': int(rfe.n_features_in_),
            'n_features_selected': int(len(feature_names)),
            'target_features': self.target_features,
            'n_autism': int(np.sum(labels)),
            'n_control': int(len(labels) - np.sum(labels)),
            'baseline_performance': performance_metrics,
            'preprocessing_steps': [
                'FreeSurfer_stats_parsing',
                'missing_value_imputation_with_median',
                'standardization_zero_mean_unit_variance', 
                'recursive_feature_elimination_ridge_classifier'
            ],
            'feature_selection_target': 'RFE to select most informative features',
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist()
        }
        
        with open(self.output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Also save in .mat format for compatibility
        import scipy.io as sio
        sio.savemat(self.output_path / 'processed_data.mat', {
            'features': features,
            'labels': labels,
            'subject_ids': np.array(subject_ids)
        })
    
    def run_complete_extraction(self) -> None:
        """Run the complete sMRI feature extraction pipeline."""
        
        # Step 1: Download phenotypic data
        pheno_file = self.download_phenotypic_data()
        phenotype_df = pd.read_csv(pheno_file)
        
        # Step 2: Parse FreeSurfer features
        subject_features = self.parse_freesurfer_stats()
        
        # Step 3: Create feature matrix
        feature_matrix, fs_subject_ids, feature_names = self.create_feature_matrix(subject_features)
        
        # Step 4: Match with phenotypic data
        matched_features, labels, matched_subject_ids, matched_phenotype = self.match_with_phenotypic_data(
            feature_matrix, fs_subject_ids, phenotype_df
        )
        
        # Step 5: Apply paper's feature selection methodology
        selected_features, selected_feature_names, scaler, rfe = self.apply_paper_feature_selection(
            matched_features, labels, feature_names
        )
        
        # Step 6: Evaluate baseline performance
        performance_metrics = self.evaluate_baseline_performance(selected_features, labels)
        
        # Step 7: Save processed data
        self.save_processed_data(
            selected_features, labels, matched_subject_ids, 
            selected_feature_names, scaler, rfe, performance_metrics
        )
        
        print("sMRI extraction complete.")
        print(f"Final dataset: {selected_features.shape[0]} subjects x {selected_features.shape[1]} features")
        print(f"Baseline SVM accuracy: {performance_metrics['svm_accuracy']:.1%}")
        print(f"Data saved to: {self.output_path}")


def main():
    """Main function to run the extraction."""
    
    # Configuration
    FREESURFER_PATH = "data/freesurfer_stats" 
    OUTPUT_PATH = "processed_smri_data_improved"
    
    # Check if FreeSurfer data exists
    if not os.path.exists(FREESURFER_PATH):
        print(f"FreeSurfer data not found at {FREESURFER_PATH}")
        print("Please ensure the data is in the correct location.")
        return
    
    # Create extractor and run
    extractor = ImprovedSMRIExtractor(
        freesurfer_path=FREESURFER_PATH,
        output_path=OUTPUT_PATH
    )
    
    try:
        extractor.run_complete_extraction()
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 