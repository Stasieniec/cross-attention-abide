import os
import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from tqdm import tqdm

from nilearn.connectome import ConnectivityMeasure


class FMRIDataProcessor:
    """Process fMRI data from ROI time series files."""

    def __init__(self, roi_dir: Path, pheno_file: Path, n_rois: int = 200):
        """
        Initialize fMRI data processor.
        
        Args:
            roi_dir: Directory containing ROI time series files
            pheno_file: Path to phenotypic data file
            n_rois: Number of ROIs in the atlas
        """
        self.roi_dir = roi_dir
        self.pheno_file = pheno_file
        self.n_rois = n_rois
        self.connectivity_measure = ConnectivityMeasure(kind='correlation')

    def load_roi_files(self) -> Tuple[List[str], List[str]]:
        """Load ROI file paths and extract subject IDs."""
        roi_files = sorted(glob.glob(str(self.roi_dir / '*.1D')))
        subject_ids = [os.path.basename(f).split('.')[0] for f in roi_files]
        return roi_files, subject_ids

    def compute_connectivity_matrix(self, roi_file: str) -> Optional[np.ndarray]:
        """
        Compute correlation-based connectivity matrix from ROI time series.
        
        Args:
            roi_file: Path to ROI time series file
            
        Returns:
            Flattened upper triangle of connectivity matrix, or None if error
        """
        try:
            # Load time series
            ts = np.loadtxt(roi_file, usecols=range(self.n_rois))

            # Check data validity
            if not np.issubdtype(ts.dtype, np.number) or np.any(np.isnan(ts)):
                return None

            # Compute correlation matrix
            fc_matrix = self.connectivity_measure.fit_transform([ts])[0]

            # Extract upper triangle (excluding diagonal)
            upper_idx = np.triu_indices(self.n_rois, k=1)
            fc_vector = fc_matrix[upper_idx]

            return fc_vector

        except Exception as e:
            print(f"Error processing {roi_file}: {e}")
            return None

    def extract_subject_id(self, filename: str) -> str:
        """Extract numeric subject ID from filename."""
        # Pattern: extract numeric ID from filenames like "0050002_rois_cc200.1D"
        match = re.search(r'(\d+)_rois_cc200', filename)
        if match:
            # Remove leading zeros to match phenotypic file
            return str(int(match.group(1)))
        return filename.split('.')[0]

    def load_phenotypic_data(self) -> pd.DataFrame:
        """Load and preprocess phenotypic data."""
        pheno = pd.read_csv(self.pheno_file)
        # Ensure SUB_ID is string for matching
        pheno['SUB_ID'] = pheno['SUB_ID'].astype(str)
        return pheno

    def process_all_subjects(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Process all subjects and return connectivity matrices with labels.

        Args:
            verbose: Whether to show progress bars and print statistics
            
        Returns:
            fc_matrices: Array of flattened connectivity matrices
            labels: Array of labels (0=control, 1=ASD)
            subject_ids: List of matched subject IDs
            skipped_ids: List of skipped subject IDs
        """
        roi_files, file_subject_ids = self.load_roi_files()
        pheno = self.load_phenotypic_data()

        fc_matrices = []
        labels = []
        subject_ids = []
        skipped_ids = []

        # Create mapping from phenotypic data
        pheno_dict = {
            str(row['SUB_ID']): row['DX_GROUP']
            for _, row in pheno.iterrows()
            if row['DX_GROUP'] in [1, 2]
        }

        # Process each subject
        iterator = tqdm(roi_files, desc="Computing FC matrices") if verbose else roi_files

        for roi_file, file_id in zip(iterator, file_subject_ids):
            # Extract subject ID
            sub_id = self.extract_subject_id(file_id)

            # Check if subject in phenotypic data
            if sub_id not in pheno_dict:
                skipped_ids.append(sub_id)
                continue

            # Compute connectivity matrix
            fc_vector = self.compute_connectivity_matrix(roi_file)
            if fc_vector is None:
                skipped_ids.append(sub_id)
                continue

            # Add to results
            fc_matrices.append(fc_vector)
            # Convert labels: DX_GROUP 1=ASD, 2=Control → 0=Control, 1=ASD
            labels.append(0 if pheno_dict[sub_id] == 2 else 1)
            subject_ids.append(sub_id)

        if verbose:
            print(f"\n✅ Processed {len(subject_ids)} subjects successfully")
            print(f"❌ Skipped {len(skipped_ids)} subjects")
            print(f"🧠 ASD: {np.sum(labels)}, Control: {len(labels) - np.sum(labels)}")

        return np.array(fc_matrices), np.array(labels), subject_ids, skipped_ids
    
    def load_all_subjects(self) -> Dict:
        """
        Load all fMRI subjects and return as dictionary (for cross-modal matching).
        
        Returns:
            Dictionary mapping subject IDs to features and labels
        """
        roi_files = sorted(glob.glob(str(self.roi_dir / '*.1D')))
        pheno = self.load_phenotypic_data()

        pheno_dict = {
            str(row['SUB_ID']): row['DX_GROUP']
            for _, row in pheno.iterrows()
            if row['DX_GROUP'] in [1, 2]
        }

        fmri_data = {}

        for roi_file in tqdm(roi_files, desc="Loading fMRI data"):
            sub_id = self.extract_subject_id(os.path.basename(roi_file))

            if sub_id not in pheno_dict:
                continue

            fc_vector = self.compute_connectivity_matrix(roi_file)
            if fc_vector is None:
                continue

            fmri_data[sub_id] = {
                'features': fc_vector,
                'label': 0 if pheno_dict[sub_id] == 2 else 1  # 0=Control, 1=ASD
            }

        return fmri_data 