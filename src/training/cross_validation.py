import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
import time

# Project imports
from config import get_config
from . import Trainer, set_seed
from .utils import create_multimodal_data_loaders
from utils.helpers import _run_multimodal_fold
from utils.optimal_configs import get_config_for_experiment

logger = logging.getLogger(__name__)


class CrossValidationFramework:
    """Cross-validation framework for neuroimaging experiments."""
    
    def __init__(self, device, data_loader, site_extractor):
        """Initialize CV framework with required components."""
        self.device = device
        self.data_loader = data_loader
        self.site_extractor = site_extractor
    
    def run_leave_site_out_cv(
        self,
        exp_name: str,
        exp_config: dict,
        matched_data: dict,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        output_dir: Path,
        seed: int,
        verbose: bool,
            early_stopping: bool = True,  # Enable early stopping for faster LSO
    patience: int = 15,           # Early stopping patience
    min_delta: float = 0.001      # Minimum improvement threshold
    ):
        """Run leave-site-out cross-validation for a single experiment."""
        # Implementation extracted from main script
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract site information
        if 'subject_ids' in matched_data:
            subject_ids = matched_data['subject_ids']
        else:
            n_subjects = len(matched_data['labels'])
            subject_ids = [f"subject_{i:05d}" for i in range(n_subjects)]
        
        config = get_config('cross_attention')
        phenotypic_file = str(config.phenotypic_file) if hasattr(config, 'phenotypic_file') else None
        
        site_labels, site_mapping, site_stats = self.site_extractor.extract_site_info(
            subject_ids, phenotypic_file
        )
        
        # Check for unknown sites
        unknown_sites = [site for site in site_mapping.keys() if site == 'UNKNOWN_SITE']
        if unknown_sites:
            raise ValueError(
                f"Cannot run leave-site-out CV with unknown sites. "
                f"Found {len([s for s in site_labels if s == 'UNKNOWN_SITE'])} subjects with unknown sites. "
                f"Leave-site-out CV requires real site information from phenotypic data."
            )
        
        n_sites = len(site_mapping)
        if n_sites < 3:
            raise ValueError(f"Need at least 3 sites for leave-site-out CV, found {n_sites}")
        
        # Log LSO setup for transparency
        if verbose:
            logger.info(f"LSO CV Setup: {n_sites} sites, early_stopping={early_stopping}")
            if early_stopping:
                logger.info(f"   Early stopping: patience={patience}, min_delta={min_delta}")
        
        # Prepare data arrays
        labels = matched_data['labels']
        features = None
        fmri_features = None
        smri_features = None
        
        # Handle modalities - Load tokenized data for tokenized experiments
        if verbose:
            logger.info(f"Loading data for {exp_config.get('modality', 'unknown')} modality")
            
        if exp_config['modality'] == 'fmri':
            # Check if this is a tokenized experiment
            if exp_config.get('type') == 'tokenized' and 'tokenization_type' in exp_config:
                # Load tokenized fMRI data
                tokenization_type = exp_config['tokenization_type']
                logger.info(f"Loading tokenized fMRI data for LSO with {tokenization_type} tokenization")
                
                try:
                    features, _ = self.data_loader.load_tokenized_fmri_data(
                        tokenization_type=tokenization_type, 
                        verbose=verbose
                    )
                    if verbose:
                        logger.info(f"Loaded tokenized fMRI data shape: {features.shape}")
                except Exception as e:
                    logger.error(f"Failed to load tokenized fMRI data: {e}")
                    logger.error(f"   Falling back to raw fMRI data")
                    features = matched_data['fmri_data']
                    if verbose:
                        logger.info(f"Using raw fMRI data shape: {features.shape}")
            else:
                features = matched_data['fmri_data']
                if verbose:
                    logger.info(f"Using raw fMRI data shape: {features.shape}")
        elif exp_config['modality'] == 'smri':
            # Check if this is a tokenized experiment
            if exp_config.get('type') == 'tokenized' and 'tokenization_type' in exp_config:
                # Load tokenized sMRI data
                tokenization_type = exp_config['tokenization_type']
                logger.info(f"Loading tokenized sMRI data for LSO with {tokenization_type} tokenization")
                
                try:
                    features, _ = self.data_loader.load_tokenized_smri_data(
                        tokenization_type=tokenization_type,
                        verbose=verbose
                    )
                    if verbose:
                        logger.info(f"Loaded tokenized sMRI data shape: {features.shape}")
                except Exception as e:
                    logger.error(f"Failed to load tokenized sMRI data: {e}")
                    logger.error(f"   Falling back to raw sMRI data")
                    features = matched_data['smri_data']
                    if verbose:
                        logger.info(f"Using raw sMRI data shape: {features.shape}")
            else:
                features = matched_data['smri_data']
                if verbose:
                    logger.info(f"Using raw sMRI data shape: {features.shape}")
        elif exp_config['modality'] in ['multimodal', 'multimodal_tokenized']:
            fmri_features = matched_data.get('fmri_data')
            smri_features = matched_data.get('smri_data')
        
        # Convert site labels to numpy array
        site_array = np.array(site_labels)
        logo = LeaveOneGroupOut()
        fold_results = []
        site_results = []
        
        # Track timing for performance analysis
        fold_times = []
        start_time = time.time()
        
        # Run CV
        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(labels, labels, site_array)):
            fold_start = time.time()
            test_sites = np.unique(site_array[test_idx])
            test_site = test_sites[0] if len(test_sites) == 1 else f"Mixed_{fold_idx}"
            
            if verbose:
                logger.info(f"   LSO Fold {fold_idx + 1}/{n_sites}: Testing site {test_site} ({len(test_idx)} subjects)")
            
            try:
                if exp_config['modality'] in ['multimodal', 'multimodal_tokenized']:
                    fold_result = _run_multimodal_fold(
                        fold=fold_idx,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        fmri_features=fmri_features,
                        smri_features=smri_features,
                        labels=labels,
                        model_class=exp_config['model_class'],
                        config=get_config('cross_attention'),
                        device=self.device,
                        verbose=False,
                        early_stopping=early_stopping,  # Pass early stopping
                        patience=patience,
                        min_delta=min_delta
                    )
                else:
                    fold_result = self._run_single_modality_fold(
                        fold_idx, train_idx, test_idx, features, labels,
                        exp_config, num_epochs, batch_size, learning_rate,
                        output_dir, seed, early_stopping, patience, min_delta
                    )
                
                fold_results.append(fold_result)
                site_results.append({
                    'test_site': test_site,
                    'test_accuracy': fold_result['test_accuracy'],
                    'test_balanced_accuracy': fold_result['test_balanced_accuracy'],
                    'test_auc': fold_result['test_auc'],
                    'n_test_subjects': len(test_idx),
                    'n_train_subjects': len(train_idx),
                    'fold_time': time.time() - fold_start,
                    'epochs_trained': fold_result.get('epochs_trained', num_epochs)  # Track actual epochs
                })
                
                fold_times.append(time.time() - fold_start)
                
                if verbose:
                    acc = fold_result['test_accuracy'] * 100
                    epochs = fold_result.get('epochs_trained', num_epochs)
                    logger.info(f"      Site {test_site}: {acc:.1f}% accuracy ({epochs} epochs, {fold_times[-1]:.1f}s)")
                
            except Exception as e:
                logger.warning(f"      Fold {fold_idx} (site {test_site}) failed: {e}")
                continue
        
        if not fold_results:
            raise RuntimeError("All leave-site-out folds failed")
        
        # Calculate results with statistics
        accuracies = [r['test_accuracy'] for r in fold_results]
        balanced_accuracies = [r['test_balanced_accuracy'] for r in fold_results]
        aucs = [r['test_auc'] for r in fold_results]
        
        total_time = time.time() - start_time
        avg_fold_time = np.mean(fold_times) if fold_times else 0
        avg_epochs = np.mean([sr['epochs_trained'] for sr in site_results])
        
        results = {
            'mean_accuracy': float(np.mean(accuracies)) * 100,
            'std_accuracy': float(np.std(accuracies)) * 100,
            'mean_balanced_accuracy': float(np.mean(balanced_accuracies)) * 100,
            'std_balanced_accuracy': float(np.std(balanced_accuracies)) * 100,
            'mean_auc': float(np.mean(aucs)),
            'std_auc': float(np.std(aucs)),
            'n_sites': len(site_mapping),
            'n_folds': len(fold_results),
            'site_results': site_results,
            'fold_results': fold_results,
            'cv_type': 'leave_site_out',
            # Performance tracking
            'total_time': total_time,
            'avg_fold_time': avg_fold_time,
            'avg_epochs_trained': avg_epochs,
            'early_stopping_used': early_stopping,
            'optimization_stats': {
                'time_per_site': fold_times,
                'epochs_per_site': [sr['epochs_trained'] for sr in site_results],
                'speedup_factor': num_epochs / avg_epochs if avg_epochs > 0 else 1.0
            }
        }
        
        if verbose:
            speedup = results['optimization_stats']['speedup_factor']
            logger.info(f"LSO Complete: {results['mean_accuracy']:.1f}% +- {results['std_accuracy']:.1f}% "
                       f"({total_time:.1f}s total, {speedup:.1f}x speedup)")
        
        return results
    
    def run_standard_cv(
        self,
        exp_config: dict,
        X,
        y: np.ndarray,
        num_folds: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        output_dir: Path,
        seed: int,
        verbose: bool,
        exp_name: str = None  # ADD exp_name parameter for config lookup
    ):
        """Run standard cross-validation for a single experiment."""
        from utils import run_cross_validation
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply good configurations for tokenized experiments
        if exp_name and ('tokenized' in exp_name):
            try:
                optimal_config = get_config_for_experiment(exp_name, exp_config)
                if verbose:
                                logger.info(f"Applying config for {exp_name}: {optimal_config}")
            
            # Override with good hyperparameters
                num_epochs = optimal_config.get('num_epochs', num_epochs)
                batch_size = optimal_config.get('batch_size', batch_size) 
                learning_rate = optimal_config.get('learning_rate', learning_rate)
                
                # Store model hyperparameters for later use
                exp_config['optimal_model_config'] = {
                    'd_model': optimal_config.get('d_model', 128),
                    'n_heads': optimal_config.get('n_heads', 4),
                    'n_layers': optimal_config.get('n_layers', 2),
                    'dropout': optimal_config.get('dropout', 0.2)
                }
                
                if verbose:
                    logger.info(f"Parameters: LR={learning_rate}, Batch={batch_size}, Epochs={num_epochs}")
                    logger.info(f"Model config: {exp_config['optimal_model_config']}")
                    
            except Exception as e:
                logger.warning(f"Could not load config for {exp_name}: {e}")
        
        if exp_config['modality'] in ['multimodal', 'multimodal_tokenized']:
            cv_results = self._run_multimodal_cv(
                exp_config['model_class'], X, y,
                num_folds, num_epochs, batch_size, learning_rate,
                output_dir, seed, verbose, exp_config
            )
        else:
            # Single-modality cross-validation
            if exp_config['modality'] in ['fmri', 'fmri_tokenized']:
                temp_config = get_config('fmri')
            elif exp_config['modality'] in ['smri', 'smri_tokenized']:
                temp_config = get_config('smri')
            else:
                temp_config = get_config('cross_attention')
                
            temp_config.num_folds = num_folds
            temp_config.num_epochs = num_epochs
            temp_config.batch_size = batch_size
            temp_config.learning_rate = learning_rate
            temp_config.output_dir = output_dir
            temp_config.seed = seed
            
            # Apply good model hyperparameters from testing
            if 'optimal_model_config' in exp_config:
                optimal_model_config = exp_config['optimal_model_config']
                temp_config.d_model = optimal_model_config.get('d_model', 128)
                temp_config.num_heads = optimal_model_config.get('n_heads', 4)
                temp_config.num_layers = optimal_model_config.get('n_layers', 2) 
                temp_config.dropout = optimal_model_config.get('dropout', 0.2)
                
                if verbose:
                    logger.info(f"Applied model config: d_model={temp_config.d_model}, "
                              f"heads={temp_config.num_heads}, layers={temp_config.num_layers}, dropout={temp_config.dropout}")
            
            # Optimize for sMRI
            if exp_config['modality'] in ['smri', 'smri_tokenized']:
                temp_config.use_class_weights = True
                temp_config.label_smoothing = 0.05
                temp_config.weight_decay = 1e-4
                temp_config.warmup_epochs = int(10)
            
            fold_results = run_cross_validation(
                features=X,
                labels=y,
                model_class=exp_config['model_class'],
                config=temp_config,
                experiment_type='single',
                verbose=verbose
            )
            
            # Convert to expected format
            accuracies = [r['test_accuracy'] for r in fold_results]
            balanced_accs = [r['test_balanced_accuracy'] for r in fold_results]
            aucs = [r['test_auc'] for r in fold_results]
            
            cv_results = {
                'fold_results': fold_results,
                'mean_accuracy': np.mean(accuracies) * 100,
                'std_accuracy': np.std(accuracies) * 100,
                'mean_balanced_accuracy': np.mean(balanced_accs) * 100,
                'std_balanced_accuracy': np.std(balanced_accs) * 100,
                'mean_auc': np.mean(aucs),
                'std_auc': np.std(aucs),
                'cv_type': 'standard'
            }
        
        return cv_results
    
    def _run_multimodal_cv(
        self,
        model_class,
        X: dict,
        y: np.ndarray,
        num_folds: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        save_dir: Path,
        seed: int,
        verbose: bool,
        exp_config: dict = None  # ADD exp_config parameter
    ):
        """Run cross-validation for multimodal models."""
        set_seed(seed)
        
        fmri_data = X['fmri']
        smri_data = X['smri']
        
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(fmri_data, y)):
            # Split data
            fmri_train, fmri_val = fmri_data[train_idx], fmri_data[val_idx]
            smri_train, smri_val = smri_data[train_idx], smri_data[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create model with correct parameter names for different model classes
            model_kwargs = {}
            
            # Different models expect different parameter names for input dimension
            if hasattr(model_class, '__name__'):
                model_name = model_class.__name__
                if 'SingleAtlasTransformer' in model_name:
                    model_kwargs['feat_dim'] = fmri_train.shape[1]  # fMRI transformer uses feat_dim
                elif 'CrossAttention' in model_name:
                    # Cross-attention models expect fmri_dim and smri_dim
                    model_kwargs['fmri_dim'] = fmri_train.shape[1]
                    model_kwargs['smri_dim'] = smri_train.shape[1]
                    model_kwargs['num_classes'] = 2
                    
                    # Only add parameters that the model actually accepts
                    import inspect
                    sig = inspect.signature(model_class.__init__)
                    if 'n_cross_layers' in sig.parameters:
                        model_kwargs['n_cross_layers'] = 3  # Default value
                    elif 'n_layers' in sig.parameters:
                        model_kwargs['n_layers'] = 3  # For MinimalCrossAttentionTransformer
                else:
                    model_kwargs['input_dim'] = fmri_train.shape[1]  # sMRI and others use input_dim
            else:
                model_kwargs['input_dim'] = fmri_train.shape[1]  # Default to input_dim
            
            # Apply optimal model configuration if available
            if exp_config and 'optimal_model_config' in exp_config:
                optimal_model_config = exp_config['optimal_model_config']
                model_kwargs.update({
                    'd_model': optimal_model_config.get('d_model', 128),
                    'n_heads': optimal_model_config.get('n_heads', 4),
                    'n_layers': optimal_model_config.get('n_layers', 2),
                    'dropout': optimal_model_config.get('dropout', 0.2)
                })
                
                if verbose and fold == 0:  # Only log once
                    logger.info(f"Creating multimodal model with config: {model_kwargs}")
            
            model = model_class(**model_kwargs).to(self.device)
            
            config = get_config('cross_attention')
            trainer = Trainer(model, self.device, config, model_type='multimodal')
            
            # Create data loaders
            train_loader, val_loader = create_multimodal_data_loaders(
                fmri_train, smri_train, y_train,
                fmri_val, smri_val, y_val,
                batch_size=int(batch_size),
                augment_train=True
            )
            
            # Train model
            history = trainer.fit(
                train_loader, val_loader,
                num_epochs=num_epochs,
                checkpoint_path=save_dir / f'best_model_fold{fold}.pt',
                y_train=y_train,
                save_model_checkpoints=True  # ENABLE for proper best model evaluation
            )
            
            # Evaluate
            val_test_loader, _ = create_multimodal_data_loaders(
                fmri_val, smri_val, y_val,
                fmri_val, smri_val, y_val,
                batch_size=int(batch_size),
                augment_train=False
            )
            
            test_metrics = trainer.evaluate_final(val_test_loader, save_dir / f'best_model_fold{fold}.pt')
            
            fold_results.append({
                'fold': fold,
                'test_accuracy': test_metrics['accuracy'],
                'test_balanced_accuracy': test_metrics['balanced_accuracy'],
                'test_auc': test_metrics['auc'],
                'history': history,
                'targets': test_metrics['targets'],
                'predictions': test_metrics['predictions'],
                'probabilities': test_metrics['probabilities']
            })
        
        # Calculate aggregate metrics
        accuracies = [r['test_accuracy'] for r in fold_results]
        balanced_accs = [r['test_balanced_accuracy'] for r in fold_results]
        aucs = [r['test_auc'] for r in fold_results]
        
        return {
            'fold_results': fold_results,
            'mean_accuracy': np.mean(accuracies) * 100,
            'std_accuracy': np.std(accuracies) * 100,
            'mean_balanced_accuracy': np.mean(balanced_accs) * 100,
            'std_balanced_accuracy': np.std(balanced_accs) * 100,
            'mean_auc': np.mean(aucs),
            'std_auc': np.std(aucs),
            'cv_type': 'multimodal'
        }
    
    def _run_single_modality_fold(
        self,
        fold_idx: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        exp_config: dict,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        output_dir: Path,
        seed: int,
        early_stopping: bool = True,
        patience: int = 15,
        min_delta: float = 0.001
    ):
        """Run a single fold for single-modality experiments with early stopping support."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
        
        set_seed(seed)
        
        # Split data
        X_train_fold, X_test = features[train_idx], features[test_idx]
        y_train_fold, y_test = labels[train_idx], labels[test_idx]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_fold, y_train_fold,
            test_size=0.2,
            stratify=y_train_fold,
            random_state=seed + fold_idx
        )
        
        # Preprocessing for sMRI
        if exp_config['modality'] == 'smri':
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        
        # Convert to tensors and create loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model with correct parameter names for different model classes
        model_kwargs = {}
        
        # Different models expect different parameter names for input dimension
        if hasattr(exp_config['model_class'], '__name__'):
            model_name = exp_config['model_class'].__name__
            if 'SingleAtlasTransformer' in model_name or 'FMRI' in model_name:
                model_kwargs['feat_dim'] = X_train.shape[1]  # fMRI transformers use feat_dim
            else:
                model_kwargs['input_dim'] = X_train.shape[1]  # sMRI and others use input_dim
        else:
            model_kwargs['input_dim'] = X_train.shape[1]  # Default to input_dim
        
        # Apply optimal model configuration if available
        if 'optimal_model_config' in exp_config:
            optimal_model_config = exp_config['optimal_model_config']
            model_kwargs.update({
                'd_model': optimal_model_config.get('d_model', 128),
                'n_heads': optimal_model_config.get('n_heads', 4),
                'n_layers': optimal_model_config.get('n_layers', 2),
                'dropout': optimal_model_config.get('dropout', 0.2)
            })
            
            # Add layer_dropout for specific models that support it
            if hasattr(exp_config['model_class'], '__name__') and 'FeatureType' in exp_config['model_class'].__name__:
                model_kwargs['layer_dropout'] = optimal_model_config.get('dropout', 0.2) * 0.5
        
        model = exp_config['model_class'](**model_kwargs).to(self.device)
        config = get_config('cross_attention')
        trainer = Trainer(model, self.device, config)
        
        history = trainer.fit(train_loader, val_loader, num_epochs=num_epochs, checkpoint_path=output_dir / f'best_model_fold{fold_idx}.pt', y_train=y_train, save_model_checkpoints=True)
        test_metrics = trainer.evaluate_final(test_loader, output_dir / f'best_model_fold{fold_idx}.pt')
        
        return {
            'test_accuracy': test_metrics['accuracy'],
            'test_balanced_accuracy': test_metrics['balanced_accuracy'], 
            'test_auc': test_metrics['auc'],
            'history': history,
            'targets': test_metrics['targets'],
            'predictions': test_metrics['predictions'],
            'probabilities': test_metrics['probabilities']
        } 