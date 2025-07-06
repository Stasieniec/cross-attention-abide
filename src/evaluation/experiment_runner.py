import time
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Coordinates execution of all thesis experiments."""
    
    def __init__(self, thesis_experiments):
        """Initialize with reference to main ThesisExperiments instance."""
        self.experiments = thesis_experiments
        self.device = thesis_experiments.device
        self.data_loader = thesis_experiments.data_loader
        self.cv_framework = thesis_experiments.cv_framework
    
    def run_all(self, **kwargs):
        """Run all experiments with multiple seeds for reliable results."""
        return self._run_experiments_by_type(
            experiment_types=['baseline', 'cross_attention', 'tokenized', 'tokenized_cross_attention'],
            **kwargs
        )
    
    def run_baselines_only(self, **kwargs):
        """Run only baseline experiments (fMRI, sMRI)."""
        return self._run_experiments_by_type(
            experiment_types=['baseline'],
            **kwargs
        )
    
    def run_cross_attention_only(self, **kwargs):
        """Run only cross-attention experiments."""
        return self._run_experiments_by_type(
            experiment_types=['cross_attention'],
            **kwargs
        )
    
    def run_tokenized_only(self, **kwargs):
        """Run only tokenized experiments (individual + cross-attention)."""
        return self._run_experiments_by_type(
            experiment_types=['tokenized', 'tokenized_cross_attention'],
            **kwargs
        )
    
    def run_quick_tokenized(self, **kwargs):
        """Run quick tokenized experiments with reduced parameters."""
        kwargs.setdefault('num_folds', 3)
        kwargs.setdefault('num_epochs', 30)
        kwargs.setdefault('include_leave_site_out', False)
        kwargs.setdefault('num_seeds', 1)  # Single seed for quick testing
        
        return self._run_experiments_by_type(
            experiment_types=['tokenized', 'tokenized_cross_attention'],
            **kwargs
        )
    
    def quick_test(self, experiments: List[str] = None, **kwargs):
        """Quick test of selected experiments with minimal training."""
        kwargs.setdefault('num_folds', 2)
        kwargs.setdefault('num_epochs', 10)
        kwargs.setdefault('include_leave_site_out', False)
        kwargs.setdefault('num_seeds', 1)
        
        if experiments is None:
            experiments = ['fmri_baseline', 'smri_baseline', 'cross_attention_basic']
        
        return self._run_specific_experiments(experiments, **kwargs)
    
    def _run_experiments_by_type(self, experiment_types: List[str], **kwargs):
        """Run experiments filtered by type."""
        selected_experiments = [
            exp_name for exp_name, exp_config in self.experiments.experiments.items()
            if exp_config['type'] in experiment_types
        ]
        
        return self._run_specific_experiments(selected_experiments, **kwargs)
    
    def _run_specific_experiments(self, experiment_names: List[str], **kwargs):
        """Run specific experiments by name with multi-seed support."""
        all_results = {}
        
        # Extract parameters with defaults
        num_folds = kwargs.get('num_folds', 5)
        num_epochs = kwargs.get('num_epochs', 30)
        batch_size = kwargs.get('batch_size', 16)
        learning_rate = kwargs.get('learning_rate', 0.001)
        verbose = kwargs.get('verbose', True)
        include_leave_site_out = kwargs.get('include_leave_site_out', True)
        # Handle None output_dir properly
        output_dir = kwargs.get('output_dir') or '/content/drive/MyDrive/thesis_experiments_default'
        
        # Multiple random seeds for reliable results
        num_seeds = kwargs.get('num_seeds', 3)  # Default to 3 seeds for robustness
        base_seed = kwargs.get('seed', 42)
        seeds = [base_seed + i * 100 for i in range(num_seeds)]  # Generate diverse seeds
        
        # Parallel leave-site-out option
        parallel_lso = kwargs.get('parallel_lso', True)  # Enable by default
        lso_early_stopping = kwargs.get('lso_early_stopping', True)  # Enable early stopping for LSO
        
        # LSO-only mode option
        skip_standard_cv = kwargs.get('skip_standard_cv', False)  # Skip standard CV and run only LSO
        
        if verbose:
            logger.info(f"Running {len(experiment_names)} experiments with:")
            logger.info(f"   Folds: {num_folds}, Epochs: {num_epochs}, Batch: {batch_size}, LR: {learning_rate}")
            logger.info(f"   Seeds: {num_seeds} seeds ({seeds})")
            logger.info(f"   Output: {output_dir}")
            logger.info(f"   Leave-site-out: {include_leave_site_out} (parallel: {parallel_lso})")
            if skip_standard_cv:
                logger.info(f"   Mode: LSO-only (skipping standard CV)")
            else:
                logger.info(f"   Mode: Full (standard CV + LSO)")
        
        # Load matched data once for all experiments
        matched_data = self.data_loader.load_matched_data(verbose=verbose)
        
        for exp_name in experiment_names:
            if exp_name not in self.experiments.experiments:
                logger.warning(f"Unknown experiment: {exp_name}, skipping")
                continue
                
            exp_config = self.experiments.experiments[exp_name]
            
            if verbose:
                logger.info(f"\nRunning: {exp_config['name']} with {num_seeds} seeds")
            
            # Run experiment with multiple seeds
            exp_results = self._run_experiment_with_multiple_seeds(
                exp_name=exp_name,
                exp_config=exp_config,
                matched_data=matched_data,
                seeds=seeds,
                num_folds=num_folds,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                output_dir=output_dir,
                verbose=verbose,
                include_leave_site_out=include_leave_site_out,
                parallel_lso=parallel_lso,
                lso_early_stopping=lso_early_stopping,
                skip_standard_cv=skip_standard_cv
            )
            
            # Store aggregated results
            all_results[exp_name] = exp_results
            
            if verbose and exp_results.get('aggregated_cv'):
                cv_acc = exp_results['aggregated_cv'].get('mean_accuracy', 0)
                cv_std = exp_results['aggregated_cv'].get('std_accuracy', 0)
                logger.info(f"{exp_config['name']} (CV): {cv_acc:.1f}% +- {cv_std:.1f}%")
                
                if exp_results.get('aggregated_lso'):
                    lso_acc = exp_results['aggregated_lso'].get('mean_accuracy', 0)
                    lso_std = exp_results['aggregated_lso'].get('std_accuracy', 0)
                    logger.info(f"{exp_config['name']} (LSO): {lso_acc:.1f}% +- {lso_std:.1f}%")
        
        return all_results
    
    def _run_experiment_with_multiple_seeds(
        self,
        exp_name: str,
        exp_config: dict,
        matched_data: dict,
        seeds: List[int],
        num_folds: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        output_dir: str,
        verbose: bool,
        include_leave_site_out: bool,
        parallel_lso: bool = True,
        lso_early_stopping: bool = True,
        skip_standard_cv: bool = False  # NEW: Option to skip standard CV and run only LSO
    ) -> Dict[str, Any]:
        """Run experiment with multiple seeds for reliable statistics."""
        
        seed_results = {
            'cv_results': [],
            'lso_results': []  # This will contain only one result now
        }
        
        # Run standard CV with multiple seeds (unless skipped)
        if not skip_standard_cv:
            for seed in seeds:
                try:
                    # Run standard CV with this seed
                    cv_result = self._run_single_experiment_cv(
                        exp_name=exp_name,
                        exp_config=exp_config,
                        matched_data=matched_data,
                        num_folds=num_folds,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        seed=seed,
                        verbose=verbose
                    )
                    
                    seed_results['cv_results'].append(cv_result)
                    
                except Exception as e:
                    logger.warning(f"Seed {seed} failed for {exp_name}: {e}")
                    continue
        else:
            if verbose:
                logger.info(f"Skipping standard CV for {exp_name} (LSO only mode)")
        
        if include_leave_site_out:
            try:
                lso_result = self._run_single_experiment_lso(
                    exp_name=exp_name,
                    exp_config=exp_config,
                    matched_data=matched_data,
                    num_epochs=num_epochs if not lso_early_stopping else min(num_epochs, 50),  # Limit epochs for LSO
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    output_dir=Path(f"{output_dir}/leave_site_out/{exp_name}"),
                    seed=seeds[0],  # Use first seed for any random initialization, but LSO itself is deterministic
                    verbose=verbose,
                    early_stopping=lso_early_stopping
                )
                
                seed_results['lso_results'].append(lso_result)
                
            except Exception as e:
                logger.warning(f"LSO failed for {exp_name}: {e}")
        
        # Aggregate results across seeds
        return self._aggregate_multi_seed_results(seed_results, exp_config)
    
    def _aggregate_multi_seed_results(self, seed_results: Dict, exp_config: dict) -> Dict[str, Any]:
        """Aggregate results from multiple seeds into reliable statistics."""
        
        result = {
            'name': exp_config['name'],
            'type': exp_config['type'],
            'num_seeds': len(seed_results['cv_results'])
        }
        
        # Aggregate CV results
        if seed_results['cv_results']:
            cv_accuracies = [r.get('mean_accuracy', 0) for r in seed_results['cv_results']]
            cv_balanced_accs = [r.get('mean_balanced_accuracy', 0) for r in seed_results['cv_results']]
            cv_aucs = [r.get('mean_auc', 0) for r in seed_results['cv_results']]
            
            result['aggregated_cv'] = {
                'mean_accuracy': float(np.mean(cv_accuracies)),
                'std_accuracy': float(np.std(cv_accuracies)),
                'mean_balanced_accuracy': float(np.mean(cv_balanced_accs)),
                'std_balanced_accuracy': float(np.std(cv_balanced_accs)),
                'mean_auc': float(np.mean(cv_aucs)),
                'std_auc': float(np.std(cv_aucs)),
                'cv_type': 'aggregated_standard',
                'individual_results': seed_results['cv_results']
            }
        
        # Aggregate LSO results
        if seed_results['lso_results']:
            # LSO now runs only once, so no aggregation needed
            # Just use the single LSO result directly
            lso_result = seed_results['lso_results'][0]
            
            result['aggregated_lso'] = {
                'mean_accuracy': lso_result.get('mean_accuracy', 0),
                'std_accuracy': lso_result.get('std_accuracy', 0),
                'mean_balanced_accuracy': lso_result.get('mean_balanced_accuracy', 0),
                'std_balanced_accuracy': lso_result.get('std_balanced_accuracy', 0),
                'mean_auc': lso_result.get('mean_auc', 0),
                'std_auc': lso_result.get('std_auc', 0),
                'n_sites': lso_result.get('n_sites', 0),
                'cv_type': 'leave_site_out',
                'individual_results': [lso_result],  # Keep as list for compatibility
                'optimization_stats': lso_result.get('optimization_stats', {}),
                'total_time': lso_result.get('total_time', 0),
                'avg_epochs_trained': lso_result.get('avg_epochs_trained', 0)
            }
        
        return result
    
    def _run_single_experiment_cv(
        self,
        exp_name: str,
        exp_config: dict,
        matched_data: dict,
        num_folds: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        seed: int,
        verbose: bool
    ) -> Dict[str, Any]:
        """Run standard cross-validation for a single experiment with one seed."""
        
        if exp_config['type'] == 'baseline':
            if 'fmri' in exp_name:
                return self._run_fmri_experiment(
                    matched_data, num_folds, num_epochs, 
                    batch_size, learning_rate, verbose, seed
                )
            elif 'smri' in exp_name:
                return self._run_smri_experiment(
                    matched_data, num_folds, num_epochs, 
                    batch_size, learning_rate, verbose, seed
                )
            else:
                raise ValueError(f"Unknown baseline type: {exp_name}")
                
        elif exp_config['type'] == 'cross_attention':
            return self._run_cross_attention_experiment(
                matched_data, num_folds, num_epochs, 
                batch_size, learning_rate, verbose, exp_config, seed
            )
        elif exp_config['type'] == 'cross_attention_minimal':
            return self._run_minimal_cross_attention_experiment(
                matched_data, num_folds, num_epochs, 
                batch_size, learning_rate, verbose, exp_config, seed
            )
        elif exp_config['type'] == 'tokenized_cross_attention':
            return self._run_tokenized_cross_attention_experiment(
                matched_data, num_folds, num_epochs, 
                batch_size, learning_rate, verbose, exp_config, exp_name, seed
            )
        elif exp_config['type'] == 'tokenized':
            if exp_config['modality'] == 'fmri':
                return self._run_tokenized_fmri_experiment(
                    matched_data, num_folds, num_epochs, 
                    batch_size, learning_rate, verbose, exp_config, exp_name, seed
                )
            elif exp_config['modality'] == 'smri':
                return self._run_tokenized_smri_experiment(
                    matched_data, num_folds, num_epochs, 
                    batch_size, learning_rate, verbose, exp_config, exp_name, seed
                )
            else:
                raise ValueError(f"Unknown tokenized modality: {exp_config['modality']}")
        else:
            raise ValueError(f"Unknown experiment type: {exp_config['type']}")
    
    def _run_single_experiment_lso(
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
        early_stopping: bool = True
    ) -> Dict[str, Any]:
        """Run leave-site-out cross-validation."""
        
        # Use the same experiment routing logic as standard CV
        # to ensure proper model_class is set for baseline experiments
        
        if exp_config['type'] == 'baseline':
            if 'fmri' in exp_name:
                from models import SingleAtlasTransformer
                lso_exp_config = {
                    'name': 'fMRI Baseline',
                    'type': 'baseline',
                    'modality': 'fmri',
                    'model_class': SingleAtlasTransformer
                }
            elif 'smri' in exp_name:
                from models.enhanced_smri import EnhancedSMRITransformer
                lso_exp_config = {
                    'name': 'sMRI Baseline',
                    'type': 'baseline',
                    'modality': 'smri',
                    'model_class': EnhancedSMRITransformer
                }
            else:
                raise ValueError(f"Unknown baseline type: {exp_name}")
        
        elif exp_config['type'] == 'cross_attention':
            from models.minimal_improved_cross_attention import MinimalCrossAttentionTransformer
            lso_exp_config = {
                'name': exp_config.get('name', 'Cross-Attention Basic'),
                'type': 'cross_attention',
                'modality': 'multimodal',
                'model_class': MinimalCrossAttentionTransformer
            }
        
        elif exp_config['type'] == 'tokenized_cross_attention':
            from models.tokenized_models import TokenizedCrossAttentionTransformer
            lso_exp_config = {
                'name': exp_config.get('name', 'Tokenized Cross-Attention'),
                'type': 'tokenized_cross_attention',
                'modality': 'multimodal_tokenized',
                'model_class': TokenizedCrossAttentionTransformer,
                'fmri_tokenization_type': exp_config.get('fmri_tokenization', 'full_connectivity'),
                'smri_tokenization_type': exp_config.get('smri_tokenization', 'feature_type')
            }
        
        elif exp_config['type'] == 'tokenized':
            # For tokenized experiments, need to determine model class based on modality and tokenization type
            if exp_config['modality'] == 'fmri':
                tokenization_type = exp_config.get('tokenization_type', 'network_based')
                
                # Select model class based on tokenization type (same logic as standard CV)
                if tokenization_type == 'full_connectivity':
                    from models.tokenized_models import FMRIFullConnectivityTransformer
                    model_class = FMRIFullConnectivityTransformer
                elif tokenization_type == 'roi_connectivity':
                    from models.tokenized_models import FMRIROIConnectivityTransformer
                    model_class = FMRIROIConnectivityTransformer
                else:
                    # Default to full connectivity for unknown tokenization types
                    from models.tokenized_models import FMRIFullConnectivityTransformer
                    model_class = FMRIFullConnectivityTransformer
                
                lso_exp_config = {
                    'name': f'fMRI {tokenization_type} Tokenized',
                    'type': 'tokenized',
                    'modality': 'fmri',
                    'model_class': model_class,
                    'tokenization_type': tokenization_type
                }
                
            elif exp_config['modality'] == 'smri':
                tokenization_type = exp_config.get('tokenization_type', 'feature_type')
                
                # Select model class based on tokenization type (same logic as standard CV)
                if tokenization_type == 'brain_network':
                    from models.tokenized_models import SMRIBrainNetworkTokenizedTransformer
                    model_class = SMRIBrainNetworkTokenizedTransformer
                elif tokenization_type == 'feature_type':
                    from models.tokenized_models import SMRIFeatureTypeTokenizedTransformer
                    model_class = SMRIFeatureTypeTokenizedTransformer
                else:
                    # Default to feature type for unknown tokenization types
                    from models.tokenized_models import SMRIFeatureTypeTokenizedTransformer
                    model_class = SMRIFeatureTypeTokenizedTransformer
                
                lso_exp_config = {
                    'name': f'sMRI {tokenization_type} Tokenized',
                    'type': 'tokenized',
                    'modality': 'smri',
                    'model_class': model_class,
                    'tokenization_type': tokenization_type
                }
            else:
                # For other tokenized modalities, use the original config
                lso_exp_config = exp_config.copy()
        
        else:
            # For other experiment types, use the original config
            lso_exp_config = exp_config.copy()
        
        return self.cv_framework.run_leave_site_out_cv(
            exp_name=exp_name,
            exp_config=lso_exp_config,  # Use the properly configured exp_config
            matched_data=matched_data,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=output_dir,
            seed=seed,
            verbose=verbose,
            early_stopping=early_stopping  # Pass early stopping flag
        )
    
    def _run_fmri_experiment(self, matched_data, num_folds, num_epochs, batch_size, learning_rate, verbose, seed):
        """Run fMRI baseline experiment."""
        from models import SingleAtlasTransformer
        exp_config = {
            'name': 'fMRI Baseline',
            'type': 'baseline',
            'modality': 'fmri',
            'model_class': SingleAtlasTransformer
        }
        return self.cv_framework.run_standard_cv(
            exp_config=exp_config,
            X=matched_data['fmri_data'],
            y=matched_data['labels'],
            num_folds=num_folds,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=Path('/tmp/fmri_baseline'),
            seed=seed,
            verbose=verbose
        )
    
    def _run_smri_experiment(self, matched_data, num_folds, num_epochs, batch_size, learning_rate, verbose, seed):
        """Run sMRI baseline experiment."""
        from models.enhanced_smri import EnhancedSMRITransformer
        exp_config = {
            'name': 'sMRI Baseline',
            'type': 'baseline',
            'modality': 'smri',
            'model_class': EnhancedSMRITransformer
        }
        return self.cv_framework.run_standard_cv(
            exp_config=exp_config,
            X=matched_data['smri_data'],
            y=matched_data['labels'],
            num_folds=num_folds,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=Path('/tmp/smri_baseline'),
            seed=seed,
            verbose=verbose
        )
    
    def _run_cross_attention_experiment(self, matched_data, num_folds, num_epochs, batch_size, learning_rate, verbose, exp_config, seed):
        """Run cross-attention experiment with the improved minimal cross-attention model."""
        # USING IMPROVED MINIMAL CROSS-ATTENTION: Use the new minimal improved model
        try:
            from models.minimal_improved_cross_attention import MinimalCrossAttentionTransformer
        except ImportError:
            raise ImportError("MinimalCrossAttentionTransformer model not found. Check models/minimal_improved_cross_attention.py")
        
        # Use optimized cross-attention params
        optimized_params = self._get_optimized_params_for_tokenization('cross_attention_basic', 'cross_attention')
        
        # Override with optimized parameters if defaults were used
        if learning_rate == 0.001:  # Default learning rate
            learning_rate = optimized_params['learning_rate']
        if batch_size == 16:  # Default batch size
            batch_size = optimized_params['batch_size']
        if num_epochs == 30:  # Default epochs
            num_epochs = optimized_params['num_epochs']
        
        if verbose:
            logger.info(f"Running cross-attention with parameters:")
            logger.info(f"LR={learning_rate}, Batch={batch_size}, Epochs={num_epochs}")
            logger.info(f"fMRI: {matched_data['fmri_data'].shape}, sMRI: {matched_data['smri_data'].shape}")
            logger.info("Using MinimalCrossAttentionTransformer with single-token processing")
        
        # Create proper experiment configuration
        full_exp_config = {
            'name': exp_config.get('name', 'Cross-Attention Basic'),
            'type': 'cross_attention',
            'modality': 'multimodal',  # Set correct modality for multimodal handling
            'model_class': MinimalCrossAttentionTransformer
        }
        
        # Apply any good configurations if available
        if 'optimal_model_config' in exp_config:
            full_exp_config['optimal_model_config'] = exp_config['optimal_model_config']
        
        X = {
            'fmri': matched_data['fmri_data'],
            'smri': matched_data['smri_data']
        }
        
        return self.cv_framework.run_standard_cv(
            exp_config=full_exp_config,
            X=X,
            y=matched_data['labels'],
            num_folds=num_folds,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=Path('/tmp/cross_attention'),
            seed=seed,
            verbose=verbose
        )
    
    def _run_minimal_cross_attention_experiment(self, matched_data, num_folds, num_epochs, batch_size, learning_rate, verbose, exp_config, seed):
        """Run MINIMAL cross-attention experiment with baseline-equivalent processing and optimized parameters."""
        # Use the baseline cross-attention model
        try:
            from models.minimal_improved_cross_attention import MinimalCrossAttentionTransformer
        except ImportError:
            raise ImportError("MinimalCrossAttentionTransformer model not found. Check models/minimal_improved_cross_attention.py")
        
        # Use optimized minimal cross-attention params
        optimized_params = self._get_optimized_params_for_tokenization('cross_attention_minimal', 'cross_attention')
        
        # Override with optimized parameters if defaults were used
        if learning_rate == 0.001:  # Default learning rate
            learning_rate = optimized_params['learning_rate']
        if batch_size == 16:  # Default batch size
            batch_size = optimized_params['batch_size']
        if num_epochs == 30:  # Default epochs
            num_epochs = optimized_params['num_epochs']
        
        if verbose:
            logger.info(f"Running minimal cross-attention with parameters:")
            logger.info(f"LR={learning_rate}, Batch={batch_size}, Epochs={num_epochs}")
            logger.info(f"fMRI: {matched_data['fmri_data'].shape}, sMRI: {matched_data['smri_data'].shape}")
            logger.info("This uses single-token processing like individual baselines")
        
        # Create proper experiment configuration
        full_exp_config = {
            'name': exp_config.get('name', 'Cross-Attention Minimal (TRUE Baseline)'),
            'type': 'cross_attention_minimal',
            'modality': 'multimodal',
            'model_class': MinimalCrossAttentionTransformer
        }
        
        X = {
            'fmri': matched_data['fmri_data'],
            'smri': matched_data['smri_data']
        }
        
        return self.cv_framework.run_standard_cv(
            exp_config=full_exp_config,
            X=X,
            y=matched_data['labels'],
            num_folds=num_folds,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=Path('/tmp/cross_attention_minimal'),
            seed=seed,
            verbose=verbose
        )
    
    def _run_tokenized_fmri_experiment(self, matched_data, num_folds, num_epochs, batch_size, learning_rate, verbose, exp_config, exp_name, seed):
        """Run tokenized fMRI experiment with proper tokenization type and optimized parameters."""
        # Get tokenization type from experiment config
        tokenization_type = exp_config.get('tokenization_type', 'network_based')
        
        # Use optimized parameters per tokenization type
        optimized_params = self._get_optimized_params_for_tokenization(tokenization_type, 'single_modal')
        
        # Override with optimized parameters if defaults were used
        if learning_rate == 0.001:  # Default learning rate
            learning_rate = optimized_params['learning_rate']
        if batch_size == 16:  # Default batch size
            batch_size = optimized_params['batch_size']
        if num_epochs == 30:  # Default epochs
            num_epochs = optimized_params['num_epochs']
        
        if verbose:
            logger.info(f"Loading tokenized fMRI data with {tokenization_type} tokenization...")
            logger.info(f"Parameters: LR={learning_rate}, Batch={batch_size}, Epochs={num_epochs}")
        
        # Load tokenized data with proper tokenization type
        fmri_tokens, labels = self.data_loader.load_tokenized_fmri_data(
            tokenization_type=tokenization_type, 
            verbose=verbose
        )
        
        # Verify subject consistency
        if len(labels) != matched_data['n_subjects']:
            raise ValueError(
                f"Subject count mismatch! "
                f"Matched data has {matched_data['n_subjects']} subjects, "
                f"but tokenized data has {len(labels)} subjects. "
                f"Tokenized experiments must use the same subjects as regular experiments."
            )
        
        # Create tokenized data in same format as matched_data for consistency
        tokenized_matched_data = {
            'fmri_data': fmri_tokens,
            'smri_data': matched_data['smri_data'],  # Keep original sMRI for single-modal experiment
            'labels': labels,
            'subject_ids': matched_data.get('subject_ids'),
            'n_subjects': len(labels),
            'fmri_dim': fmri_tokens.shape[1],
            'smri_dim': matched_data['smri_dim']
        }
        
        # Select model based on tokenization type
        if tokenization_type == 'full_connectivity':
            from models.tokenized_models import FMRIFullConnectivityTransformer
            model_class = FMRIFullConnectivityTransformer
        elif tokenization_type == 'roi_connectivity':
            from models.tokenized_models import FMRIROIConnectivityTransformer
            model_class = FMRIROIConnectivityTransformer
        else:
            # Default to full connectivity for unknown tokenization types
            from models.tokenized_models import FMRIFullConnectivityTransformer
            model_class = FMRIFullConnectivityTransformer
            logger.warning(f"Unknown fMRI tokenization type '{tokenization_type}', using full_connectivity model")
        
        exp_config_full = {
            'name': f'fMRI {tokenization_type} Tokenized',
            'type': 'tokenized', 
            'modality': 'fmri_tokenized',
            'model_class': model_class,
            'tokenization_type': tokenization_type
        }
        
        return self.cv_framework.run_standard_cv(
            exp_config=exp_config_full,
            X=fmri_tokens,
            y=labels,
            num_folds=num_folds,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=Path(f'/tmp/tokenized_fmri_{tokenization_type}_{exp_name}'),
            seed=seed,
            verbose=verbose,
            exp_name=exp_name
        )
    
    def _run_tokenized_smri_experiment(self, matched_data, num_folds, num_epochs, batch_size, learning_rate, verbose, exp_config, exp_name, seed):
        """Run tokenized sMRI experiment with proper tokenization type and optimized parameters."""
        # Get tokenization type from experiment config
        tokenization_type = exp_config.get('tokenization_type', 'feature_type')
        
        # Use optimized parameters per tokenization type
        optimized_params = self._get_optimized_params_for_tokenization(tokenization_type, 'single_modal')
        
        # Override with optimized parameters if defaults were used
        if learning_rate == 0.001:  # Default learning rate
            learning_rate = optimized_params['learning_rate']
        if batch_size == 16:  # Default batch size
            batch_size = optimized_params['batch_size']
        if num_epochs == 30:  # Default epochs
            num_epochs = optimized_params['num_epochs']
        
        if verbose:
            logger.info(f"Loading tokenized sMRI data with {tokenization_type} tokenization")
            logger.info(f"Parameters: LR={learning_rate}, Batch={batch_size}, Epochs={num_epochs}")
        
        # Load tokenized data with proper tokenization type
        smri_tokens, labels = self.data_loader.load_tokenized_smri_data(
            tokenization_type=tokenization_type, 
            verbose=verbose
        )
        
        # Verify subject consistency
        if len(labels) != matched_data['n_subjects']:
            raise ValueError(
                f"Subject count mismatch! "
                f"Matched data has {matched_data['n_subjects']} subjects, "
                f"but tokenized data has {len(labels)} subjects. "
                f"Tokenized experiments must use the same subjects as regular experiments."
            )
        
        # Create tokenized data in same format as matched_data for consistency
        tokenized_matched_data = {
            'fmri_data': matched_data['fmri_data'],  # Keep original fMRI for single-modal experiment
            'smri_data': smri_tokens,
            'labels': labels,
            'subject_ids': matched_data.get('subject_ids'),
            'n_subjects': len(labels),
            'fmri_dim': matched_data['fmri_dim'],
            'smri_dim': smri_tokens.shape[1]
        }
        
        # Select correct model based on tokenization type
        if tokenization_type == 'brain_network':
            from models.tokenized_models import SMRIBrainNetworkTokenizedTransformer
            model_class = SMRIBrainNetworkTokenizedTransformer
        elif tokenization_type == 'feature_type':
            from models.tokenized_models import SMRIFeatureTypeTokenizedTransformer
            model_class = SMRIFeatureTypeTokenizedTransformer
        else:
            # Default to feature type for unknown tokenization types
            from models.tokenized_models import SMRIFeatureTypeTokenizedTransformer
            model_class = SMRIFeatureTypeTokenizedTransformer
            logger.warning(f"Unknown sMRI tokenization type '{tokenization_type}', using feature_type model")
        
        exp_config_full = {
            'name': f'sMRI {tokenization_type} Tokenized',
            'type': 'tokenized',
            'modality': 'smri_tokenized', 
            'model_class': model_class,
            'tokenization_type': tokenization_type
        }
        
        return self.cv_framework.run_standard_cv(
            exp_config=exp_config_full,
            X=smri_tokens,
            y=labels,
            num_folds=num_folds,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=Path(f'/tmp/tokenized_smri_{tokenization_type}_{exp_name}'),
            seed=seed,
            verbose=verbose,
            exp_name=exp_name
        )
    
    def _run_tokenized_cross_attention_experiment(self, matched_data, num_folds, num_epochs, batch_size, learning_rate, verbose, exp_config, exp_name, seed):
        """Run tokenized cross-attention experiment with proper tokenization types and optimized parameters."""
        # Get tokenization types from experiment config
        fmri_tokenization = exp_config.get('fmri_tokenization', 'full_connectivity')
        smri_tokenization = exp_config.get('smri_tokenization', 'feature_type')
        
        # Use cross-attention optimized params
        optimized_params = self._get_optimized_params_for_tokenization(f"{fmri_tokenization}+{smri_tokenization}", 'cross_attention')
        
        # Override with optimized parameters if defaults were used
        if learning_rate == 0.001:  # Default learning rate
            learning_rate = optimized_params['learning_rate']
        if batch_size == 16:  # Default batch size
            batch_size = optimized_params['batch_size']
        if num_epochs == 30:  # Default epochs
            num_epochs = optimized_params['num_epochs']
        
        if verbose:
            logger.info(f"Loading tokenized data: fMRI {fmri_tokenization}, sMRI {smri_tokenization}")
            logger.info(f"Parameters: LR={learning_rate}, Batch={batch_size}, Epochs={num_epochs}")
        
        # Load tokenized data with proper tokenization types
        fmri_tokens, fmri_labels = self.data_loader.load_tokenized_fmri_data(
            tokenization_type=fmri_tokenization, 
            verbose=verbose
        )
        smri_tokens, smri_labels = self.data_loader.load_tokenized_smri_data(
            tokenization_type=smri_tokenization, 
            verbose=verbose
        )
        
        # Verify subject consistency
        if len(fmri_labels) != matched_data['n_subjects'] or len(smri_labels) != matched_data['n_subjects']:
            raise ValueError(
                f"Subject count mismatch! "
                f"Matched data has {matched_data['n_subjects']} subjects, "
                f"but tokenized data has fMRI:{len(fmri_labels)}, sMRI:{len(smri_labels)} subjects. "
                f"Tokenized experiments must use the same subjects as regular experiments."
            )
        
        # Verify labels are consistent
        import numpy as np
        if not np.array_equal(fmri_labels, smri_labels):
            raise ValueError(
                f"Label mismatch between tokenized fMRI and sMRI data!"
            )
        
        # Create multimodal tokenized data
        X = {
            'fmri': fmri_tokens,
            'smri': smri_tokens
        }
        
        from models.tokenized_models import TokenizedCrossAttentionTransformer
        exp_config_full = {
            'name': f'Tokenized Cross-Attention ({fmri_tokenization.replace("_", " ").title()}+{smri_tokenization.replace("_", " ").title()})',
            'type': 'tokenized_cross_attention',
            'modality': 'multimodal_tokenized',
            'model_class': TokenizedCrossAttentionTransformer,
            'fmri_tokenization_type': fmri_tokenization,
            'smri_tokenization_type': smri_tokenization
        }
        
        return self.cv_framework.run_standard_cv(
            exp_config=exp_config_full,
            X=X,
            y=fmri_labels,
            num_folds=num_folds,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=Path(f'/tmp/tokenized_cross_attention_{fmri_tokenization}_{smri_tokenization}_{exp_name}'),
            seed=seed,
            verbose=verbose,
            exp_name=exp_name
        )
    
    def _get_optimized_params_for_tokenization(self, tokenization_type: str, experiment_type: str = 'single_modal') -> Dict:
        """
        Get optimized training parameters for specific tokenization types.
        Based on parameter optimization results.
        """
        # Parameter optimization results
        
        if experiment_type == 'single_modal':
            # Parameters for good tokenization strategies
            if tokenization_type == 'full_connectivity':
                return {
                    'learning_rate': 0.0008,  # Slightly lower than baseline 0.001 (conservative)
                    'batch_size': 12,         # Between baseline 16 and previous 4
                    'num_epochs': 35,         # Slightly more than baseline 30
                    'd_model': 128,           # Keep baseline model size
                    'n_heads': 4,             # Keep baseline attention heads
                    'n_layers': 2,            # Keep baseline depth
                    'dropout': 0.25,          # Slightly higher than baseline 0.2
                    'layer_dropout': 0.22
                }
            elif tokenization_type == 'roi_connectivity':
                return {
                    'learning_rate': 0.0005,  # Lower LR for high-dimensional ROI tokens (200 tokens)
                    'batch_size': 8,          # Smaller batch due to high memory usage (200x199 tokens)
                    'num_epochs': 50,         # More epochs for complex ROI patterns
                    'd_model': 192,           # Larger model to handle 200 ROI tokens
                    'n_heads': 6,             # More attention heads for ROI interactions
                    'n_layers': 3,            # Deeper network for ROI complexity
                    'dropout': 0.3,           # Higher dropout for regularization with many tokens
                    'layer_dropout': 0.25
                }
            elif tokenization_type == 'feature_type':
                return {
                    'learning_rate': 0.0008,  # Slightly higher LR for feature types
                    'batch_size': 18,         # Larger batch for feature diversity
                    'num_epochs': 45,         # More epochs for feature learning
                    'dropout': 0.18          # Higher dropout for feature types
                }
            elif tokenization_type == 'brain_network':
                return {
                    'learning_rate': 0.0006,  # Lower LR for brain networks
                    'batch_size': 14,         # Moderate batch size
                    'num_epochs': 48,         # More epochs for network patterns
                    'dropout': 0.22          # Higher dropout for brain networks
                }
        
        elif experiment_type == 'cross_attention':
            # Cross-attention parameters based on baseline results
            if tokenization_type == 'cross_attention_basic':
                return {
                    'learning_rate': 0.0015,  # Slightly higher than baseline 0.001
                    'batch_size': 12,         # Smaller batch for multimodal complexity
                    'num_epochs': 40,         # More epochs for cross-modal learning
                    'd_model': 160,           # Slightly larger than baseline 128
                    'n_heads': 5,             # One more than baseline 4
                    'n_layers': 2,            # Keep baseline depth
                    'dropout': 0.25,          # Higher dropout for multimodal complexity
                    'layer_dropout': 0.18
                }
            elif tokenization_type == 'cross_attention_minimal':
                return {
                    'learning_rate': 0.0012,  # Slightly higher than baseline 0.001
                    'batch_size': 14,         # Between baseline 16 and multimodal 12
                    'num_epochs': 35,         # Slightly more than baseline 30
                    'd_model': 128,           # Keep baseline model size for minimal version
                    'n_heads': 4,             # Keep baseline attention heads
                    'n_layers': 2,            # Keep baseline depth
                    'dropout': 0.22,          # Slightly higher than baseline 0.2
                    'layer_dropout': 0.15
                }
            elif 'full_connectivity' in tokenization_type:
                return {
                    'learning_rate': 0.0006,  # Conservative LR for full connectivity cross-attention
                    'batch_size': 8,          # Small batch for memory efficiency
                    'num_epochs': 45,         # More epochs for complex patterns
                    'd_model': 128,           # Keep baseline model size
                    'n_heads': 4,             # Keep baseline attention heads
                    'n_layers': 2,            # Keep baseline depth
                    'dropout': 0.25,          # Higher dropout for regularization
                    'layer_dropout': 0.2
                }
            elif 'roi_connectivity' in tokenization_type:
                return {
                    'learning_rate': 0.0003,  # Very conservative LR for high-complexity ROI cross-attention
                    'batch_size': 6,          # Very small batch for memory efficiency with ROI tokens
                    'num_epochs': 60,         # More epochs for complex ROI cross-modal learning
                    'd_model': 128,           # Conservative model size for multimodal ROI attention
                    'n_heads': 4,             # Conservative attention heads
                    'n_layers': 2,            # Conservative depth for cross-attention
                    'dropout': 0.35,          # High dropout for regularization
                    'layer_dropout': 0.25
                }
            else:
                # Default cross-attention parameters based on baseline results
                return {
                    'learning_rate': 0.001,   # Same as baseline
                    'batch_size': 16,         # Same as baseline
                    'num_epochs': 35,         # Slightly more than baseline 30
                    'd_model': 128,           # Same as baseline
                    'n_heads': 4,             # Same as baseline
                    'n_layers': 2,            # Same as baseline
                    'dropout': 0.22,          # Slightly higher than baseline 0.2
                    'layer_dropout': 0.15
                }
        
        # Default fallback parameters
        return {
            'learning_rate': 0.001,
            'batch_size': 16,
            'num_epochs': 40,
            'dropout': 0.15
        }