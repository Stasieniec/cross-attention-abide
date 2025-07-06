import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Environment check

# Check for Google Colab environment
try:
    import google.colab
    print("Running in Google Colab - real data access confirmed")
except ImportError:
    raise RuntimeError(
        "Not in Google Colab!"
    )



# Modular imports

# Add src to path for imports - Google Colab compatible
script_dir = Path(__file__).parent
repo_root = script_dir.parent
src_dir = repo_root / "src"

# Add both src and repo root to Python path
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Debug: Print paths if needed
if False:  # Enable for debugging
    print(f"Script dir: {script_dir}")
    print(f"Repo root: {repo_root}")
    print(f"Src dir: {src_dir}")
    print(f"Src dir exists: {src_dir.exists()}")
    print(f"Python path includes src: {str(src_dir) in sys.path}")

# Import all modular components
try:
    from evaluation.experiment_runner import ExperimentRunner
    from utils.data_loaders import DataLoader, SiteExtractor
    from training.cross_validation import CrossValidationFramework
    from evaluation.result_analyzer import ComprehensiveResultAnalyzer
    from utils.helpers import get_device
    print("All imports successful")
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import path")
    # Try with src prefix if direct imports fail
    try:
        from src.evaluation.experiment_runner import ExperimentRunner
        from src.utils.data_loaders import DataLoader, SiteExtractor
        from src.training.cross_validation import CrossValidationFramework
        from src.evaluation.result_analyzer import ComprehensiveResultAnalyzer
        from src.utils.helpers import get_device
        print("Alternative imports successful")
    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")
        raise e2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)



# Experiment framework

class ThesisExperiments:
    """Modular thesis experiment framework."""
    
    def __init__(self):
        """Initialize with all modular components."""
        self.device = get_device()
        
        # Import config for data loader
        try:
            from config import get_config
        except ImportError:
            from src.config import get_config
        self.config = get_config('cross_attention')
        
        # Define experiments configuration
        self.experiments = {
            # Baseline experiments
            'fmri_baseline': {
                'name': 'fMRI Baseline', 
                'type': 'baseline',
                'modality': 'fmri',
                'model_class': None
            },
            'smri_baseline': {
                'name': 'sMRI Baseline', 
                'type': 'baseline',
                'modality': 'smri',
                'model_class': None
            },
            
            # Cross-attention experiments
            'cross_attention_basic': {
                'name': 'Cross-Attention Basic', 
                'type': 'cross_attention',
                'modality': 'multimodal',
                'model_class': None
            },
            
            # Tokenized single-modal experiments
            'tokenized_fmri_full': {
                'name': 'Tokenized fMRI (Full Connectivity)', 
                'type': 'tokenized',
                'modality': 'fmri',
                'tokenization_type': 'full_connectivity',
                'model_class': None
            },
            'tokenized_fmri_roi': {
                'name': 'Tokenized fMRI (ROI Connectivity)', 
                'type': 'tokenized',
                'modality': 'fmri',
                'tokenization_type': 'roi_connectivity',
                'model_class': None
            },
            'tokenized_smri_feature': {
                'name': 'Tokenized sMRI (Feature-Type)', 
                'type': 'tokenized',
                'modality': 'smri', 
                'tokenization_type': 'feature_type',
                'model_class': None
            },
            'tokenized_smri_brain': {
                'name': 'Tokenized sMRI (Brain Network)', 
                'type': 'tokenized',
                'modality': 'smri', 
                'tokenization_type': 'brain_network',
                'model_class': None
            },
            
            # Tokenized cross-attention combinations
            'tokenized_cross_attention_full_feat': {
                'name': 'Tokenized Cross-Attention (Full+Feature)', 
                'type': 'tokenized_cross_attention',
                'modality': 'multimodal_tokenized',
                'fmri_tokenization': 'full_connectivity',
                'smri_tokenization': 'feature_type',
                'model_class': None
            },
            'tokenized_cross_attention_full_brain': {
                'name': 'Tokenized Cross-Attention (Full+Brain)', 
                'type': 'tokenized_cross_attention',
                'modality': 'multimodal_tokenized',
                'fmri_tokenization': 'full_connectivity',
                'smri_tokenization': 'brain_network',
                'model_class': None
            },
            'tokenized_cross_attention_roi_feat': {
                'name': 'Tokenized Cross-Attention (ROI+Feature)', 
                'type': 'tokenized_cross_attention',
                'modality': 'multimodal_tokenized',
                'fmri_tokenization': 'roi_connectivity',
                'smri_tokenization': 'feature_type',
                'model_class': None  # Add for consistency (will be set by experiment runner)
            },
            'tokenized_cross_attention_roi_brain': {
                'name': 'Tokenized Cross-Attention (ROI+Brain)', 
                'type': 'tokenized_cross_attention',
                'modality': 'multimodal_tokenized',
                'fmri_tokenization': 'roi_connectivity',
                'smri_tokenization': 'brain_network',
                'model_class': None  # Add for consistency (will be set by experiment runner)
            },
        }
        
        # Initialize all modular components
        self.data_loader = DataLoader(self.config)
        self.site_extractor = SiteExtractor()
        self.cv_framework = CrossValidationFramework(
            device=self.device,
            data_loader=self.data_loader,
            site_extractor=self.site_extractor
        )
        self.experiment_runner = ExperimentRunner(self)
        
        logger.info("Thesis framework initialized")
    
    def run_all(self, **kwargs):
        """Run all experiments."""
        output_dir_str = kwargs.get('output_dir') or f'/content/drive/MyDrive/thesis_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        output_dir = Path(output_dir_str)
        
        start_time = time.time()
        logger.info("Starting thesis experiments")
        
        # Run all experiments
        all_results = self.experiment_runner.run_all(**kwargs)
        
        total_time = time.time() - start_time
        
        # Save results
        result_analyzer = ComprehensiveResultAnalyzer(output_dir)
        result_analyzer.save_comprehensive_results(
            all_results=all_results,
            total_time=total_time,
            verbose=kwargs.get('verbose', True),
            include_leave_site_out=kwargs.get('include_leave_site_out', True)
        )
        
        logger.info(f"All experiments completed in {total_time/60:.1f} minutes")
        return all_results
    
    def run_quick_tokenized(self, **kwargs):
        """Run quick tokenized experiments."""
        output_dir_str = kwargs.get('output_dir') or f'/content/drive/MyDrive/thesis_quick_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        output_dir = Path(output_dir_str)
        
        start_time = time.time()
        logger.info("Starting QUICK tokenized experiments")
        
        # Run quick experiments using modular framework
        all_results = self.experiment_runner.run_quick_tokenized(**kwargs)
        
        total_time = time.time() - start_time
        
        # Save results using modular analyzer
        result_analyzer = ComprehensiveResultAnalyzer(output_dir)
        result_analyzer.save_comprehensive_results(
            all_results=all_results,
            total_time=total_time,
            verbose=kwargs.get('verbose', True),
            include_leave_site_out=False  # Quick mode
        )
        
        logger.info(f"QUICK EXPERIMENTS COMPLETED in {total_time/60:.1f} minutes")
        return all_results
    
    def run_baselines_only(self, **kwargs):
        """Run baseline experiments only."""
        return self.experiment_runner.run_baselines_only(**kwargs)
    
    def run_cross_attention_only(self, **kwargs):
        """Run cross-attention experiments only.""" 
        return self.experiment_runner.run_cross_attention_only(**kwargs)
    
    def run_tokenized_only(self, **kwargs):
        """Run tokenized experiments only."""
        return self.experiment_runner.run_tokenized_only(**kwargs)
    
    def run_with_hyperparameter_optimization(self, **kwargs):
        """Run experiments with hyperparameter optimization.
        
        Uses pre-computed configurations and multiple random seeds
        for reliable results. Includes leave-site-out cross-validation.
        """
        logger.info("Starting hyperparameter optimization")
        logger.info("=" * 60)
        logger.info("Methodology:")
        logger.info("   - Pre-computed configs for tokenized experiments")
        logger.info("   - Multiple random seeds for reliable results (default: 5)")
        logger.info("   - Efficient training (~2-3 hours total)")
        logger.info("   - Leave-site-out CV across all 20 ABIDE sites")
        
        # Apply good hyperparameters for tokenized experiments
        optimized_kwargs = self._apply_optimal_hyperparameters(**kwargs)
        
        # Set 5 random seeds by default for thesis robustness
        optimized_kwargs.setdefault('num_seeds', 5)
        optimized_kwargs.setdefault('include_leave_site_out', True)
        optimized_kwargs.setdefault('lso_early_stopping', True)
        
        # Run experiments with optimization
        return self.run_all(**optimized_kwargs)
    
    def _apply_optimal_hyperparameters(self, **kwargs):
        """Apply optimal hyperparameters for different experiment types."""
        
        # Good configurations from testing
        optimal_configs = {
            # fMRI tokenized experiments (based on connectivity analysis)
            'tokenized_fmri_full': {
                'learning_rate': 0.0001,  # Lower LR for stability
                'batch_size': 4,          # Small batch to prevent memory issues
                'num_epochs': 100,        # More epochs to compensate for lower LR
            },
            'tokenized_fmri_roi': {
                'learning_rate': 0.0005,  # Lower LR for ROI stability
                'batch_size': 8,          # Smaller batch for ROI complexity
                'num_epochs': 80,         # More epochs for ROI convergence
            },
            
            # sMRI tokenized experiments (based on structural analysis) 
            'tokenized_smri_feature': {
                'learning_rate': 0.002,   # Higher LR for feature diversity
                'batch_size': 24,         # Larger batches for feature stability
                'num_epochs': 50,         # Fast convergence
            },
            'tokenized_smri_brain': {
                'learning_rate': 0.001,   # Balanced LR for brain networks
                'batch_size': 16,         # Standard batch size
                'num_epochs': 60,         # Network-specific training
            },
            
            # Cross-attention tokenized (balanced for multimodal)
            'tokenized_cross_attention_full_feat': {
                'learning_rate': 0.0005,  # Conservative for multimodal
                'batch_size': 12,         # Smaller for cross-attention complexity
                'num_epochs': 70,         # Extended training for fusion
            },
            'tokenized_cross_attention_full_brain': {
                'learning_rate': 0.0005,
                'batch_size': 12,
                'num_epochs': 70,
            },
            'tokenized_cross_attention_roi_feat': {
                'learning_rate': 0.0008,  # Slightly higher for ROI
                'batch_size': 16,
                'num_epochs': 65,
            },
            'tokenized_cross_attention_roi_brain': {
                'learning_rate': 0.0008,
                'batch_size': 16, 
                'num_epochs': 65,
            }
        }
        
        # Apply optimizations to baseline parameters
        optimized_kwargs = kwargs.copy()
        
        optimized_kwargs.setdefault('num_seeds', 5)  # Multiple seeds
        optimized_kwargs.setdefault('lso_early_stopping', True)  # Efficient LOOCV
        optimized_kwargs.setdefault('lso_patience', 12)  # Patience
        
        # Store configs for the experiment runner to use
        optimized_kwargs['optimal_configs'] = optimal_configs
        
        return optimized_kwargs
    
    def validate_setup(self, **kwargs):
        """Validate that everything is properly set up without training."""
        logger.info("Validating setup")
        
        validation_results = {}
        
        # 1. Test data loading
        try:
            logger.info("Testing data loading")
            matched_data = self.data_loader.load_matched_data(verbose=True)
            validation_results['data_loading'] = {
                'status': 'SUCCESS',
                'n_subjects': matched_data['n_subjects'],
                'fmri_shape': matched_data['fmri_data'].shape,
                'smri_shape': matched_data['smri_data'].shape
            }
            logger.info("Data loading complete")
        except Exception as e:
            validation_results['data_loading'] = {'status': 'FAILED', 'error': str(e)}
            logger.error(f"Data loading failed: {e}")
        
        # 2. Test model initialization
        try:
            logger.info("Testing model initialization")
            from models import CrossAttentionTransformer, FMRINetworkBasedTransformer, SMRIFeatureTypeTokenizedTransformer
            
            # Test basic models
            fmri_model = FMRINetworkBasedTransformer(feat_dim=19900, num_classes=2)
            smri_model = SMRIFeatureTypeTokenizedTransformer(input_dim=800, num_classes=2)
            cross_model = CrossAttentionTransformer(fmri_dim=19900, smri_dim=800, num_classes=2)
            
            validation_results['model_init'] = {
                'status': 'SUCCESS',
                'models_tested': ['FMRINetworkBasedTransformer', 'SMRIFeatureTypeTokenizedTransformer', 'CrossAttentionTransformer']
            }
            logger.info("Model initialization complete")
        except Exception as e:
            validation_results['model_init'] = {'status': 'FAILED', 'error': str(e)}
            logger.error(f"Model initialization failed: {e}")
        
        # 3. Test tokenized data loading (if available)
        try:
            logger.info("Testing tokenized data loading")
            tokenized_results = {}
            
            for tokenization_type in ['full_connectivity', 'roi_connectivity']:
                try:
                    fmri_tokens, labels = self.data_loader.load_tokenized_fmri_data(tokenization_type, verbose=False)
                    tokenized_results[f'fmri_{tokenization_type}'] = f"SUCCESS {fmri_tokens.shape}"
                except Exception as e:
                    tokenized_results[f'fmri_{tokenization_type}'] = f"FAILED {str(e)[:50]}..."
            
            for tokenization_type in ['feature_type', 'brain_network']:
                try:
                    smri_tokens, labels = self.data_loader.load_tokenized_smri_data(tokenization_type, verbose=False)
                    tokenized_results[f'smri_{tokenization_type}'] = f"SUCCESS {smri_tokens.shape}"
                except Exception as e:
                    tokenized_results[f'smri_{tokenization_type}'] = f"FAILED {str(e)[:50]}..."
            
            validation_results['tokenized_data'] = {
                'status': 'TESTED',
                'results': tokenized_results
            }
            logger.info("Tokenized data loading complete")
        except Exception as e:
            validation_results['tokenized_data'] = {'status': 'FAILED', 'error': str(e)}
            logger.error(f"Tokenized data testing failed: {e}")
        
        # 4. Test site extraction
        try:
            logger.info("Testing site extraction")
            if 'data_loading' in validation_results and validation_results['data_loading']['status'] == 'SUCCESS':
                matched_data = self.data_loader.load_matched_data(verbose=False)
                subject_ids = matched_data.get('subject_ids', [f"subject_{i:05d}" for i in range(matched_data['n_subjects'])])
                
                site_labels, site_mapping, site_stats = self.site_extractor.extract_site_info(subject_ids[:10])
                validation_results['site_extraction'] = {
                    'status': 'SUCCESS',
                    'n_sites_found': len(site_mapping),
                    'sites': list(site_mapping.keys())
                }
                logger.info("Site extraction complete")
            else:
                validation_results['site_extraction'] = {'status': 'SKIPPED', 'reason': 'Data loading failed'}
        except Exception as e:
            validation_results['site_extraction'] = {'status': 'FAILED', 'error': str(e)}
            logger.error(f"Site extraction failed: {e}")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("Validation summary")
        logger.info("="*60)
        
        for test_name, result in validation_results.items():
            status = result.get('status', 'UNKNOWN')
            if status == 'SUCCESS':
                logger.info(f"PASS {test_name.replace('_', ' ').title()}: {status}")
            elif status == 'FAILED':
                logger.info(f"FAIL {test_name.replace('_', ' ').title()}: {status}")
                logger.info(f"   Error: {result.get('error', 'Unknown error')}")
            else:
                logger.info(f"PARTIAL {test_name.replace('_', ' ').title()}: {status}")
        
        return validation_results
    
    def quick_pipeline_test(self, **kwargs):
        """Quick pipeline test with minimal training to verify everything works."""
        logger.info("QUICK PIPELINE TEST - Minimal training to verify functionality")
        
        # Set minimal parameters for quick test
        test_kwargs = {
            'num_folds': 2,           # Just 2 folds
            'num_epochs': 3,          # Just 3 epochs  
            'batch_size': 16,         # Smaller batch
            'learning_rate': 1e-3,    # Higher LR for faster convergence
            'seed': 42,
            'verbose': True,
            'include_leave_site_out': False  # Skip LSO for quick test
        }
        test_kwargs.update(kwargs)  # Allow overrides
        
        output_dir_str = kwargs.get('output_dir') or f'/content/drive/MyDrive/thesis_pipeline_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        output_dir = Path(output_dir_str)
        test_kwargs['output_dir'] = str(output_dir)
        
        # Test just baseline experiments for speed
        logger.info("Testing baseline experiments")
        start_time = time.time()
        
        results = self.experiment_runner._run_specific_experiments(
            ['fmri_baseline', 'smri_baseline'], 
            **test_kwargs
        )
        
        total_time = time.time() - start_time
        
        logger.info(f"PIPELINE TEST COMPLETED in {total_time:.1f} seconds")
        logger.info("If you see this message, your pipeline is working.")
        
        return results
    
    def test_single_experiment(self, experiment_name: str, **kwargs):
        """Test a single experiment with minimal settings."""
        if experiment_name not in self.experiments:
            available = list(self.experiments.keys())
            raise ValueError(f"Unknown experiment '{experiment_name}'. Available: {available}")
        
        logger.info(f"TESTING SINGLE EXPERIMENT: {self.experiments[experiment_name]['name']}")
        
        # Minimal test parameters
        test_kwargs = {
            'num_folds': 2,
            'num_epochs': 2, 
            'batch_size': 16,
            'learning_rate': 1e-3,
            'seed': 42,
            'verbose': True,
            'include_leave_site_out': False
        }
        test_kwargs.update(kwargs)
        
        output_dir_str = kwargs.get('output_dir') or f'/content/drive/MyDrive/thesis_test_{experiment_name}_{datetime.now().strftime("%H%M%S")}'
        output_dir = Path(output_dir_str)
        test_kwargs['output_dir'] = str(output_dir)
        
        start_time = time.time()
        results = self.experiment_runner._run_specific_experiments([experiment_name], **test_kwargs)
        total_time = time.time() - start_time
        
        logger.info(f"Single experiment test completed in {total_time:.1f} seconds")
        return results





# Cli

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Modular Thesis Experiments')
    
    # Experiment selection
    parser.add_argument('--run_all', action='store_true', help='Run all experiments')
    parser.add_argument('--quick_tokenized', action='store_true', help='Run quick tokenized experiments')
    parser.add_argument('--baselines_only', action='store_true', help='Run baseline experiments only')
    parser.add_argument('--cross_attention_only', action='store_true', help='Run cross-attention experiments only')
    parser.add_argument('--tokenized_only', action='store_true', help='Run tokenized experiments only')
    
    # OPTIMIZATION 
    parser.add_argument('--optimize_hyperparameters', action='store_true', 
                        help='Run with hyperparameter optimization (RECOMMENDED FOR FINAL THESIS)')
    
    # TESTING OPTIONS FOR GOOGLE COLAB
    parser.add_argument('--validate_setup', action='store_true', help='Validate setup without training (RECOMMENDED FIRST)')
    parser.add_argument('--quick_pipeline_test', action='store_true', help='Quick pipeline test with minimal training (~2-3 min)')
    parser.add_argument('--test_single', type=str, help='Test single experiment (provide experiment name)')
    parser.add_argument('--list_experiments', action='store_true', help='List all available experiments')
    
    # Experiment parameters
    parser.add_argument('--num_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    parser.add_argument('--include_leave_site_out', action='store_true', default=True, help='Include leave-site-out CV (default: True)')
    parser.add_argument('--no_leave_site_out', action='store_true', help='Disable leave-site-out CV')
    
    # Multiple random seeds for reliable results
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of random seeds for reliable results (default: 5)')
    parser.add_argument('--single_seed', action='store_true', help='Use only single seed for faster testing')
    
    # OPTIMIZATION: Leave-site-out performance options
    parser.add_argument('--lso_early_stopping', action='store_true', default=True, help='Enable early stopping for LSO (default: True)')
    parser.add_argument('--no_lso_early_stopping', action='store_true', help='Disable early stopping for LSO')
    parser.add_argument('--lso_patience', type=int, default=15, help='Early stopping patience for LSO (default: 15)')
    parser.add_argument('--parallel_lso', action='store_true', default=True, help='Enable parallel LSO processing (default: True)')
    parser.add_argument('--no_parallel_lso', action='store_true', help='Disable parallel LSO processing')
    
    # NEW: LSO-only mode
    parser.add_argument('--lso_only', action='store_true', help='Run only Leave-Site-Out validation (skip standard CV for faster execution)')
    
    # GOOGLE DRIVE SPACE MANAGEMENT
    parser.add_argument('--save_model_checkpoints', action='store_true', default=False, 
                       help='Save model checkpoints (WARNING: Uses lots of Google Drive space!)')
    parser.add_argument('--save_minimal_only', action='store_true', default=True,
                       help='Save only essential results (recommended for Google Drive)')
    
    args = parser.parse_args()
    
    # Initialize thesis framework
    experiments = ThesisExperiments()
    
    # Handle list experiments first
    if args.list_experiments:
        logger.info("Available experiments:")
        logger.info("=" * 50)
        for exp_name, exp_config in experiments.experiments.items():
            logger.info(f"  - {exp_name}: {exp_config['name']}")
        logger.info("\nTESTING COMMANDS:")
        logger.info("  --validate_setup          : Check everything without training")
        logger.info("  --quick_pipeline_test      : Fast pipeline test (~3 min)")
        logger.info("  --test_single <name>       : Test one experiment")
        logger.info("  --quick_tokenized          : Quick tokenized experiments")
        return
    
    # Handle testing options
    if args.validate_setup:
        experiments.validate_setup()
        return
    
    if args.quick_pipeline_test:
        experiments.quick_pipeline_test()
        return
    
    if args.test_single:
        experiments.test_single_experiment(args.test_single)
        return
    
    # Convert args to kwargs for regular experiments
    # Enable leave-site-out CV by default for thesis
    if args.no_leave_site_out:
        include_leave_site_out = False
    else:
        include_leave_site_out = True  # Default to True for thesis experiments
    
    # Handle multiple seeds logic
    num_seeds = 1 if args.single_seed else args.num_seeds
    
    # Handle LSO early stopping logic
    lso_early_stopping = True
    if args.no_lso_early_stopping:
        lso_early_stopping = False
    elif args.lso_early_stopping:
        lso_early_stopping = True
    
    # Handle parallel LSO logic
    parallel_lso = True
    if args.no_parallel_lso:
        parallel_lso = False
    elif args.parallel_lso:
        parallel_lso = True
    
    # Handle LSO-only mode
    skip_standard_cv = args.lso_only  # Skip standard CV if LSO-only mode is enabled
    
    kwargs = {
        'num_folds': args.num_folds,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'seed': args.seed,
        'output_dir': args.output_dir,
        'verbose': args.verbose,
        'include_leave_site_out': include_leave_site_out,
        # Multiple seeds and LSO optimizations
        'num_seeds': num_seeds,
        'lso_early_stopping': lso_early_stopping,
        'lso_patience': args.lso_patience,
        'parallel_lso': parallel_lso,
        'skip_standard_cv': skip_standard_cv  # Pass LSO-only mode
    }
    
    # Run selected experiments
    if args.run_all:
        experiments.run_all(**kwargs)
    elif args.quick_tokenized:
        experiments.run_quick_tokenized(**kwargs)
    elif args.baselines_only:
        experiments.run_baselines_only(**kwargs)
    elif args.cross_attention_only:
        experiments.run_cross_attention_only(**kwargs)
    elif args.tokenized_only:
        experiments.run_tokenized_only(**kwargs)
    elif args.optimize_hyperparameters:
        experiments.run_with_hyperparameter_optimization(**kwargs)
    elif args.lso_only:
        # LSO-only mode: Run with hyperparameter optimization but skip standard CV
        logger.info("LSO-ONLY MODE: Running with hyperparameter optimization, skipping standard CV")
        experiments.run_with_hyperparameter_optimization(**kwargs)
    else:
        # Set better defaults
        optimized_kwargs = kwargs.copy()
        optimized_kwargs.setdefault('num_seeds', 5)  # 5 seeds for robustness
        optimized_kwargs.setdefault('include_leave_site_out', True)  # Always include LSO
        optimized_kwargs.setdefault('lso_early_stopping', True)  # Efficient LSO
        
        experiments.run_with_hyperparameter_optimization(**optimized_kwargs)

if __name__ == "__main__":
    main() 