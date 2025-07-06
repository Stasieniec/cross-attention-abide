import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")


class ComprehensiveResultAnalyzer:
    """Result analyzer for thesis experiments."""
    
    def __init__(self, output_dir: Path):
        """Initialize the result analyzer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_comprehensive_results(
        self,
        all_results: dict,
        total_time: float,
        verbose: bool,
        include_leave_site_out: bool = True
    ):
        import sys
        import json
        import os
        from datetime import datetime
        
        # Save all results with metadata
        results_with_metadata = {
            'metadata': {
                'total_experiments': len(all_results),
                'successful_experiments': len([r for r in all_results.values() if 'error' not in r]),
                'failed_experiments': len([r for r in all_results.values() if 'error' in r]),
                'total_runtime_minutes': total_time / 60,
                'timestamp': datetime.now().isoformat(),
                'included_leave_site_out': include_leave_site_out,
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'device_used': 'GPU' if 'cuda' in str(os.environ.get('DEVICE', '')).lower() else 'CPU'
            },
            'experiments': all_results
        }
        
        # Save master results file
        master_results_path = self.output_dir / 'complete_thesis_results.json'
        with open(master_results_path, 'w') as f:
            json.dump(results_with_metadata, f, indent=2, default=str)
        
        # Create simple performance summary
        self.create_simple_performance_summary(all_results, include_leave_site_out, verbose)
        
        if verbose:
            print("\nResults Summary")
            print("-" * 30)
            self.print_simple_summary(all_results, include_leave_site_out)
            print("-" * 30)
            print(f"Results saved to: {self.output_dir}")

    def create_simple_performance_summary(self, all_results: dict, include_leave_site_out: bool, verbose: bool = False):
        """Create simple performance summary CSV."""
        summary_data = []
        
        for exp_name, result in all_results.items():
            if 'error' in result:
                summary_data.append({
                    'experiment': exp_name,
                    'name': result.get('name', exp_name),
                    'type': result.get('type', 'unknown'),
                    'modality': result.get('modality', 'unknown'),
                    'status': 'FAILED',
                    'error': str(result['error'])
                })
                continue
            
            base_entry = {
                'experiment': exp_name,
                'name': result.get('name', exp_name),
                'type': result.get('type', 'unknown'),
                'modality': result.get('modality', 'unknown'),
                'status': 'SUCCESS'
            }
            
            # Standard CV results
            if 'aggregated_cv' in result:
                cv = result['aggregated_cv']
                standard_entry = base_entry.copy()
                standard_entry.update({
                    'cv_type': 'Standard CV',
                    'accuracy_mean': cv.get('mean_accuracy', 0),
                    'accuracy_std': cv.get('std_accuracy', 0),
                    'balanced_accuracy_mean': cv.get('mean_balanced_accuracy', 0),
                    'balanced_accuracy_std': cv.get('std_balanced_accuracy', 0),
                    'auc_mean': cv.get('mean_auc', 0),
                    'auc_std': cv.get('std_auc', 0),
                    'n_folds': len(cv.get('individual_results', [])),
                    'performance_string': f"{cv.get('mean_accuracy', 0):.1f}% +- {cv.get('std_accuracy', 0):.1f}%"
                })
                summary_data.append(standard_entry)
            elif 'standard_cv' in result:
                cv = result['standard_cv']
                standard_entry = base_entry.copy()
                standard_entry.update({
                    'cv_type': 'Standard CV',
                    'accuracy_mean': cv.get('mean_accuracy', 0),
                    'accuracy_std': cv.get('std_accuracy', 0),
                    'balanced_accuracy_mean': cv.get('mean_balanced_accuracy', 0),
                    'balanced_accuracy_std': cv.get('std_balanced_accuracy', 0),
                    'auc_mean': cv.get('mean_auc', 0),
                    'auc_std': cv.get('std_auc', 0),
                    'n_folds': len(cv.get('fold_results', [])),
                    'performance_string': f"{cv.get('mean_accuracy', 0):.1f}% +- {cv.get('std_accuracy', 0):.1f}%"
                })
                summary_data.append(standard_entry)
            
            # Leave-site-out CV results
            if include_leave_site_out and 'aggregated_lso' in result:
                lso = result['aggregated_lso']
                lso_entry = base_entry.copy()
                lso_entry.update({
                    'cv_type': 'Leave-Site-Out CV',
                    'accuracy_mean': lso.get('mean_accuracy', 0),
                    'accuracy_std': lso.get('std_accuracy', 0),
                    'balanced_accuracy_mean': lso.get('mean_balanced_accuracy', 0),
                    'balanced_accuracy_std': lso.get('std_balanced_accuracy', 0),
                    'auc_mean': lso.get('mean_auc', 0),
                    'auc_std': lso.get('std_auc', 0),
                    'n_sites': lso.get('n_sites', 0),
                    'performance_string': f"{lso.get('mean_accuracy', 0):.1f}% +- {lso.get('std_accuracy', 0):.1f}%"
                })
                summary_data.append(lso_entry)
            elif include_leave_site_out and 'leave_site_out_cv' in result and 'error' not in result['leave_site_out_cv']:
                lso = result['leave_site_out_cv']
                lso_entry = base_entry.copy()
                lso_entry.update({
                    'cv_type': 'Leave-Site-Out CV',
                    'accuracy_mean': lso.get('mean_accuracy', 0),
                    'accuracy_std': lso.get('std_accuracy', 0),
                    'balanced_accuracy_mean': lso.get('mean_balanced_accuracy', 0),
                    'balanced_accuracy_std': lso.get('std_balanced_accuracy', 0),
                    'auc_mean': lso.get('mean_auc', 0),
                    'auc_std': lso.get('std_auc', 0),
                    'n_sites': lso.get('n_sites', 0),
                    'performance_string': f"{lso.get('mean_accuracy', 0):.1f}% +- {lso.get('std_accuracy', 0):.1f}%"
                })
                summary_data.append(lso_entry)
        
        # Save CSV summary
        if summary_data:
            import pandas as pd
            df = pd.DataFrame(summary_data)
            csv_path = self.output_dir / 'detailed_performance_summary.csv'
            df.to_csv(csv_path, index=False)
            if verbose:
                print(f"Performance summary: {csv_path}")
        else:
            if verbose:
                print("No valid experiment data found")

    def print_simple_summary(self, all_results: dict, include_leave_site_out: bool):
        """Print simple summary to console."""
        type_groups = {}
        for exp_name, result in all_results.items():
            if 'error' in result:
                continue
            
            exp_type = result.get('type', 'unknown')
            if exp_type not in type_groups:
                type_groups[exp_type] = []
            type_groups[exp_type].append((exp_name, result))
        
        for exp_type, experiments in type_groups.items():
            print(f"\n{exp_type.upper().replace('_', ' ')} EXPERIMENTS:")
            
            for exp_name, result in experiments:
                print(f"  {result.get('name', exp_name)}")
                
                if 'standard_cv' in result:
                    cv = result['standard_cv']
                    print(f"Standard CV: {cv['mean_accuracy']:.1f}% +- {cv['std_accuracy']:.1f}%")
                
                if include_leave_site_out and 'leave_site_out_cv' in result and 'error' not in result['leave_site_out_cv']:
                    lso = result['leave_site_out_cv']
                    print(f"Leave-Site-Out: {lso['mean_accuracy']:.1f}% +- {lso['std_accuracy']:.1f}%")