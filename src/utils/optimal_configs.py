import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class OptimalConfigManager:
    """Manages configurations for tokenized experiments."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        logger.info("OptimalConfigManager initialized with configurations")
        
        # Configurations found through testing and validation
        self.optimal_configs = {
            # fMRI Tokenized Experiments
            'fmri_functional_network': {
                'learning_rate': 5e-5,
                'batch_size': 4,
                'num_epochs': 120,
                'dropout': 0.3,
                'weight_decay': 1e-5,
                'd_model': 128,
                'nhead': 8,
                'num_layers': 3,
                'dim_feedforward': 256
            },
            
            # Good performing strategies from testing
            
            # #1: cross_full_feat - 71.3% test accuracy, 71.1% CV accuracy  
            'cross_full_feat': {
                'learning_rate': 0.001,
                'batch_size': 16,
                'num_epochs': 30,
                'd_model': 128,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.18
            },
            
            # #2: smri_brain_network - 64.4% test accuracy, 66.0% CV accuracy
            'smri_brain_network': {
                'learning_rate': 0.001,
                'batch_size': 16,
                'num_epochs': 30,
                'd_model': 192,
                'n_heads': 6,
                'n_layers': 2,
                'dropout': 0.2
            },
            
            # #3: smri_feature_type - 59.2% test accuracy, 62.2% CV accuracy
            'smri_feature_type': {
                'learning_rate': 0.001,
                'batch_size': 16,
                'num_epochs': 30,
                'd_model': 128,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.18
            },
            
            # Other cross-attention strategies
            'cross_func_brain': {
                'learning_rate': 0.001,
                'batch_size': 16,
                'num_epochs': 30,
                'd_model': 128,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.2
            },
            
            'cross_full_brain': {
                'learning_rate': 0.001,
                'batch_size': 16,
                'num_epochs': 30,
                'd_model': 128,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.15
            },
            
            'cross_net_feat': {
                'learning_rate': 0.001,
                'batch_size': 16,
                'num_epochs': 30,
                'd_model': 128,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.2
            },
            
            'cross_net_brain': {
                'learning_rate': 0.001,
                'batch_size': 16,
                'num_epochs': 30,
                'd_model': 128,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.18
            },
            
            'cross_func_feat': {
                'learning_rate': 0.001,
                'batch_size': 16,
                'num_epochs': 30,
                'd_model': 128,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.15
            },
            
            # fMRI tokenized strategies
            'fmri_full_connectivity': {
                'learning_rate': 0.001,
                'batch_size': 16,
                'num_epochs': 30,
                'd_model': 128,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.2
            },
            
            'fmri_network_based': {
                'learning_rate': 0.001,
                'batch_size': 16,
                'num_epochs': 30,
                'd_model': 128,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.2
            }
        }
    
    def get_optimal_config(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get configuration for a given strategy.
        
        Args:
            strategy_name: Name of the experiment strategy
            
        Returns:
            Dictionary containing hyperparameters
        """
        if strategy_name in self.optimal_configs:
            config = self.optimal_configs[strategy_name].copy()
            logger.info(f"Using config for {strategy_name}: {config}")
            return config
        else:
            # Provide reasonable defaults for unknown strategies
            default_config = {
                'learning_rate': 1e-4,
                'batch_size': 8, 
                'num_epochs': 80,
                'dropout': 0.2,
                'weight_decay': 1e-4
            }
            logger.warning(f"No config found for {strategy_name}, using defaults: {default_config}")
            return default_config
    
    def get_config_for_experiment(self, exp_name: str, exp_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get configuration for a thesis experiment.
        
        Args:
            exp_name: Experiment name from thesis_experiments.py
            exp_config: Experiment configuration dictionary
            
        Returns:
            Dictionary with hyperparameters
        """
        # Map experiment names to strategy names (EXACT match with thesis_experiments.py)
        strategy_mapping = {
            # Single-modal tokenized (EXACT names from thesis_experiments.py)
            'tokenized_smri_brain': 'smri_brain_network',
            'tokenized_smri_feature': 'smri_feature_type', 
            'tokenized_fmri_functional': 'fmri_functional_network',
            'tokenized_fmri_network': 'fmri_network_based',
            'tokenized_fmri_full': 'fmri_full_connectivity',
            
            # Cross-attention tokenized (EXACT names from thesis_experiments.py)
            'tokenized_cross_attention_func_feat': 'cross_func_feat',
            'tokenized_cross_attention_func_brain': 'cross_func_brain',
            'tokenized_cross_attention_net_feat': 'cross_net_feat',
            'tokenized_cross_attention_net_brain': 'cross_net_brain',
            'tokenized_cross_attention_full_feat': 'cross_full_feat',
            'tokenized_cross_attention_full_brain': 'cross_full_brain'
        }
        
        if exp_name in strategy_mapping:
            strategy_name = strategy_mapping[exp_name]
            return self.get_optimal_config(strategy_name)
        else:
            # For baseline experiments, return default config
            return {
                'learning_rate': 0.001,
                'batch_size': 16,
                'num_epochs': 30,
                'd_model': 128,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.2
            }
    
    def get_top_strategies(self, n: int = 3) -> List[Dict[str, Any]]:
        """
        Get the top N performing strategies from testing.
        
        Args:
            n: Number of top strategies to return
            
        Returns:
            List of top strategies with their configurations
        """
        # Top strategies based on testing results
        top_strategies = [
            {
                'name': 'cross_full_feat',
                'description': 'Cross-Attention (Full Connectivity + Feature Type)',
                'test_accuracy': 71.3,
                'cv_accuracy': 71.1,
                'config': self.optimal_configs['cross_full_feat']
            },
            {
                'name': 'smri_brain_network', 
                'description': 'sMRI Brain Network Tokenized',
                'test_accuracy': 64.4,
                'cv_accuracy': 66.0,
                'config': self.optimal_configs['smri_brain_network']
            },
            {
                'name': 'smri_feature_type',
                'description': 'sMRI Feature Type Tokenized', 
                'test_accuracy': 59.2,
                'cv_accuracy': 62.2,
                'config': self.optimal_configs['smri_feature_type']
            }
        ]
        
        return top_strategies[:n]
    
    def print_all_configs(self):
        """Print all available configurations."""
        logger.info("\nConfigurations:")
        
        for strategy, config in self.optimal_configs.items():
            logger.info(f"\n{strategy}:")
            for param, value in config.items():
                logger.info(f"  {param}: {value}")


# Global instance for easy access
optimal_config_manager = OptimalConfigManager()


def get_optimal_config(strategy_name: str) -> Dict[str, Any]:
    """Convenience function to get config for a strategy."""
    return optimal_config_manager.get_optimal_config(strategy_name)


def get_config_for_experiment(exp_name: str, exp_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to get config for an experiment."""
    return optimal_config_manager.get_config_for_experiment(exp_name, exp_config) 