import os
import warnings
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import EarlyStopping, calculate_class_weights

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer with metrics tracking."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Any,
        model_type: str = 'single',
        experiment_name: str = "experiment"
    ):
        self.model = model
        self.device = device
        self.config = config
        self.model_type = model_type
        self.experiment_name = experiment_name
        
        # Initialize optimizers and schedulers
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = None  # Will be created in fit() with training data
        self.scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
        
        # Metrics tracking
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_balanced_accuracy': [],
            'val_balanced_accuracy': [],
            'train_auc': [],
            'val_auc': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rates': [],
            'epoch_times': [],
            'gradient_norms': []
        }
        
        # Best model tracking
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.early_stop_counter = 0
        
        # Scientific analysis data
        self.fold_analyses = []
        self.detailed_predictions = {}
    
    def _create_optimizer(self) -> AdamW:
        """Create optimizer with config parameters."""
        return AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        return ExponentialLR(self.optimizer, gamma=0.97)
    
    def _create_criterion(self, y_train: Optional[np.ndarray] = None) -> nn.Module:
        """Create loss criterion."""
        if hasattr(self.config, 'use_class_weights') and self.config.use_class_weights and y_train is not None:
            # Always use class weights for sMRI as it helps with performance
            class_weights = calculate_class_weights(y_train, self.device)
            return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        
        if hasattr(self.config, 'label_smoothing') and self.config.label_smoothing > 0:
            return nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        
        return nn.CrossEntropyLoss()
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],  # FIXED: Make validation loader optional for CV
        num_epochs: int,
        checkpoint_path: Optional[Path] = None,
        y_train: Optional[np.ndarray] = None,
        save_detailed_history: bool = True,
        save_model_checkpoints: bool = True   # Enable by default for best model evaluation
    ) -> Dict[str, Any]:
        """
        Training with metrics tracking.
        
        Args:
            save_model_checkpoints: If False, skips saving model checkpoints to prevent Google Drive spam.
                                  Only saves final results and metrics.
        
        Returns detailed training history for scientific analysis.
        """
        
        # Create criterion with training data if not already created
        if self.criterion is None:
            self.criterion = self._create_criterion(y_train)
        
        logger.info(f"Starting training: {self.experiment_name}")
        val_batches_info = f"{len(val_loader)}" if val_loader is not None else "None (CV mode)"
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {val_batches_info}")
        logger.info(f"Target epochs: {num_epochs}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader, epoch)
            else:
                # CV mode: use training metrics as validation (no separate validation set)
                val_metrics = train_metrics.copy()  # Use training performance for early stopping
            
            # Update learning rate
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['accuracy'])
            else:
                self.scheduler.step()
            
            # Record metrics
            epoch_time = time.time() - epoch_start_time
            self._record_epoch_metrics(train_metrics, val_metrics, epoch_time, epoch)
            
            # Early stopping (without model checkpointing unless explicitly enabled)
            if val_metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['accuracy']
                self.best_epoch = epoch
                self.early_stop_counter = 0
                
                # Only save checkpoint if explicitly requested
                if save_model_checkpoints and checkpoint_path:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'val_accuracy': val_metrics['accuracy'],
                        'epoch': epoch
                    }, checkpoint_path)
            else:
                self.early_stop_counter += 1
            
            # Progress logging
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}] - "
                    f"Train Acc: {train_metrics['accuracy']:.3f}, "
                    f"Val Acc: {val_metrics['accuracy']:.3f}, "
                    f"Best: {self.best_val_accuracy:.3f} (epoch {self.best_epoch+1})"
                )
            
            # Early stopping check
            patience = getattr(self.config, 'early_stop_patience', 20)
            if self.early_stop_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (patience: {patience})")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s")
        
        # Finalize training history
        self.training_history['total_training_time'] = total_time
        self.training_history['best_epoch'] = self.best_epoch
        self.training_history['best_val_accuracy'] = self.best_val_accuracy
        self.training_history['final_lr'] = self.optimizer.param_groups[0]['lr']
        
        # Only save training analysis (lightweight), not model checkpoints
        if save_detailed_history:
            self._save_training_analysis(checkpoint_path.parent if checkpoint_path else Path('.'))
        
        return self.training_history
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch with metrics tracking."""
        
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        gradient_norms = []
        
        for i, batch in enumerate(train_loader):
            # Handle device placement
            if self.model_type == 'multimodal':
                # Multimodal batch: (fmri_data, smri_data, targets)
                fmri_data, smri_data, targets = batch
                fmri_data = fmri_data.to(self.device, non_blocking=True)
                smri_data = smri_data.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                inputs = (fmri_data, smri_data)
            else:
                # Single modality batch: (inputs, targets)
                inputs, targets = batch
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision if available
            if self.scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    if self.model_type == 'multimodal':
                        outputs = self.model(fmri_data, smri_data)
                    else:
                        outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                self.scaler.scale(loss).backward()

                # Gradient clipping
                if hasattr(self.config, 'gradient_clip_norm'):
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_norm
                    )
                    gradient_norms.append(grad_norm.item())

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.model_type == 'multimodal':
                    outputs = self.model(fmri_data, smri_data)
                else:
                    outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                loss.backward()
                
                # Gradient clipping
                if hasattr(self.config, 'gradient_clip_norm'):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_norm
                    )
                    gradient_norms.append(grad_norm.item())
                
                self.optimizer.step()

            # Collect metrics
            total_loss += loss.item()
            
            with torch.no_grad():
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # AUC needs probabilities
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        balanced_accuracy = balanced_accuracy_score(all_targets, all_predictions)
        
        try:
            auc = roc_auc_score(all_targets, all_probabilities) if len(set(all_targets)) > 1 else 0.0
        except:
            auc = 0.0
        
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'auc': auc,
            'f1': f1,
            'gradient_norm': avg_grad_norm
        }
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch with metrics tracking."""
        
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for batch in val_loader:
                if self.model_type == 'multimodal':
                    fmri_data, smri_data, targets = batch
                    fmri_data = fmri_data.to(self.device)
                    smri_data = smri_data.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(fmri_data, smri_data)
                else:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        balanced_accuracy = balanced_accuracy_score(all_targets, all_predictions)
        
        try:
            auc = roc_auc_score(all_targets, all_probabilities) if len(set(all_targets)) > 1 else 0.0
        except:
            auc = 0.0
        
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        # Store detailed predictions for analysis
        self.detailed_predictions[f'epoch_{epoch}'] = {
            'targets': all_targets,
            'predictions': all_predictions,
            'probabilities': all_probabilities
        }
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'auc': auc,
            'f1': f1
        }
    
    def _record_epoch_metrics(
        self, 
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float],
        epoch_time: float,
        epoch: int
    ):
        """Record epoch metrics."""
        
        self.training_history['train_loss'].append(train_metrics['loss'])
        self.training_history['val_loss'].append(val_metrics['loss'])
        self.training_history['train_accuracy'].append(train_metrics['accuracy'])
        self.training_history['val_accuracy'].append(val_metrics['accuracy'])
        self.training_history['train_balanced_accuracy'].append(train_metrics['balanced_accuracy'])
        self.training_history['val_balanced_accuracy'].append(val_metrics['balanced_accuracy'])
        self.training_history['train_auc'].append(train_metrics['auc'])
        self.training_history['val_auc'].append(val_metrics['auc'])
        self.training_history['train_f1'].append(train_metrics['f1'])
        self.training_history['val_f1'].append(val_metrics['f1'])
        self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
        self.training_history['epoch_times'].append(epoch_time)
        self.training_history['gradient_norms'].append(train_metrics['gradient_norm'])
    
    def _save_training_analysis(self, save_dir: Path):
        """Save training analysis."""
        
        # Save training history as JSON
        history_path = save_dir / f'{self.experiment_name}_training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        # Save detailed predictions
        predictions_path = save_dir / f'{self.experiment_name}_detailed_predictions.json'
        with open(predictions_path, 'w') as f:
            json.dump(self.detailed_predictions, f, indent=2, default=str)
        
        logger.info(f"Training analysis saved to: {save_dir}")
    
    def evaluate_final(self, test_loader: DataLoader, checkpoint_path: Optional[Path] = None) -> Dict[str, float]:
        """
        Final evaluation with all metrics.
        Loads the best checkpoint if available before evaluation.
        """
        
        # Load best checkpoint before evaluation
        if checkpoint_path and checkpoint_path.exists():
            logger.info(f"Loading best checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from epoch {checkpoint['epoch']} with val_acc: {checkpoint['val_accuracy']:.3f}")
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                if self.model_type == 'multimodal':
                    fmri_data, smri_data, targets = batch
                    fmri_data = fmri_data.to(self.device)
                    smri_data = smri_data.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(fmri_data, smri_data)
                else:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        balanced_accuracy = balanced_accuracy_score(all_targets, all_predictions)
        
        try:
            auc = roc_auc_score(all_targets, all_probabilities) if len(set(all_targets)) > 1 else 0.0
        except:
            auc = 0.0
        
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        
        # Generate classification report
        class_report = classification_report(all_targets, all_predictions, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }

