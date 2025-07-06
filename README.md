# Cross-Attention Between Functional and Structural MRI for Autism Classification

This repository contains the code for my bachelor's thesis project exploring cross-attention mechanisms between structural MRI (sMRI) and functional MRI (fMRI) for autism spectrum disorder classification using the ABIDE dataset.

## Details
Author: Stanisław Wasilewski    
Supervised by: Bob Borsboom     
Thesis content: [paper.pdf](paper.pdf)     
Institution: Vrije Universiteit Amsterdam

## Thesis Abstract

Autism Spectrum Disorder (ASD) affects 1 in 127 people
worldwide, yet current diagnostic methods rely on behavioral assess-
ments that are time-consuming and prone to bias. While machine learn-
ing approaches using neuroimaging data show promise, many existing
models suffer from poor generalizability and evaluation methodologies
that may inflate performance estimates. This thesis presents the first
systematic investigation of bidirectional cross-attention between func-
tional and structural MRI for ASD classification, exploring how different
tokenization strategies affect model performance and generalizability.    
    
This study develops eleven Transformer-based architectures that imple-
ment cross-attention mechanisms between fMRI connectivity patterns
and sMRI structural features, using various tokenization approaches in-
cluding ROI-based and network-based strategies. All models were evalu-
ated using both standard k-fold cross-validation and rigorous leave-one-
out cross-validation on 871 subjects from the ABIDE dataset to assess
true generalizability across acquisition sites.    
    
Results demonstrate that cross-attention consistently outperforms single-
modality baselines, achieving 69.9% accuracy compared to 63.5% for
fMRI-only models. However, complex tokenization strategies that im-
proved performance under standard evaluation (70.1% peak accuracy)
showed limited benefits under leave-one-out validation, revealing signif-
icant site bias effects that inflate traditional performance estimates. All
approaches converged around a 70% performance ceiling, suggesting fun-
damental limitations in current neuroimaging-based ASD classification.
The findings highlight the critical importance of rigorous cross-site eval-
uation in neuroimaging research and demonstrate that while multimodal
integration provides meaningful improvements, current approaches may
be approaching their discriminatory limits for ASD classification.


## Installation

### Requirements
- Python 3.7+
- PyTorch
- scikit-learn
- numpy, pandas
- nilearn
- Additional dependencies in `requirements.txt`

### Google Colab Setup
```python
# Clone repository
!git clone [repository-url]
%cd thesis-in-progress

# Install dependencies
!pip install -r requirements.txt

# Mount Google Drive for data access
from google.colab import drive
drive.mount('/content/drive')
```


## Usage

### System Validation
```bash
# Validate system setup
python scripts/thesis_experiments.py --validate_setup
```

### Running Experiments

**Quick Pipeline Test:**
```bash
# Test training pipeline (~3 minutes)
python scripts/thesis_experiments.py --quick_pipeline_test
```

**Individual Experiments:**
```bash
# List available experiments
python scripts/thesis_experiments.py --list_experiments

# Test specific experiment
python scripts/thesis_experiments.py --test_single fmri_baseline
```

**Full Experimental Suite:**
```bash
# Run all experiments (2-6 hours)
python scripts/thesis_experiments.py --run_all
```

**Experiment Categories:**
```bash
# Baseline experiments only
python scripts/thesis_experiments.py --baselines_only

# Cross-attention experiments only
python scripts/thesis_experiments.py --cross_attention_only

# Tokenized experiments only
python scripts/thesis_experiments.py --tokenized_only
```

### Available Experiments
1. **fmri_baseline**: fMRI-only classifier
2. **smri_baseline**: sMRI-only classifier
3. **cross_attention**: Basic multimodal fusion
4. **tokenized_fmri_functional**: Functional network tokenization
5. **tokenized_fmri_anatomical**: Anatomical network tokenization
6. **tokenized_smri_features**: Feature-type tokenization
7. **tokenized_smri_networks**: Network-based tokenization
8. **tokenized_cross_attention**: Combined tokenization strategies

## Configuration

Configuration files are in `configs/default_experiment_config.json`. Key parameters:
- Learning rates, batch sizes, epochs
- Cross-validation settings
- Model architecture parameters
- Tokenization strategies

## Output Files

Results are saved to:
- `results/complete_thesis_results.json`: Complete experimental results
- `results/detailed_performance_summary.csv`: Performance metrics
- `results/[experiment_name]/`: Individual experiment outputs