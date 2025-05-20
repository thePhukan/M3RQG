# M3RQG

Official implementation for the M3RQG framework, exploring multi-modal multi-hop question generation using a multi-decoder setup.

## 📥 Installation

1. Clone the repository:
```bash
git clone https://github.com/thePhukan/M3RQG.git
cd m3rqg

conda env create -f environment_webqaqwen.yml
conda activate webqaqwen

.
├── bart_md.py            # Bart-Large baseline + M3RQG implementation
├── md_phi.py             # Phi-3.5-Vision model with M3RQG
├── md_llava.py           # LLaVA model with M3RQG
├── 1d_llava.py           # LLaVA ablation (single decoder)
├── 1d_phi.py             # Phi-3.5-Vision ablation (single decoder)
├── dataset_split/
│   ├── dataset_train_gem_gpt4.xlsx # Sample of training data
│   └── dataset_val_gem_gpt4.xlsx   # Sample of validation data
│   └── dataset_test_gem_gpt4.xlsx   # Sample of test data
└── environment_webqaqwen.yml # Conda environment specification
