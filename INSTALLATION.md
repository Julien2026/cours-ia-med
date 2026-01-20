# 🛠️ Guide d'Installation

Configuration rapide pour le curriculum Python → IA Médicale

## 📋 Prérequis

- **OS:** Windows 10+, macOS 10.15+, Linux
- **RAM:** 8GB minimum
- **Stockage:** 10GB libre
- **Internet:** Pour télécharger datasets

## 🐍 Installation Python

### Option 1: Anaconda (Recommandé)

```bash
# 1. Télécharger Anaconda depuis https://www.anaconda.com/download
# 2. Installer et vérifier
conda --version

# 3. Créer environnement
conda create -n ia-medicale python=3.11
conda activate ia-medicale
```

### Option 2: Python Standard

```bash
# 1. Télécharger Python 3.9+ depuis https://python.org/downloads
# 2. Créer environnement virtuel
python -m venv ia-medicale-env

# Activer (Windows)
ia-medicale-env\Scripts\activate

# Activer (macOS/Linux)
source ia-medicale-env/bin/activate
```

## 📦 Installation Packages

### Essentiels
```bash
pip install numpy pandas matplotlib jupyter
pip install torch torchvision torchaudio
pip install scikit-learn opencv-python
```

### IA Médicale
```bash
pip install torchxrayvision nibabel
pip install nnunetv2
pip install transformers
```

## 🚀 Google Colab (Alternative)

Plus simple - aucune installation:

1. Aller sur https://colab.research.google.com
2. Se connecter avec Google
3. Uploader les notebooks
4. Activer GPU: Runtime → Change runtime → GPU

Configuration Colab:
```python
# Au début de chaque notebook
!pip install torchxrayvision nnunetv2

import torch
print(f"GPU: {torch.cuda.is_available()}")
```

## ✅ Test Installation

```python
# Copier dans test.py et exécuter
import numpy as np
import torch
import matplotlib.pyplot as plt

print("✅ Numpy:", np.__version__)
print("✅ PyTorch:", torch.__version__)
print("✅ GPU:", "Oui" if torch.cuda.is_available() else "Non")

try:
    import torchxrayvision as xrv
    print("✅ TorchXRayVision installé")
except:
    print("⚠️ TorchXRayVision manquant")

print("\n🎯 Prêt à commencer!")
```

## ❓ Problèmes Courants

**GPU non détecté:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Package manquant:**
```bash
pip install nom-du-package
```

**Jupyter ne démarre pas:**
```bash
pip install --upgrade jupyter notebook
jupyter notebook
```

## 📞 Support

- **Problèmes techniques:** GitHub Issues
- **Contact:** Emmanuel Noutahi, PhD

---

**Conseil:** Commencez avec Google Colab si vous débutez!