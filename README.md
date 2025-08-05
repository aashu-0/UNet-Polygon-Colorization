# Polygon Colorization with Conditional UNet

## Installation

**Installing uv**

If you don't have `uv` installed, install it first:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: Install via pip
pipx install uv
```

**Setup**

1. **Clone the repository**
```bash
git clone https://github.com/aashu-0/UNet-Polygon-Colorization.git
cd polygon-colorization
```

2. **Create virtual environment**
```bash
uv venv
# Activate environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install project dependencies**
```bash
uv sync
```

4. **Setup WandB (for experiment tracking)**
```bash
# Get API key from wandb.ai/settings
wandb login
```

## Project Structure

```
UNet-Polygon-Colorization/
├── pyproject.toml           # Project dependencies and configuration
├── README.md             
├── model.py          # UNet architecture with color conditioning
├── train.py               # Training script with WandB integration
├── inference_notebook.ipynb        # Inference and testing notebook
├── dataset/               # Training and validation data
│   ├── training/
│   │   ├── inputs/        # Polygon outline images
│   │   ├── outputs/       # Colored polygon images
│   │   └── data.json      # Training data mappings
│   └── validation/
│       ├── inputs/        # Validation polygon images
│       ├── outputs/       # Validation colored images
│       └── data.json      # Validation data mappings
└── best_model.pth         # Best trained model (created after training)
```

**Training Configuration:**
- Batch Size: 8
- Learning Rate: 1e-4
- Epochs:200
- Optimizer: AdamW with weight decay
- Loss: MSE Loss

## Configuration

**Training Parameters (train.py):**
```python
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200
IMG_SIZE = 256
PATIENCE = 20  # Early stopping patience
```

**Model Parameters:**
```python
n_channels = 3      # Input channels (RGB)
n_classes = 3       # Output channels (RGB)
num_colors = 8      # Number of supported colors
bilinear = True     # bilinear upsampling
```
## WandB Project
The training runs are tracked in WandB project: [polygon-colorization](https://api.wandb.ai/links/aashu-0-mnit/bi08dcaw)

## Inference Notebook
Inference notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fU6WD7i2Pd17AmyYiy0tdk531-J-2yl8?usp=sharing)