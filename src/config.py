from pathlib import Path
import torch

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "kaggle_3m"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINTS_DIR / "best_unet.pt"
METRICS_CSV_PATH = RESULTS_DIR / "metrics.csv"

# Ensure output directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# Data & Preprocessing
IMG_SIZE = (128, 128)
IN_CHANNELS = 3
NUM_CLASSES = 1
TEST_SIZE = 0.2      # 80% Train, 20% Val + Test
VAL_TEST_SPLIT = 0.5 # 10% Val, 10% Test
RANDOM_SEED = 42

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10
THRESHOLD = 0.5

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")