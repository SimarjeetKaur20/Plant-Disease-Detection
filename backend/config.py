"""
config.py
Central configuration for the PlantDiseaseDetection project.
All hyper-parameters, paths and constants live here so every other
module can simply `from config import CFG`.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "dataset", "PlantVillage")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
]
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------
IMAGE_SIZE = (160, 160)   # (H, W) fed to both teacher and student
MEAN         = [0.485, 0.456, 0.406]   # ImageNet stats
STD          = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Teacher training
# ---------------------------------------------------------------------------
TEACHER_MODEL_NAME   = "resnet18"          # torchvision model key
TEACHER_PRETRAINED   = True
TEACHER_EPOCHS       = 5    
TEACHER_BATCH_SIZE   = 16
TEACHER_LR           = 1e-4
TEACHER_WEIGHT_DECAY = 1e-4
TEACHER_SCHEDULER    = "cosine"            # "cosine" | "step"
TEACHER_SAVE_PATH    = os.path.join(MODEL_DIR, "teacher_best.pth")

# ---------------------------------------------------------------------------
# Student training (Knowledge Distillation)
# ---------------------------------------------------------------------------
STUDENT_MODEL_NAME   = "mobilenet_v2"      # lightweight backbone
STUDENT_PRETRAINED   = True
STUDENT_EPOCHS       = 5
STUDENT_BATCH_SIZE   = 16
STUDENT_LR           = 1e-3
STUDENT_WEIGHT_DECAY = 1e-5
STUDENT_SAVE_PATH    = os.path.join(MODEL_DIR, "student_best.pth")

# Knowledge Distillation hyper-params
KD_TEMPERATURE = 4.0    # softens teacher logits
KD_ALPHA       = 0.7    # weight of distillation loss  (1-alpha → hard CE loss)

# ---------------------------------------------------------------------------
# Evaluation / inference
# ---------------------------------------------------------------------------
DEVICE      = "cuda"     # "cuda" | "cpu"  (auto-detected in utils.py)
NUM_WORKERS = 4
