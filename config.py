from pathlib import Path

RANDOM_STATE = 420

DATA_DIR = Path(".")
TEST_DIR = DATA_DIR / "test_series"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"

N_SPLITS = 5

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
           'Contusion', 'Fracture']

ALL_SLOTS = [
    ("Sagittal", 1, 1),
    ("Coronal",  1, 1),
    ("Axial",    1, 1),
    ("Sagittal", 0, 1),
    ("Coronal",  0, 0),
    ("Sagittal", 0, 0),
]

DROP_IDXS = {5}
KEEP_IDXS = [i for i in range(len(ALL_SLOTS)) if i not in DROP_IDXS]
SLOTS = [ALL_SLOTS[i] for i in KEEP_IDXS]
WINDOWS_BY_SLOT = {
    0: 12, 
    1: 12, 
    2: 12, 
    3: 12, 
    4: 12, 
    5: 12,
}
WINDOW_SIZE = 3
WINDOWS_PER_SLOT = 12
OUT_SIZE = 224
CROP_MM = 140.0

UNFREEZE_LAST = 10

AUGMENT_FRACTION = 0.80

LR = 3e-4
BATCH_SIZE = 4
EPOCHS = 20
WEIGHT_DECAY = 5e-2
CKPT_PATH = "best_model.pt"

PATIENCE = 5
