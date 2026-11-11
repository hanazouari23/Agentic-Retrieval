# Evaluation queries
**50 queries from topics.ms-marco-dev.tsv :**
1093322, 922335, 999637, 135386, 1007550, 1086075, 1032011, 384406, 472024, 780613, 1099816, 1098860, 818798, 1003997, 42568, 1100167, 704072, 1022178, 499568, 1059601, 862742, 169778, 944700, 34015, 987230, 359040, 526671, 756949, 453705, 984434, 833268, 1080537, 1087105, 1063177, 1090987, 952388, 277799, 1073721, 1094191, 541135, 769630, 1095942, 523621, 1067826, 1036800, 1091688, 527769, 266760, 1083945, 989296
- For evaluation, the retrieval corpus has a document granularity. Also the topk retrieval window is 50 but could be less if that quantity is not met. We had to switch to a document index because the segmented index is unstable and keep returning 500 error code for most queries. 
# Reference Training Config
**Reward:**  α =2.0, β = 0.5, γ = 0.2, and δ = 0.1.
**Training Hyperparameters:** 
RANDOM_SEED = 42
HIDDEN_DIMS = (256, 128)
BATCH_SIZE = 256
GAMMA = 0.99
LEARNING_RATE = 3e-4
N_CRITICS = 2
TARGET_UPDATE_INTERVAL = 8000
CQL_ALPHA = 1.0
N_STEPS = 50_000
N_STEPS_PER_EPOCH = 10_000  