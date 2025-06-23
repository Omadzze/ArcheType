#!/usr/bin/env python
import os
import json
import subprocess
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
import shutil

training_env = os.environ.copy()
#env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
training_env["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

inference_env = os.environ.copy()
inference_env["CUDA_VISIBLE_DEVICES"] = "6,7"

# ────────────────────────────────────────────────────────────────
#  Edit these to your own paths
TRAIN_SCRIPT = "/home/omadbek/projects/alpaca/train.py"
BASE_MODEL   = (
    "/home/omadbek/.cache/huggingface/hub/"
    "models--chavinlo--alpaca-native/"
    "snapshots/3bf09cbff2fbd92d7d88a0f70ba24fca372befdf"
)
DATA_PATH    = "/home/omadbek/projects/ArcheType/notebooks/output/full_train_finetune.json"
NOTEBOOK     = "test.ipynb"

#PAPERMILL_OUTPUT = "/home/omadbek/projects/ArcheType/papermill_notebooks/test_out_{name}.ipynb"

OUTPUT_WEIGHTS_ROOT = "/home/omadbek/projects/alpaca/outputs"

# Create root
PAPERMILL_ROOT = Path("/home/omadbek/projects/ArcheType/papermill_notebooks")

CUSTOM_DATA_LOGS = Path("/home/omadbek/projects/ArcheType/custom_data_logs")

for d in (PAPERMILL_ROOT, CUSTOM_DATA_LOGS):
    if d.exists():
        print(f"Cleaning old directory: {d}")
        shutil.rmtree(d)


PAPERMILL_ROOT.mkdir(parents=True, exist_ok=True)

#  Fixed train args (we’ll prepend with conda run -n alpaca)

# DEFAULT

TRAIN_ARGS = [
    "torchrun",
    "--nproc_per_node=4",
    "--rdzv_backend=c10d",
    "--rdzv_endpoint=127.0.0.1:29500",
    TRAIN_SCRIPT,
    "--model_name_or_path", BASE_MODEL,
    "--data_path", DATA_PATH,
    "--bf16", "True",
    "--per_device_train_batch_size", "2",
    "--per_device_eval_batch_size", "2",
    "--gradient_accumulation_steps", "16",
    "--eval_strategy", "no",
    "--save_strategy", "steps",
    "--save_steps", "2000",
    "--save_total_limit", "1",
    "--weight_decay", "0.0",
    "--warmup_ratio", "0.03",
    "--lr_scheduler_type", "cosine",
    "--logging_steps", "1",
    "--fsdp", "full_shard auto_wrap",
    "--fsdp_transformer_layer_cls_to_wrap", "LlamaDecoderLayer",
    "--tf32", "True",
    "--overwrite_output_dir",
]

#  Default notebook parameters (we’ll override model_dir and save_log)
NOTEBOOK_PARAMS = {
    "model_dir": None,
    "data_dir": "/home/omadbek/projects/ArcheType/notebooks/output",
    "save_log": None,
}

"""
# ────────────────────────────────────────────────────────────────
#  1) TRAIN only once with the best hyperparams
BEST_EPOCHS = 10
BEST_LR      = 2e-5
best_name    = f"e{BEST_EPOCHS}_lr{BEST_LR:.0e}"

train_cmd = (
        ["conda", "run", "-n", "alpaca"]
        + TRAIN_ARGS
        + [
            "--num_train_epochs", str(BEST_EPOCHS),
            "--learning_rate",      str(BEST_LR),
            "--output_dir",         OUTPUT_WEIGHTS_ROOT,
            "--run_name",           best_name,
        ]
)
print(f"\n→ TRAIN {best_name}")
subprocess.run(train_cmd, check=True, env=training_env)
"""
# ────────────────────────────────────────────────────────────────
#  2) INFERENCE N times to capture variability
N_RUNS = 10
inference_results = []

best_name = "llama-3.1"

for run_idx in range(1, N_RUNS + 1):
    run_id = f"{best_name}_run{run_idx}"
    pm_cmd = [
        "conda", "run", "-n", "archetype",
        "--no-capture-output",
        "papermill",
        NOTEBOOK,
        str(PAPERMILL_ROOT / f"test_out_{run_id}.ipynb"),
        "-p", "tune", run_id,
        # add any other -p overrides you need
    ]
    print(f"→ INFERENCE {run_id}")
    subprocess.run(pm_cmd, check=True, env=inference_env)

    # Optionally: after each run you could parse the notebook output
    # or a log file to extract F1 & accuracy, e.g.:
    # metrics = parse_metrics_from_log(f"/path/to/logs/{run_id}.json")
    # inference_results.append(metrics)

# ────────────────────────────────────────────────────────────────
print("\n=== ALL DONE! ===")

# Old code for the tuning model
"""
#  Your hyperparameter grid
GRID = [
    (3, 1e-5),
    (5, 1e-5),
    (8, 1e-5),
    (10, 1e-5),
    (15, 1e-5),
    (18, 1e-5),
    (20, 1e-5),
    (3, 2e-5),
    (5, 2e-5),
    (8, 2e-5),
    (10, 2e-5),
    (15, 2e-5),
    (18, 2e-5),
    (20, 2e-5),
]

results = []

for epochs, lr in GRID:
    name   = f"e{epochs}_lr{lr:.0e}"

    # ────────────────────────────────────────────────────────────
    
    # 1) TRAIN under alpaca env
    train_cmd = (
        ["conda", "run", "-n", "alpaca"]
        + TRAIN_ARGS
        + [
            "--num_train_epochs", str(epochs),
            "--learning_rate",      str(lr),
            "--output_dir",         OUTPUT_WEIGHTS_ROOT,
            "--run_name",           name,
        ]
    )
    print(f"\n→ TRAIN {name}")
    subprocess.run(train_cmd, check=True, env=training_env)
    
    # ────────────────────────────────────────────────────────────
    # 2) INFERENCE via Papermill under archetype env
    pm_cmd = [
        "conda", "run", "-n", "archetype",
        "--no-capture-output",
        "papermill",
        NOTEBOOK,
        str(PAPERMILL_ROOT / f"test_out_{name}.ipynb"),
        "-p", "tune", str(name),
        #"-p", "model_dir", outdir,
        #"-p", "data_dir", NOTEBOOK_PARAMS["data_dir"],
        #"-p", "save_log", logfn,
    ]
    print(f"→ INFERENCE {name}")
    subprocess.run(pm_cmd, check=True, env=inference_env)

# ────────────────────────────────────────────────────────────────
print("\n=== EVERYTHING IS DONE! ===")
"""