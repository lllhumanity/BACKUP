# file: train_xgboost_gpu_external_memory.py

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from pathlib import Path
from tqdm import tqdm
import time


def reshape_features(features: np.ndarray) -> np.ndarray:
    return features.reshape(features.shape[0], -1)  # (n, 257, 4) -> (n, 1028)


def save_to_libsvm(features: np.ndarray, labels: np.ndarray, output_path: Path, mode='a'):
    with output_path.open(mode) as f:
        for i in range(features.shape[0]):
            label = labels[i]
            row = features[i]
            row_str = " ".join([f"{j + 1}:{val}" for j, val in enumerate(row) if val != 0])
            f.write(f"{label} {row_str}\n")


def convert_chunks_to_libsvm(feature_chunk_dir: str, label_chunk_dir: str, output_libsvm: str):
    print(f"[INFO] Starting conversion to libsvm format: {output_libsvm}")
    output_path = Path(output_libsvm)
    if output_path.exists():
        output_path.unlink()

    fx0 = Path(feature_chunk_dir) / "chunk_0.fX.npy"
    fx1 = Path(feature_chunk_dir) / "chunk_1.fX.npy"
    fy0 = Path(label_chunk_dir) / "chunk_0.fY.npy"
    fy1 = Path(label_chunk_dir) / "chunk_1.fY.npy"

    fx_files = [fx0, fx1]
    fy_files = [fy0, fy1]

    for fx_file, fy_file in tqdm(zip(fx_files, fy_files), total=2, desc="Processing Chunks"):
        print(f"[INFO] Processing chunk files: {fx_file.name} & {fy_file.name}")
        X = np.load(fx_file)
        y = np.load(fy_file)
        X_flat = reshape_features(X)
        save_to_libsvm(X_flat, y, output_path, mode='a')
    print(f"[INFO] Finished conversion: {output_libsvm}\n")


def train_with_external_memory(train_file: str, test_file: str, model_output: str = "xgb_model.json"):
    print(f"[INFO] Loading training and testing data...")
    dtrain = xgb.DMatrix(f"{train_file}?format=libsvm#train.cache")
    dtest = xgb.DMatrix(f"{test_file}?format=libsvm#test.cache")

    params = {
        "tree_method": "gpu_hist",
        "gpu_id": 0,
        "objective": "binary:logistic",
        "eval_metric": "logloss"
    }

    print("[INFO] Starting training...")
    start_time = time.time()

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtest, 'test')],
        early_stopping_rounds=10,
        verbose_eval=1
    )

    elapsed_time = time.time() - start_time
    print(f"[INFO] Training completed in {elapsed_time:.2f} seconds")

    # Save model and best iteration
    bst.save_model(model_output)
    print(f"[INFO] Model saved to {model_output}")
    print(f"[INFO] Best iteration: {bst.best_iteration}")

    print("[INFO] Evaluating test accuracy...")
    preds = (bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1)) > 0.5).astype(int)
    labels = dtest.get_label()
    acc = accuracy_score(labels, preds)
    print(f"[INFO] Test Accuracy: {acc:.4f}")


if __name__ == "__main__":
    # Update these directories
    train_chunk_dir = "./train_data"
    test_chunk_dir = "./validate_data"

    train_libsvm = "train.libsvm"
    test_libsvm = "test.libsvm"
    model_file = "xgb_model.json"

    #convert_chunks_to_libsvm(train_chunk_dir, train_chunk_dir, train_libsvm)
    #convert_chunks_to_libsvm(test_chunk_dir, test_chunk_dir, test_libsvm)

    train_with_external_memory(train_libsvm, test_libsvm, model_output=model_file)
