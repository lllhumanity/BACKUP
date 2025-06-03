# file: train_xgboost_gpu_external_memory.py

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # 添加缺失的指标导入
from pathlib import Path
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os  # 添加os模块用于路径操作

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
    # 添加缓存后缀优化数据加载
    dtrain = xgb.DMatrix(f"{train_file}?format=libsvm#dtrain.cache")
    dtest = xgb.DMatrix(f"{test_file}?format=libsvm#dtest.cache")

    # 优化GPU参数配置
    params = {
        "tree_method": "gpu_hist",  # 使用GPU直方图算法
        "gpu_id": 0,               # 指定GPU设备ID
        "predictor": "gpu_predictor",  # 添加GPU预测器
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_bin": 512,             # 优化GPU内存使用
        "single_precision_histogram": True  # 使用单精度加速
    }

    print("[INFO] Starting GPU training...")
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
    print(f"[INFO] GPU training completed in {elapsed_time:.2f} seconds")

    # 保存模型和最佳迭代
    bst.save_model(model_output)
    print(f"[INFO] Model saved to {model_output}")
    print(f"[INFO] Best iteration: {bst.best_iteration}")

    # 直接使用训练好的模型进行评估
    print("\n[Final Evaluation] Evaluating best model on test data...")
    predictions = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
    preds = (predictions > 0.5).astype(int)
    labels = dtest.get_label()

    # 计算评估指标
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    conf_matrix = confusion_matrix(labels, preds)
    class_report = classification_report(labels, preds)

    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = np.unique(labels)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('xgboost_confusion_matrix.png')
    plt.close()

    # 打印详细结果
    print("\n" + "="*50)
    print("FINAL MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nClassification Report:")
    print(class_report)

    print("\nConfusion Matrix:")
    print(conf_matrix)

    # 保存评估结果
    with open('xgboost_evaluation_results.txt', 'w') as f:
        f.write("FINAL MODEL EVALUATION RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(conf_matrix))

    print("Evaluation results saved to xgboost_evaluation_results.txt")
    print("Confusion matrix saved to xgboost_confusion_matrix.png")

if __name__ == "__main__":
    train_chunk_dir = "./train_data"
    test_chunk_dir = "./validate_data"

    train_libsvm = "train.libsvm"
    test_libsvm = "test.libsvm"
    model_file = "xgb_model.json"

    convert_chunks_to_libsvm(train_chunk_dir, train_chunk_dir, train_libsvm)
    convert_chunks_to_libsvm(test_chunk_dir, test_chunk_dir, test_libsvm)

    train_with_external_memory(train_libsvm, test_libsvm, model_output=model_file)