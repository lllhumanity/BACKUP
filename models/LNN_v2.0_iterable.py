import toml
import torch
import numpy as np
from sklearn.metrics import fbeta_score
from scipy.stats import pearsonr
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm
import os

batch_size = 25600
train_data_dir = "train_data"
test_data_dir = "validate_data"
checkpoint_path = "checkpoint.pth"

class classificationIterableDataset(IterableDataset):
    def __init__(self, feature_chunk_file, label_chunk_file, batch_size):
        self.feature_chunk_file = feature_chunk_file
        self.label_chunk_file = label_chunk_file
        self.batch_size = batch_size

    def __iter__(self):
        fX_chunk = np.load(self.feature_chunk_file, mmap_mode='r')
        fy_chunk = np.load(self.label_chunk_file, mmap_mode='r')
        start = 0
        while start < len(fy_chunk):
            end = start + self.batch_size
            features = torch.from_numpy(fX_chunk[start:end].copy()).float()
            labels = torch.from_numpy(fy_chunk[start:end].copy()).long()
            yield features, labels
            start = end
        print(f'[Info] load features finished! features shape: {features.shape}, labels shape: {labels.shape}')


def load_data(outdir: str, batch_size: int, ndp: int):
    chunk_idx = 0
    feature_chunk_file = os.path.join(outdir, f"chunk_{chunk_idx}.fX.npy")
    label_chunk_file = os.path.join(outdir, f"chunk_{chunk_idx}.fy.npy")
    dataset = classificationIterableDataset(feature_chunk_file, label_chunk_file, batch_size)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=4)
    print(f'[Info] Data loader initialized for streaming data!')
    return dataloader

class LNNmodelv2(nn.Module):
    '''
    this class defines the LNN v2 model, a most simple LSTM model with 2 layers and 128 hidden units.
    '''
    def __init__(self, device, input_size, hidden_size=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )
    
    def forward(self, x):
        return self.net(x)
    
def calculate_metrics(all_labels, all_preds, all_probs):
    results = {}
    
    #1. acc
    results['accuracy']=(all_labels == all_preds).mean()
    #2. f2 score
    try:
        results['f2_score'] = fbeta_score(all_labels, all_preds, beta=2, average='macro')
    except ValueError as e:
        results['f2_score'] = 0.0
        print(f"[Error] f2 score calculation failed: {e}")
    #3. pearson correlation coefficient
    try:
        flat_probs = all_probs.reshape(-1)
        repeated_labels = np.repeat(all_labels, all_probs.shape[1])
        pcc, _ = pearsonr(flat_probs, repeated_labels)
        results['pcc'] = pcc
    except ValueError as e:
        results['pcc'] = 0.0
        print(f"[Error] Pearson correlation calculation failed: {e}")
    return results

def initialize_model(m):
    '''
    this function is meant for linear layers' weight initialization
    '''
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)

def train_model(
    num_epochs=10,
    batch_size = 256,
    checkpoint_path = "checkpoint.pth"
):
    '''
    this function trains the model on the training data, and evaluates the model on the test data.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device info] current using device: {device}")
    if device.type == 'cuda':
        print(f"[GPU name] {torch.cuda.get_device_name(device)}")
    
    #1. model initialization
    model = LNNmodelv2(
        device=device,
        input_size = 1028,
        hidden_size = 128,
        dropout = 0.1
    )
    start_epoch = 0
    best_accuracy = 0.0
    criterion = nn.CrossEntropyLoss()                          #cross entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  #adam optimizer                                            
    model.apply(initialize_model)                               #initialize model weight
    
    #2. load validate and train data once
    print(f"loading validate data")
    val_dataloader = load_data("validate_data", batch_size=batch_size, ndp=840484)
    print(f"validate data loaded")
    print(f"loading train data")
    train_dataloader = load_data("train_data", batch_size=batch_size, ndp=842057)
    print(f"train data loaded")
      
    #3. ckpt loading
    if os.path.exists(checkpoint_path):
        train_from_ckpt = True
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        # 将优化器的状态字典中的张量移动到正确的设备上
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        print(f"[checkpoint loaded] loaded checkpoint from epoch {start_epoch}, best accuracy: {best_accuracy}")
    
    #4. torch profiler and training
    print(f"[info]start training")
    for epoch in range(start_epoch, num_epochs):
        model.to(device)
        model.train()
        train_loss = 0.0
        print(f"epoch {epoch+1} start")
        print(f"loading train data")
        pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (X, y) in enumerate(pbar):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            pbar.set_postfix({"loss": loss.item()})
        pbar.close()  # 关闭进度条
        
        #5. evaluate model on validate data
        model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            pbar_test = tqdm(val_dataloader, desc=f"Validating Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (X, y) in enumerate(pbar_test):
                X, y = X.to(device), y.to(device)
                with torch.autocast(device_type=device.type):
                    outputs = model(X)                    
                    test_loss += criterion(outputs, y).item() * X.size(0)
                    correct += (torch.argmax(torch.softmax(outputs, dim = -1), dim=-1) == y).sum().item()
                pbar_test.set_postfix({"loss": test_loss / ((batch_idx + 1) * batch_size)})
            pbar_test.close()
        
        train_loss = train_loss / ((batch_idx + 1) * batch_size)
        test_loss = test_loss / ((batch_idx + 1) * batch_size)
        accuracy = correct / ((batch_idx + 1) * batch_size)
        
        print(f"Epoch {epoch+1:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "LNN_best_model.pth")
            print(f"[saving] Epoch {epoch+1:03d} accuracy to {best_accuracy:.4f}")
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_accuracy': best_accuracy
        }, checkpoint_path)
        
        # 内存回收
        #import gc
        #gc.collect()
        #torch.cuda.empty_cache()
            
    return model


if __name__ == "__main__":
    model = train_model(
        num_epochs=150,
        batch_size = 25600,
        checkpoint_path = "checkpoint.pth"
    )
    torch.save(model.state_dict(), "LNN_final_model.pth")

# 6. 训练结束后评估最佳模型
print("\n[Final Evaluation] Evaluating best model on test data...")

# 加载最佳模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if os.path.exists("LNN_best_model.pth"):
    model.load_state_dict(torch.load("LNN_best_model.pth", map_location=device))
    print("Loaded best model for evaluation")
else:
    print("Warning: Best model not found. Using last trained model for evaluation")

# 创建测试数据加载器
test_data_dir = "test_data"  # 替换为实际的测试数据目录
if not os.path.exists(test_data_dir):
    print(f"Warning: Test data directory '{test_data_dir}' not found. Using validation data for evaluation")
    test_data_dir = "validate_data"

test_dataloader = load_data(test_data_dir, batch_size=batch_size, ndp=0)

# 详细评估最佳模型
model.eval()
test_loss = 0.0
total_samples = 0

all_labels = []
all_preds = []
all_probs = []
all_outputs = []

with torch.no_grad():
    pbar = tqdm(test_dataloader, desc="Evaluating Model")
    for batch_idx, (X, y) in enumerate(pbar):
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        loss = criterion(outputs, y)
        test_loss += loss.item() * X.size(0)

        probs = torch.softmax(outputs, dim=-1)
        _, preds = torch.max(outputs, dim=-1)

        total_samples += X.size(0)
        all_labels.append(y.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_outputs.append(outputs.cpu().numpy())

all_labels = np.concatenate(all_labels)
all_preds = np.concatenate(all_preds)
all_probs = np.concatenate(all_probs)
all_outputs = np.concatenate(all_outputs)

metrics = calculate_metrics(all_labels, all_preds, all_probs)

if criterion is not None:
    test_loss = test_loss / total_samples
    metrics['loss'] = test_loss

# 计算额外指标
metrics['classification_report'] = classification_report(all_labels, all_preds, output_dict=True)
metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds)

# 可视化混淆矩阵
plt.figure(figsize=(10, 8))
plt.imshow(metrics['confusion_matrix'], interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

classes = np.unique(all_labels)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

thresh = metrics['confusion_matrix'].max() / 2.
for i in range(metrics['confusion_matrix'].shape[0]):
    for j in range(metrics['confusion_matrix'].shape[1]):
        plt.text(j, i, format(metrics['confusion_matrix'][i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if metrics['confusion_matrix'][i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('LNN_confusion_matrix.png')
plt.close()

# 打印详细结果
print("\n" + "="*50)
print("FINAL MODEL EVALUATION RESULTS")
print("="*50)
print(f"Test Loss: {metrics['loss']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F2 Score: {metrics['f2_score']:.4f}")
print(f"Pearson CC: {metrics['pcc']:.4f}")

# 打印分类报告
print("\nClassification Report:")
report = metrics['classification_report']
for label in report:
    if label in ['0', '1']:  # 假设是二分类
        print(f"Class {label}: Precision={report[label]['precision']:.4f}, Recall={report[label]['recall']:.4f}, F1-score={report[label]['f1-score']:.4f}")

# 打印混淆矩阵
print("\nConfusion Matrix:")
print(metrics['confusion_matrix'])

# 保存评估结果
with open('LNN_evaluation_results.txt', 'w') as f:
    f.write("FINAL MODEL EVALUATION RESULTS\n")
    f.write("="*50 + "\n")
    f.write(f"Test Loss: {metrics['loss']:.4f}\n")
    f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
    f.write(f"F2 Score: {metrics['f2_score']:.4f}\n")
    f.write(f"Pearson CC: {metrics['pcc']:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(all_labels, all_preds))
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(metrics['confusion_matrix']))

print("Evaluation results saved to LNN_evaluation_results.txt")
print("Confusion matrix saved to LNN_confusion_matrix.png")
                        
           
    