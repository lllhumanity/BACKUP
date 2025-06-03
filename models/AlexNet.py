import toml
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm
import os
from sklearn.metrics import fbeta_score
from scipy.stats import pearsonr

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
        #print(f'[Info] load features finished! features shape: {features.shape}, labels shape: {labels.shape}')


def load_data(outdir: str, batch_size: int, ndp: int):
    chunk_idx = 0
    feature_chunk_file = os.path.join(outdir, f"chunk_{chunk_idx}.fX.npy")
    label_chunk_file = os.path.join(outdir, f"chunk_{chunk_idx}.fy.npy")
    dataset = classificationIterableDataset(feature_chunk_file, label_chunk_file, batch_size)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=4)
    print(f'[Info] Data loader initialized for streaming data!')
    return dataloader

class AlexNet1d(nn.Module):
    '''
    this class defines the AlexNet model
    '''
    def __init__(self, num_class=2):
        super(AlexNet1d, self).__init__()
        
        # feature extraction layers
        self.features = nn.Sequential(
            nn.Conv1d(4, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            
            nn.Conv1d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),

            nn.Conv1d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        
        #auto pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool1d(6)
        
        #classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_class),
        )
    
    def forward(self, x):
        #input shape: (batch_size, 4, 257)
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    
def initialize_model(m):
    '''
    this function is meant for layers' weight initialization
    '''
    for m in m.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def calculate_metrics(all_labels, all_preds, all_probs):
    '''
    this function calculates the f2 score and pearson correlation coefficient for the model's predictions.
    '''
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
    model = AlexNet1d(
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
        
        total_samples = 0
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            pbar_test = tqdm(val_dataloader, desc=f"Validating Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (X, y) in enumerate(pbar_test):
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
                
                current_loss = test_loss / total_samples
                pbar_test.set_postfix({"loss": current_loss})
            pbar_test.close()
        
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)
        
        metrics = calculate_metrics(all_labels, all_preds, all_probs)
        
        train_loss = train_loss / ((batch_idx + 1) * batch_size)
        test_loss = test_loss / ((batch_idx + 1) * batch_size)
        accuracy = metrics['accuracy']
        f2_score = metrics['f2_score']
        pcc = metrics['pcc']
        
        
        print(f"Epoch {epoch+1:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Accuracy: {accuracy:.4f} | "
                f"F2 Score: {f2_score:.4f} | "
                f"Pearson CC: {pcc:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "Alex_best_model.pth")
            print(f"[saving] Epoch {epoch+1:03d} accuracy to {best_accuracy:.4f}")
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_accuracy': best_accuracy,
            'f2_score': f2_score,
            'pcc': pcc
        }, checkpoint_path)
        
        # 内存回收
        #import gc
        #gc.collect()
        #torch.cuda.empty_cache()
            
    return model


if __name__ == "__main__":
    model = train_model(
        num_epochs=100,
        batch_size = 25600,
        checkpoint_path = "checkpoint.pth"
    )
    torch.save(model.state_dict(), "Alex_final_model.pth")
                        
           
    