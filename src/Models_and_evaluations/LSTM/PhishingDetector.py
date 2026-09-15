import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt
from joblib import dump, load
from torch.utils.tensorboard import SummaryWriter
import tqdm

# 加载数据集
df = pd.read_csv('Secure_Cyber_Systems/data/final_dataset_with_features.csv')

# 准备X和y
X = df.drop(['url', 'label'], axis=1)
y = df['label']

# 检查并转换非数值列
for col in X.columns:
    if X[col].dtype == 'object':
        print(f"Column {col} is non-numeric. Converting to numeric.")
        X[col] = pd.to_numeric(X[col], errors='coerce')

# 处理缺失值
X = X.fillna(X.mean())

# 标准化数值特征
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 分割训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 转换为张量
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val.values, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

# 检查类别分布
num_pos = y_train.sum().item()
num_neg = len(y_train) - num_pos
pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
print(f"正样本数量：{num_pos}")
print(f"负样本数量：{num_neg}")
print(f"正样本权重：{pos_weight}")

# 定义模型
class PhishingDetector(nn.Module):
    def __init__(self, input_dim):
        super(PhishingDetector, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)

# 初始化模型
input_dim = X_train.shape[1]
model = PhishingDetector(input_dim)

# 不再使用 DataParallel，避免批量大小为1的问题
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器
pos_weight = torch.tensor([pos_weight], device=device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

# 定义数据加载器
batch_size = 2048  # 根据 GPU 内存调整
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 使用 TensorBoard
log_dir = 'Secure_Cyber_Systems/phishing_detector/logs'
writer = SummaryWriter(log_dir=log_dir)

# 训练循环，包含早停
num_epochs = 50
patience = 5
best_auc = 0
counter = 0
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    writer.add_scalar('Training Loss', train_loss, epoch)

    # 验证
    model.eval()
    all_probs = []
    all_labels = []
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            val_loss += loss.item()
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.cpu().numpy())
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    writer.add_scalar('Validation Loss', val_loss, epoch)

    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten().astype(int)
    all_preds = (all_probs > 0.5).astype(int)
    acc = accuracy_score(all_labels, all_preds)
    val_accuracies.append(acc)
    writer.add_scalar('Validation Accuracy', acc, epoch)
    auc = roc_auc_score(all_labels, all_probs)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {acc:.4f}, Val AUC: {auc:.4f}')
    scheduler.step(auc)

    if auc > best_auc:
        best_auc = auc
        counter = 0
        # 保存最佳模型
        model.cpu()
        dump(model, 'Secure_Cyber_Systems/phishing_detector/best_model.joblib')
        model.to(device)
    else:
        counter += 1
        if counter >= patience:
            print('Early stopping')
            break

# 关闭 TensorBoard writer
writer.close()

# 绘制训练和验证的 Loss 曲线
plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('Secure_Cyber_Systems/phishing_detector/loss_curve.png')
plt.show()

# 绘制验证的 Accuracy 曲线
plt.figure()
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.savefig('Secure_Cyber_Systems/phishing_detector/accuracy_curve.png')
plt.show()

# 模型评估
model.eval()
all_predictions = []
all_labels = []
all_probs = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        outputs = model(X_batch)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        probs = torch.sigmoid(outputs)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# 混淆矩阵
cm = confusion_matrix(all_labels, all_predictions)
print("Confusion Matrix:")
print(cm)

# 精确率和召回率
precision = precision_score(all_labels, all_predictions)
recall = recall_score(all_labels, all_predictions)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# F1 分数
f1 = f1_score(all_labels, all_predictions)
print(f"F1 Score: {f1:.4f}")

# ROC 曲线和 AUC
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
auc = roc_auc_score(all_labels, all_probs)

# 使用 TensorBoard 记录 ROC 曲线
writer = SummaryWriter(log_dir=log_dir)
writer.add_pr_curve('ROC Curve', torch.tensor(all_labels), torch.tensor(all_probs), global_step=epoch)
writer.close()

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('Secure_Cyber_Systems/phishing_detector/roc_curve.png')
plt.show()

# 保存模型为 .joblib 文件
model.cpu()
dump(model, 'Secure_Cyber_Systems/phishing_detector/best_model.joblib')
print("Model saved as Secure_Cyber_Systems/phishing_detector/best_model.joblib")