import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from joblib import load

# 特征列
FEATURE_COLUMNS = [
    'url_length', 'digit_count', 'special_char_count', 
    'num_dots', 'hyphen_count', 'is_ip', 'num_subdomains',
    'registered', 'domain_age', 'domain_expiry', 'dns_valid',
    
    'num_links', 'num_images', 'num_forms', 'num_stylesheets',
    'num_meta_tags', 'title_length', 'num_iframes',
    'num_obfuscated_scripts',
    
    'protocol_http', 'protocol_https',
    
    'tld_BIZ', 'tld_BR', 'tld_CA', 'tld_COM', 'tld_DE', 'tld_FR',
    'tld_INFO', 'tld_JP', 'tld_LA', 'tld_MX', 'tld_NET', 'tld_ORG',
    'tld_RU', 'tld_TECH', 'tld_TH', 'tld_TR', 'tld_TW', 'tld_UK',
    'tld_ac', 'tld_ae', 'tld_africa', 'tld_ar', 'tld_au', 'tld_biz',
    'tld_ca', 'tld_cloud', 'tld_cn', 'tld_co', 'tld_com', 'tld_cz',
    'tld_de', 'tld_eu', 'tld_fr', 'tld_host', 'tld_id', 'tld_in',
    'tld_info', 'tld_io', 'tld_it', 'tld_jp', 'tld_kr', 'tld_mx',
    'tld_net', 'tld_nl', 'tld_org', 'tld_pk', 'tld_pl', 'tld_rs',
    'tld_ru', 'tld_se', 'tld_tr', 'tld_tw', 'tld_ua', 'tld_uk', 'tld_za'
]
device='cuda' if torch.cuda.is_available() else 'cpu'
# 定义双向LSTM模型
class FeatureLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64,
                 output_dim=1, num_layers=2):
        super(FeatureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, 
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # 添加序列维度
        output, (hidden, cell) = self.lstm(x)
        output = self.dropout(output[:, -1, :])
        return self.fc(output)

LSTMClassifier = FeatureLSTM
def load_pipeline(filepath, device='cpu'):
    model = load(filepath)
    model = model.to(device)
    model.eval()

    return model, None
    
    

def prepare_sample(sample_row):
    """处理单行样本数据"""
    fields = sample_row.strip().split(',')
    
    # 前两个字段是url和label
    url = fields[0]
    label = fields[1]
    feature_values = fields[2:]
    
    # 创建特征字典
    features = {}
    for col, val in zip(FEATURE_COLUMNS, feature_values):
        # 处理空字符串
        if val == '':
            features[col] = np.nan
        # 处理布尔值
        elif val.lower() in ['true', 'false']:
            features[col] = 1 if val.lower() == 'true' else 0
        # 处理数值型
        else:
            try:
                features[col] = float(val)
            except:
                features[col] = np.nan
    
    return pd.DataFrame([features])


def predict_single_sample(model, preprocessor, sample_df):
    """Predict using trained model"""

    # convert dataframe to numpy
    X = sample_df.values.astype(np.int64)

    # convert to tensor (Embedding requires LongTensor)
    X_tensor = torch.LongTensor(X).to(device)

    model.eval()
    with torch.no_grad():
        output = model(X_tensor)
        prob = torch.sigmoid(output).item()

    return (1 if prob > 0.5 else 0), prob

def predict_from_sample_row(
    sample_row,
    model_path="phishing_detector_lstm.joblib"
):
    # 加载模型和预处理
    model, preprocessor = load_pipeline(model_path, device=device)

    # 准备样本
    sample_df = prepare_sample(sample_row)

    # 进行预测
    pred_class, pred_prob = predict_single_sample(model, preprocessor, sample_df)

    return {
        'original_url': sample_row.split(',')[0],
        'probability': pred_prob,
        'prediction': 'phishing' if pred_class == 1 else 'legitimate',
        'class': pred_class
    }

if __name__ == "__main__":
    # 配置
    #model_path = "best_fold4.joblib"  # 选择其中一个fold的模型
    sample_row = "https://www.baidu.com,nan,21,0,5,2,0,0,1,1,9316,1276,nan,1,11,2,1,1,3,27,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"

    # 加载模型和预处理
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model, preprocessor = load_pipeline(model_path, device=device)

    # 准备样本
    #sample_df = prepare_sample(sample_row)

    # 进行预测
    #pred_class, pred_prob = predict_single_sample(model, preprocessor, sample_df)

    # 输出结果
    #print(f"预测概率: {pred_prob:.4f}")
    #print(f"预测类别: {'钓鱼网站' if pred_class == 1 else '正常网站'} (class={pred_class})")
    print(predict_from_sample_row(sample_row))