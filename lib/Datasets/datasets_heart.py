import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 暂时不用
class YourExcelDataset(Dataset):
    def __init__(self, user_file, imposter_file, train=True, transform=None, pca_components=3):
        self.transform = transform
        self.train = train
        self.pca_components = pca_components

        # 读取两个Excel文件
        user_df = pd.read_excel(user_file)
        imposter_df = pd.read_excel(imposter_file)

        # 合并数据，打乱顺序
        df = pd.concat([user_df, imposter_df], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # 选择特征列（0~23）作为输入
        selected_feature_columns = df.columns[df.columns.get_loc(0) : df.columns.get_loc('Class')].tolist()
        features = df[selected_feature_columns].values.astype('float32')
        labels = df['Class'].values.astype('int64')

        # 标准化
        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(features)

        # PCA降维
        self.pca = PCA(n_components=self.pca_components, random_state=42)
        features = self.pca.fit_transform(features)

        # 划分训练集/验证集（8:2分）
        split_idx = int(0.8 * len(features))
        if self.train:
            self.features = features[:split_idx]
            self.labels = labels[:split_idx]
        else:
            self.features = features[split_idx:]
            self.labels = labels[split_idx:]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
