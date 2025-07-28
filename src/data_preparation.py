import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import cloudpickle

laptop_train = pd.read_csv('data//laptop_train_set.csv')
X = laptop_train.drop('Price', axis=1)
y = laptop_train['Price']

class LaptopFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Nothing to learn here
        return self

    def transform(self, df):
        df = df.copy()
        df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
        df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
        df['Inches'] = df['Inches'].astype(float)

        df['Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen', case=False).astype(int)
        df['IPS'] = df['ScreenResolution'].str.contains('IPS', case=False).astype(int)
        df['Resolution'] = df['ScreenResolution'].str.extract(r'(\d+x\d+)')[0]
        df[['X_res', 'Y_res']] = df['Resolution'].str.split('x', expand=True).astype(float)
        df['PPI'] = ((df['X_res']**2 + df['Y_res']**2)**0.5) / df['Inches']

        df['cpu'] = df['Cpu'].apply(lambda x: ' '.join(x.split()[:2])).apply(self.clean_cpu)
        df['gpu'] = df['Gpu'].apply(lambda x: x.split()[0]).apply(self.clean_gpu)
        df['os'] = df['OpSys'].apply(self.clean_os)

        df[['HDD', 'SSD', 'Hybrid', 'Flash']] = df['Memory'].apply(self.clean_memory)

        df.drop(columns=[
            'ScreenResolution', 'Resolution', 'X_res', 'Y_res',
            'Cpu', 'Gpu', 'Memory', 'OpSys', 'Inches'
        ], inplace=True, errors='ignore')

        return df

    def clean_memory(self, mem):
        hdd = ssd = hybrid = flash = 0
        if isinstance(mem, str):
            parts = mem.split('+')
            for part in parts:
                part = part.strip()
                size = 0
                if 'TB' in part:
                    size = int(float(part.split('TB')[0]) * 1000)
                elif 'GB' in part:
                    size = int(part.split('GB')[0])
                if 'HDD' in part:
                    hdd += size
                elif 'SSD' in part:
                    ssd += size
                elif 'Hybrid' in part:
                    hybrid += size
                elif 'Flash Storage' in part:
                    flash += size
        return pd.Series([hdd, ssd, hybrid, flash])

    def clean_cpu(self, cpu):
        cpu = cpu.lower()
        if "intel core" in cpu:
            return "Intel Core"
        elif "intel celeron" in cpu or "intel pentium" in cpu or "intel atom" in cpu or "intel xeon" in cpu:
            return "Intel Other"
        elif "amd ryzen" in cpu:
            return "AMD Series"
        elif "samsung" in cpu:
            return "Others"
        else:
            return "Others"

    def clean_gpu(self, gpu):
        gpu = gpu.lower()
        if "intel" in gpu:
            return "Intel"
        elif "nvidia" in gpu:
            return "Nvidia"
        elif "amd" in gpu:
            return "AMD"
        else:
            return "Others"

    def clean_os(self, os):
        if os in ['Windows 10', 'Windows 10 S', 'Windows 7']:
            return 'Windows'
        elif os in ['MacOS', 'Mac OS X']:
            return 'Mac'
        else:
            return 'Linux/Others/No Os'


# Transform Categorical data
cat_cols = ['Company', 'TypeName', 'cpu', 'gpu', 'os']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='passthrough'
)

def pipeline():
    pipeline = Pipeline([
        ('feature_engineering', LaptopFeatureEngineer()),
        ('encoding', preprocessor)
    ])
    return pipeline

def log_transform(column):
    column = np.log(column)
    return column

def get_x_transformed_data():
    X_prepared = pipe.fit_transform(X)
    return X_prepared

def get_y_transformed_data():
    y_prepared = log_transform(y)
    return y_prepared

# Create the preprocessing pipeline
pipe = pipeline()
pipe.fit(X, y)

# save the preprocessing pipeline
if __name__ == "__main__":
    with open("models/laptop_preprocessor.pkl", "wb") as f:
        cloudpickle.dump(pipe, f)
    print("Preprocessing pipeline saved to models/laptop_preprocessor.pkl")
