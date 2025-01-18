# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def split_dataset(data_path):
    """
    Veri setini kullanıcı ve ürün özellikleri olarak ikiye ayırır.
    """
    # Ana veri setini yükle
    df = pd.read_csv(data_path)
    
    # NaN değerleri kontrol et
    if df.isnull().any().any():
        print("\nUyarı: Veri setinde NaN değerler bulundu!")
        print("NaN değerlerin dağılımı:")
        print(df.isnull().sum())
        print("\nNaN değerler temizleniyor...")
        df = df.dropna()
    
    # Kullanıcı özellikleri güncellendi
    user_features = [
        'Customer ID',            # Bağlantı için
        'Age',                    # Sayısal
        'Gender',                 # Kategorik
        'Location',               # Kategorik
        'Size',                   # Özel kodlama
        'Previous Purchases',     # Sayısal
        'Frequency of Purchases', # Özel kodlama
        'Subscription Status'     # Yeni eklenen özellik
    ]
    
    # Ürün özellikleri
    item_features = [
        'Customer ID',            # Bağlantı için
        'Item Purchased',         # Ürün ismi
        'Category',               # Kategorisi
        'Purchase Amount (USD)',  # Fiyat
        'Color',                  # Renk
        'Season'                  # Sezon
    ]
    
    # DataFrameleri ayır
    user_df = df[user_features].copy()
    item_df = df[item_features].copy()
    
    return user_df, item_df


def encode_features(df):
    """
    Özellikleri kodlar: Sayısal, kategorik ve özel kodlamalar
    """
    # Özel kodlamalar için sözlükler
    size_mapping = {'S': 0, 'M': 1, 'L': 2, 'XL': 3}
    frequency_mapping = {
        'Rarely': 0,
        'Occasionally': 1,
        'Monthly': 2,
        'Weekly': 3,
        'Often': 4
    }
    subscription_mapping = {
        'Yes': 1,
        'No': 0
    }
    
    # DataFrame'in bir kopyasını oluştur
    df_encoded = df.copy()
    
    # CustomerID'yi çıkar
    if 'Customer ID' in df_encoded.columns:
        df_encoded = df_encoded.drop('Customer ID', axis=1)
    
    # NaN kontrolü
    if df_encoded.isnull().any().any():
        df_encoded = df_encoded.fillna('Unknown')
        print("Uyarı: Boş değerler 'Unknown' ile dolduruldu")
    
    # Özel kodlamaları uygula
    if 'Size' in df_encoded.columns:
        df_encoded['Size'] = df_encoded['Size'].map(size_mapping).fillna(-1)
    
    if 'Frequency of Purchases' in df_encoded.columns:
        df_encoded['Frequency of Purchases'] = df_encoded['Frequency of Purchases'].map(frequency_mapping).fillna(-1)
        
    if 'Subscription Status' in df_encoded.columns:
        df_encoded['Subscription Status'] = df_encoded['Subscription Status'].map(subscription_mapping).fillna(-1)
    
    # Sayısal sütunları ölçeklendir
    numeric_cols = ['Age', 'Previous Purchases']
    numeric_cols = [col for col in numeric_cols if col in df_encoded.columns]
    
    if numeric_cols:
        scaler = MinMaxScaler()
        df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
    
    # Kategorik sütunları belirle (özel kodlama ve sayısal olanlar hariç)
    categorical_cols = [col for col in df_encoded.columns 
                       if col not in numeric_cols + ['Size', 'Frequency of Purchases', 'Subscription Status']]
    
    # One-Hot Encoding uygula
    if categorical_cols:
        onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cats = onehot.fit_transform(df_encoded[categorical_cols])
        
        # One-Hot encoded verileri DataFrame'e çevir
        encoded_cats_df = pd.DataFrame(
            encoded_cats,
            columns=onehot.get_feature_names_out(categorical_cols)
        )
        
        # Sayısal ve özel kodlanmış sütunları al
        final_cols = [col for col in df_encoded.columns if col not in categorical_cols]
        
        # Tüm kodlanmış verileri birleştir
        if final_cols:
            encoded_matrix = pd.concat([
                df_encoded[final_cols].reset_index(drop=True),
                encoded_cats_df
            ], axis=1)
        else:
            encoded_matrix = encoded_cats_df
            
        # Son bir NaN kontrolü
        if encoded_matrix.isnull().any().any():
            print("Uyarı: Kodlama sonrası NaN değerler tespit edildi!")
            print("NaN değerlerin dağılımı:")
            print(encoded_matrix.isnull().sum())
            encoded_matrix = encoded_matrix.fillna(0)
            
        return encoded_matrix.values
    
    return df_encoded.values
