import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Veri Hazırlık Fonksiyonu
def prepare_data(data_path="./data/shopping_trends_updated.csv"):
    # Dosya yolunu dinamik hale getirelim (src'den bağımsız çalışabilir)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(base_dir, data_path)
    
    # Veri yükleme
    df = pd.read_csv(data_path)
    
    # Boş değerleri çıkaralım
    df = df.dropna()

    # Kullanıcı ID'sini index olarak ayarla
    df = df.set_index("Customer ID")

    # Numerik ve kategorik sütunların ayrımı
    numeric_cols = ["Age", "Previous Purchases"]
    categorical_cols = ["Gender", "Item Purchased", "Category", "Location", 
                        "Size", "Color", "Season", "Frequency of Purchases"]
    
    # Numerik sütunları ölçeklendirme
    scaler = MinMaxScaler()
    numeric_data = scaler.fit_transform(df[numeric_cols])
    
    # Kategorik sütunları One-Hot encode etme
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_data = ohe.fit_transform(df[categorical_cols])
    
    # Özellikleri birleştirme
    feature_matrix = np.hstack([numeric_data, cat_data])
    
    # Sonuçları döndürme
    return df, feature_matrix, ohe, numeric_cols, categorical_cols


# Kullanıcılar Arası Benzerlik Hesaplama
def compute_similarity(user_features):
    similarity_matrix = cosine_similarity(user_features)
    return similarity_matrix


# Benzer Kullanıcıları Bulma
def get_similar_users(user_id, df, similarity_matrix, top_n=10):
    user_ids = df.index.to_list()
    if user_id not in user_ids:
        raise ValueError(f"User ID {user_id} dataset içinde bulunamadı.")
    
    user_idx = user_ids.index(user_id)
    user_similarities = similarity_matrix[user_idx]
    similar_users = [(uid, sim) for uid, sim in zip(user_ids, user_similarities) if uid != user_id]
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)
    
    return similar_users[:top_n]


# Kullanıcıya Öneri Yapma
def recommend_items_for_user(user_id, df, similarity_matrix, num_recommendations=3):
    similar_users = get_similar_users(user_id, df, similarity_matrix, top_n=50)
    items_info = []
    
    for (sim_user_id, score) in similar_users:
        sim_user_row = df.loc[sim_user_id]
        item_name = sim_user_row["Item Purchased"]
        category = sim_user_row["Category"]
        amount = sim_user_row["Purchase Amount (USD)"]
        color = sim_user_row["Color"]
        season = sim_user_row["Season"]
        
        item_dict = {
            "Customer ID": sim_user_id,
            "Item Purchased": item_name,
            "Category": category,
            "Amount (USD)": amount,
            "Color": color,
            "Season": season,
        }
        
        if item_dict not in items_info:
            items_info.append(item_dict)
        
        if len(items_info) == num_recommendations:
            break
    
    return items_info


# Ana Program
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="User-based collaborative filtering")
    parser.add_argument("--user_id", type=int, required=True, help="Öneri istenen kullanıcı ID")
    parser.add_argument("--num_recommendations", type=int, default=3, help="Öneri sayısı")
    args = parser.parse_args()

    # Veri hazırlığı
    df, feature_matrix, _, _, _ = prepare_data()
    similarity_matrix = compute_similarity(feature_matrix)

    # Öneriler
    recommended_items = recommend_items_for_user(args.user_id, df, similarity_matrix, num_recommendations=args.num_recommendations)

    print(f"Kullanıcı {args.user_id} için önerilen ürünler ve özellikleri:")
    for item_info in recommended_items:
        print("------------------------------------")
        print(f"Müşteri ID: {item_info['Customer ID']}")
        print(f"Ürün: {item_info['Item Purchased']}")
        print(f"Kategori: {item_info['Category']}")
        print(f"Fiyat (USD): {item_info['Amount (USD)']}")
        print(f"Renk: {item_info['Color']}")
        print(f"Sezon: {item_info['Season']}")
