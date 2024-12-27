import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def prepare_content_based_data(data_path):
    # Veriyi yükle
    df = pd.read_csv(data_path)

    # Kullanılacak sütunlar
    feature_cols = ["Item Purchased", "Category", "Color", "Season"]

    # One-Hot Encoding
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    feature_matrix = ohe.fit_transform(df[feature_cols])

    return df, feature_matrix


def recommend_content_based(user_id, data_path, top_n=3):
    df, feature_matrix = prepare_content_based_data(data_path)

    # Kullanıcı indexini bul
    if user_id not in df["Customer ID"].values:
        raise ValueError(f"User ID {user_id} veri kümesinde bulunamadı.")
    
    user_index = df[df["Customer ID"] == user_id].index[0]

    # Kullanıcıya benzer ürünler bul
    user_features = feature_matrix[user_index].reshape(1, -1)
    similarities = cosine_similarity(user_features, feature_matrix).flatten()

    # Kullanıcı hariç benzerlik skorlarını al
    similar_indices = np.argsort(similarities)[::-1]
    similar_indices = similar_indices[similar_indices != user_index]

    # En benzer ürünleri döndür
    recommended_items = df.iloc[similar_indices[:top_n]]
    return recommended_items



# Örnek kullanım
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Content-Based Filtering ile öneri")
    parser.add_argument("--user_id", type=int, required=True, help="Kullanıcı ID'si")
    parser.add_argument("--top_n", type=int, default=3, help="Öneri sayısı")
    args = parser.parse_args()

    data_path = "./data/shopping_trends_updated.csv"
    recommendations = recommend_content_based(args.user_id, data_path, top_n=args.top_n)
    print(f"Kullanıcı {args.user_id} için önerilen ürünler:")
    print(recommendations[["Item Purchased", "Category", "Color", "Season"]])
