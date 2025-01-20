# item_recommender.py

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from common.data_preprocessing import encode_features
from common import MIN_SIMILARITY_THRESHOLD

# Ürün bazlı öneri sistemi için:
# |
# v
class ItemBasedRecommender:

    def __init__(self, user_df, item_df):
        self.user_df = user_df.copy()
        self.item_df = item_df.copy()
        
        # Fiyat ve ID kolonlarını çıkar
        item_features = self.item_df.drop(['Customer ID', 'Purchase Amount (USD)'], axis=1)
        
        # Ürün özelliklerini kodla
        self.encoded_items = encode_features(item_features)
        self.similarity_matrix = cosine_similarity(self.encoded_items)

        
    def get_recommendations(self, user_id, n_recommendations=3):
        try:
            # Kullanıcının satın aldığı ürünü bul
            user_item_idx = self.item_df[self.item_df['Customer ID'] == user_id].index[0]
            target_item = self.item_df.iloc[user_item_idx]
            
            item_similarities = self.similarity_matrix[user_item_idx]
            
            # Benzerlik eşiğini uygula (Eklendi)
            similar_mask = item_similarities >= MIN_SIMILARITY_THRESHOLD
            item_similarities = item_similarities[similar_mask]
            
            # Eğer eşiği geçen ürün yoksa boş döndür
            if len(item_similarities) == 0:
                return pd.DataFrame(), None
            
            # Benzerlik sırasına göre ürünleri al
            similar_items_idx = np.argsort(item_similarities)[::-1][1:n_recommendations+1]

            
            # Önerileri hazırla
            recommendations = []
            
            for idx in similar_items_idx:
                similar_item = self.item_df.iloc[idx]
                # Ürünü alan kullanıcının bilgilerini al
                user_info = self.user_df[self.user_df['Customer ID'] == similar_item['Customer ID']].iloc[0]
                
                recommendation = {
                    'Item Purchased': similar_item['Item Purchased'],
                    'Category': similar_item['Category'],
                    'Color': similar_item['Color'],
                    'Season': similar_item['Season'],
                    'Purchase Amount': similar_item['Purchase Amount (USD)'],
                    'Similarity': item_similarities[idx],
                    'User_Age': user_info['Age'],
                    'User_Gender': user_info['Gender'],
                    'User_Location': user_info['Location'],
                    'User_Size': user_info['Size'],
                    'User_Previous_Purchases': user_info['Previous Purchases'],
                    'User_Frequency': user_info['Frequency of Purchases']
                }
                recommendations.append(recommendation)
            
            return pd.DataFrame(recommendations), target_item.to_dict()
            
        except Exception as e:
            print(f"Öneri hatası: {str(e)}")
            return pd.DataFrame(), None
