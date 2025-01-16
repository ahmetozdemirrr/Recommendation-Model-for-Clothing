# recommender_models.py

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from data_preprocessing import encode_features


# Kullanıcı bazlı öneri sistemi için:
# |
# v
class UserBasedRecommender:

    def __init__(self, user_df, item_df):
        self.user_df = user_df.copy()
        self.item_df = item_df.copy()
        self.encoded_users = encode_features(user_df)
        self.similarity_matrix = cosine_similarity(self.encoded_users)
        

    def get_recommendations(self, user_id, n_recommendations=3):
        try:
            # Kullanıcının indeksini bul
            user_idx = self.user_df[self.user_df['Customer ID'] == user_id].index[0]
            
            # Benzerlik skorlarını al
            user_similarities = self.similarity_matrix[user_idx]
            
            # En benzer kullanıcıların indekslerini bul (kendisi hariç)
            similar_users_idx = np.argsort(user_similarities)[::-1][1:n_recommendations+1]
            similar_users = self.user_df.iloc[similar_users_idx]
            
            # Hedef kullanıcının özellikleri
            target_user = self.user_df[self.user_df['Customer ID'] == user_id].iloc[0]
            
            # Önerileri hazırla
            recommendations = []
            for idx, similar_user in similar_users.iterrows():
                user_item = self.item_df[self.item_df['Customer ID'] == similar_user['Customer ID']].iloc[0]
                
                recommendation = {
                    'Item Purchased': user_item['Item Purchased'],
                    'Category': user_item['Category'],
                    'Color': user_item['Color'],
                    'Season': user_item['Season'],
                    'Purchase Amount': user_item['Purchase Amount (USD)'],
                    'Similarity': user_similarities[idx],
                    'User_Age': similar_user['Age'],
                    'User_Gender': similar_user['Gender'],
                    'User_Location': similar_user['Location'],
                    'User_Size': similar_user['Size'],
                    'User_Previous_Purchases': similar_user['Previous Purchases'],
                    'User_Frequency': similar_user['Frequency of Purchases']
                }
                recommendations.append(recommendation)
            
            return pd.DataFrame(recommendations), target_user.to_dict()
            
        except Exception as e:
            print(f"Öneri hatası: {str(e)}")
            return pd.DataFrame(), None


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
            
            # Ürün benzerliklerini al
            item_similarities = self.similarity_matrix[user_item_idx]
            
            # En benzer ürünlerin indekslerini bul (kendisi hariç)
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
