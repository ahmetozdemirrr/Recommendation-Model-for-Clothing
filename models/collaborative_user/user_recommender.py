# user_recommender.py

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from common.data_preprocessing import encode_features
from common import MIN_SIMILARITY_THRESHOLD

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
            user_similarities = self.similarity_matrix[user_idx]
            
            # Benzerlik eşiğini uygula (Eklendi)
            similar_mask = user_similarities >= MIN_SIMILARITY_THRESHOLD
            user_similarities = user_similarities[similar_mask]
            
            # Eğer eşiği geçen kullanıcı yoksa boş döndür
            if len(user_similarities) == 0:
                return pd.DataFrame(), None
            
            # Benzerlik sırasına göre kullanıcıları al
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
