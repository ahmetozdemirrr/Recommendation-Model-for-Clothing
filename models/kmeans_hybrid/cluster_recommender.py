# cluster_recommender.py

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from kneed import KneeLocator
from common.data_preprocessing import encode_features
from common import MIN_SIMILARITY_THRESHOLD

class ClusteringRecommender:

    def find_optimal_k(self, data, k_range=range(1, 11)):
        """Elbow metodu ile optimal k değerini bulur"""
        inertias = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        
        kneedle = KneeLocator(
            x=list(k_range), 
            y=inertias, 
            S=1.0, 
            curve='convex', 
            direction='decreasing'
        )
        optimal_k = kneedle.knee
        return optimal_k if optimal_k else 3


    def __init__(self, user_df, item_df):
        self.user_df = user_df.copy()
        self.item_df = item_df.copy()
        
        # Kullanıcı kümelemesi için seçilen özellikler
        user_features = [
            'Age',
            'Gender',
            'Previous Purchases',
            'Frequency of Purchases',
            'Size',
            'Subscription Status'
        ]
        
        # Ürün kümelemesi için seçilen özellikler
        item_features = [
            'Item Purchased',
            'Category',
            'Season'
        ]
        
        # Benzerlik hesaplama için kullanılacak özellikler
        similarity_features = [
            'Item Purchased',
            'Category',
            'Color',
            'Season'
        ]
        
        # Seçilen özellikleri kullanarak kümeleme için veri hazırla
        self.encoded_users = encode_features(self.user_df[user_features])
        self.encoded_items = encode_features(self.item_df[item_features])
        self.encoded_similarity = encode_features(self.item_df[similarity_features])
        
        # Optimal k değerlerini bul
        self.n_user_clusters = self.find_optimal_k(self.encoded_users)
        self.n_item_clusters = self.find_optimal_k(self.encoded_items)
        
        print(f"\nOptimal küme sayıları belirlendi:")
        print(f"- Kullanıcı kümeleri: {self.n_user_clusters}")
        print(f"- Ürün kümeleri: {self.n_item_clusters}")
        
        # Kümeleme modellerini oluştur ve eğit
        self.user_clustering = KMeans(n_clusters=self.n_user_clusters, random_state=42)
        self.item_clustering = KMeans(n_clusters=self.n_item_clusters, random_state=42)
        
        self.user_clusters = self.user_clustering.fit_predict(self.encoded_users)
        self.item_clusters = self.item_clustering.fit_predict(self.encoded_items)
        
        # Küme etiketlerini DataFrame'lere ekle
        self.user_df['Cluster'] = self.user_clusters
        self.item_df['Cluster'] = self.item_clusters


    def calculate_similarity_score(self, idx1, idx2, user_cluster):
        """
        İki ürün arasındaki benzerlik skorunu hesaplar.
        
        Parametreler:
        idx1: Birinci ürünün indeksi
        idx2: İkinci ürünün indeksi
        user_cluster: Hedef kullanıcının kümesi
        
        Dönüş:
        float: 0-1 arası benzerlik skoru
        """
        # Ürünleri al
        item1 = self.item_df.iloc[idx1]
        item2 = self.item_df.iloc[idx2]
        
        # Base similarity - encoded özellikler üzerinden cosine similarity
        base_similarity = cosine_similarity(
            self.encoded_similarity[idx1].reshape(1, -1),
            self.encoded_similarity[idx2].reshape(1, -1)
        )[0][0]
        
        # Ağırlıklar
        weights = {
            'cluster': 0.31,      # Ürün kümesi benzerliği
            'category': 0.25,     # Kategori benzerliği
            'season': 0.15,       # Sezon benzerliği
            'user_cluster': 0.19, # Kullanıcı kümesi benzerliği
            'price': 0.05,        # Fiyat benzerliği
            'color': 0.05         # Renk benzerliği
        }
        
        # Her faktör için benzerlik hesapla (0-1 arası)
        similarities = {
            'cluster': 1.0 if item1['Cluster'] == item2['Cluster'] else 0.0,
            'category': 1.0 if item1['Category'] == item2['Category'] else 0.0,
            'season': 1.0 if item1['Season'] == item2['Season'] else 0.0,
            'user_cluster': 1.0 if self.user_df[self.user_df['Customer ID'] == item1['Customer ID']].iloc[0]['Cluster'] == user_cluster else 0.0,
            'price': max(0, 1 - abs(item1['Purchase Amount (USD)'] - item2['Purchase Amount (USD)']) / max(item2['Purchase Amount (USD)'], 1)),
            'color': 1.0 if item1['Color'] == item2['Color'] else 0.0
        }
        
        # Ağırlıklı toplam faktör benzerliği
        factor_similarity = sum(weights[k] * similarities[k] for k in weights)
        
        # Final benzerlik skoru
        final_similarity = base_similarity * factor_similarity
        
        return final_similarity


    def get_cluster_recommendations(self, user_id, n_recommendations=5):
        try:
            print(f"\nKullanıcı ID: {user_id} için öneriler hazırlanıyor...")
            
            # Kullanıcının ve ürününün bilgilerini al
            user_idx = self.user_df[self.user_df['Customer ID'] == user_id].index[0]
            user_cluster = self.user_clusters[user_idx]
            user_info = self.user_df.iloc[user_idx]
            
            # Kullanıcının mevcut ürününü bul
            user_item_idx = self.item_df[self.item_df['Customer ID'] == user_id].index[0]
            user_item = self.item_df.iloc[user_item_idx]
            
            print(f"Kullanıcı kümesi: {user_cluster}")
            print(f"Kullanıcının ürün kümesi: {user_item['Cluster']}")
            print(f"Kullanıcının mevcut ürünü: {user_item['Item Purchased']}")
            
            # Tüm ürünleri değerlendir (kendisi hariç)
            other_items = self.item_df[self.item_df['Customer ID'] != user_id].copy()
            print(f"Toplam değerlendirilecek ürün sayısı: {len(other_items)}")
            
            # Her ürün için benzerlik skorunu hesapla
            similarities = []
            for idx in other_items.index:
                similarity = self.calculate_similarity_score(user_item_idx, idx, user_cluster)
                similarities.append(similarity)
            
            other_items['Similarity'] = similarities
            
            # Benzerlik dağılımını göster
            similarity_stats = other_items['Similarity'].describe()
            print("\nBenzerlik skorları dağılımı:")
            print(similarity_stats)
            
            # Minimum benzerlik skoru filtresi
            recommendations = other_items[other_items['Similarity'] >= MIN_SIMILARITY_THRESHOLD]
            print(f"\n{MIN_SIMILARITY_THRESHOLD} üzeri benzerlik skoruna sahip ürün sayısı: {len(recommendations)}")
            
            if len(recommendations) == 0:
                print(f"\nUyarı: {MIN_SIMILARITY_THRESHOLD} benzerlik eşiği için yeterli öneri bulunamadı.")
                return pd.DataFrame(), None
            
            # Benzerliğe göre sırala ve önerileri seç
            recommendations = recommendations.sort_values('Similarity', ascending=False)
            recommendations = recommendations.head(n_recommendations)
            
            # Önerileri hazırla
            final_recommendations = []
            for _, item in recommendations.iterrows():
                item_user = self.user_df[self.user_df['Customer ID'] == item['Customer ID']].iloc[0]
                
                recommendation = {
                    'Item Purchased': item['Item Purchased'],
                    'Category': item['Category'],
                    'Color': item['Color'],
                    'Season': item['Season'],
                    'Purchase Amount': item['Purchase Amount (USD)'],
                    'Similarity': item['Similarity'],
                    'User_Cluster': user_cluster,
                    'Item_Cluster': item['Cluster'],
                    'User_Age': item_user['Age'],
                    'User_Gender': item_user['Gender'],
                    'User_Location': item_user['Location'],
                    'User_Size': item_user['Size'],
                    'User_Previous_Purchases': item_user['Previous Purchases'],
                    'User_Frequency': item_user['Frequency of Purchases'],
                    'User_Subscription': item_user['Subscription Status']
                }
                final_recommendations.append(recommendation)
            
            print(f"\nToplam önerilen ürün sayısı: {len(final_recommendations)}")
            
            # Hedef ürün bilgilerini hazırla
            target_dict = user_item.to_dict()
            target_dict['Purchase Amount'] = target_dict.pop('Purchase Amount (USD)', 0)
            
            return pd.DataFrame(final_recommendations), target_dict
            
        except Exception as e:
            import traceback
            print(f"Kümeleme önerisi hatası: {str(e)}")
            print("Hata detayı:")
            print(traceback.format_exc())
            return pd.DataFrame(), None


    def get_cluster_insights(self):
        """Kümeleme analizi sonuçlarını döndürür"""
        insights = {
            'user_clusters': {},
            'item_clusters': {}
        }
        
        # Kullanıcı kümeleri için içgörüler
        for cluster in range(self.n_user_clusters):
            cluster_users = self.user_df[self.user_df['Cluster'] == cluster]
            insights['user_clusters'][cluster] = {
                'size': len(cluster_users),
                'avg_age': cluster_users['Age'].mean(),
                'common_gender': cluster_users['Gender'].mode()[0],
                'avg_purchases': cluster_users['Previous Purchases'].mean(),
                'subscription_ratio': (cluster_users['Subscription Status'] == 'Yes').mean() * 100,
                'common_size': cluster_users['Size'].mode()[0]
            }
            
        # Ürün kümeleri için içgörüler
        for cluster in range(self.n_item_clusters):
            cluster_items = self.item_df[self.item_df['Cluster'] == cluster]
            insights['item_clusters'][cluster] = {
                'size': len(cluster_items),
                'common_category': cluster_items['Category'].mode()[0],
                'common_season': cluster_items['Season'].mode()[0],
                'avg_price': cluster_items['Purchase Amount (USD)'].mean(),
                'common_color': cluster_items['Color'].mode()[0]
            }
            
        return insights