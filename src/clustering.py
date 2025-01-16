# clustering.py

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
from data_preprocessing import encode_features
from kneed import KneeLocator


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
        
        # Seçilen özellikleri kullanarak kümeleme için veri hazırla
        self.encoded_users = encode_features(self.user_df[user_features])
        self.encoded_items = encode_features(self.item_df[item_features])
        
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


    def get_cluster_recommendations(self, user_id, n_recommendations=5):
        try:
            print(f"\nKullanıcı ID: {user_id} için öneriler hazırlanıyor...")
            
            # Kullanıcının ve ürününün kümelerini bul
            user_idx = self.user_df[self.user_df['Customer ID'] == user_id].index[0]
            user_cluster = self.user_clusters[user_idx]
            user_info = self.user_df.iloc[user_idx]
            print(f"Kullanıcı kümesi: {user_cluster}")
            
            user_item = self.item_df[self.item_df['Customer ID'] == user_id].iloc[0]
            item_cluster = user_item['Cluster']
            print(f"Kullanıcının ürün kümesi: {item_cluster}")
            print(f"Kullanıcının mevcut ürünü: {user_item['Item Purchased']}")
            
            # Tüm ürünleri değerlendir (kendisi hariç)
            recommendations = self.item_df[self.item_df['Customer ID'] != user_id].copy()
            print(f"Toplam değerlendirilecek ürün sayısı: {len(recommendations)}")
            
            # Benzerlik skoru hesaplama
            # Benzerlik bileşenleri ve ağırlıkları
            cluster_weight = 0.4  # Ürün kümesi benzerliği
            category_weight = 0.2  # Kategori benzerliği
            season_weight = 0.15   # Sezon benzerliği
            user_cluster_weight = 0.15  # Kullanıcı kümesi benzerliği
            price_weight = 0.05    # Fiyat benzerliği
            color_weight = 0.05    # Renk benzerliği

            # Başlangıç benzerlik skoru
            recommendations['Similarity'] = 0.0

            # 1. Ürün kümesi benzerliği (0.4)
            matching_cluster = recommendations['Cluster'] == item_cluster
            recommendations.loc[matching_cluster, 'Similarity'] += cluster_weight
            print(f"Aynı ürün kümesinde olan ürün sayısı: {matching_cluster.sum()}")

            # 2. Kategori benzerliği (0.2)
            matching_category = recommendations['Category'] == user_item['Category']
            recommendations.loc[matching_category, 'Similarity'] += category_weight
            print(f"Aynı kategoride olan ürün sayısı: {matching_category.sum()}")

            # 3. Sezon benzerliği (0.15)
            matching_season = recommendations['Season'] == user_item['Season']
            recommendations.loc[matching_season, 'Similarity'] += season_weight
            print(f"Aynı sezonda olan ürün sayısı: {matching_season.sum()}")

            # 4. Kullanıcı kümesi benzerliği (0.15)
            recommendations['User_Cluster'] = recommendations['Customer ID'].map(
                self.user_df.set_index('Customer ID')['Cluster'])
            matching_user_cluster = recommendations['User_Cluster'] == user_cluster
            recommendations.loc[matching_user_cluster, 'Similarity'] += user_cluster_weight
            print(f"Aynı kullanıcı kümesindeki kullanıcılardan ürün sayısı: {matching_user_cluster.sum()}")

            # 5. Fiyat aralığı benzerliği (0.05)
            price_range = 0.2 * user_item['Purchase Amount (USD)']
            price_mask = (
                (recommendations['Purchase Amount (USD)'] >= user_item['Purchase Amount (USD)'] - price_range) & 
                (recommendations['Purchase Amount (USD)'] <= user_item['Purchase Amount (USD)'] + price_range)
            )
            recommendations.loc[price_mask, 'Similarity'] += price_weight
            print(f"Benzer fiyat aralığında olan ürün sayısı: {price_mask.sum()}")

            # 6. Renk benzerliği (0.05)
            matching_color = recommendations['Color'] == user_item['Color']
            recommendations.loc[matching_color, 'Similarity'] += color_weight
            print(f"Aynı renkte olan ürün sayısı: {matching_color.sum()}")
            
            # Benzerlik dağılımını göster
            similarity_stats = recommendations['Similarity'].describe()
            print("\nBenzerlik skorları dağılımı:")
            print(similarity_stats)
            
            # Minimum benzerlik skoru filtresi (artık daha düşük bir eşik)
            MIN_SIMILARITY_THRESHOLD = 0.3  # Eşiği düşürdük
            recommendations = recommendations[recommendations['Similarity'] >= MIN_SIMILARITY_THRESHOLD]
            print(f"\n{MIN_SIMILARITY_THRESHOLD} üzeri benzerlik skoruna sahip ürün sayısı: {len(recommendations)}")
            
            if len(recommendations) == 0:
                print(f"\nUyarı: {MIN_SIMILARITY_THRESHOLD} benzerlik eşiği için yeterli öneri bulunamadı.")
                return pd.DataFrame(), None
            
            # Benzerliğe göre sırala
            recommendations = recommendations.sort_values('Similarity', ascending=False)
            
            # İstenen sayıda öneriyi seç
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
