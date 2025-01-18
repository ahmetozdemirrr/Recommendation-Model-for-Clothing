# evaluation.py

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from models.collaborative_user.user_recommender import UserBasedRecommender
from models.collaborative_item.item_recommender import ItemBasedRecommender
from models.kmeans_hybrid.cluster_recommender import ClusteringRecommender
from common.data_preprocessing import encode_features, split_dataset


class RecommenderEvaluator:

    def __init__(self, user_df, item_df):
        self.user_df = user_df.copy()
        self.item_df = item_df.copy()
        
        # Benzerlik matrisi için kullanılacak özellikleri seç
        self.similarity_features = [
            'Item Purchased',
            'Category',
            'Color',
            'Season',
            'Purchase Amount (USD)'
        ]
        
        # Özellik matrisini oluştur
        self.encoded_items = encode_features(self.item_df[self.similarity_features])
        
        # Benzerlik matrisini hesapla
        self.similarity_matrix = cosine_similarity(self.encoded_items)
        
        # Modelleri başlat
        self.user_recommender = UserBasedRecommender(self.user_df, self.item_df)
        self.item_recommender = ItemBasedRecommender(self.user_df, self.item_df)
        self.cluster_recommender = ClusteringRecommender(self.user_df, self.item_df)
        
        # Sonuçları saklamak için sözlükler
        self.results = {
            'user_based': [],
            'item_based': [],
            'cluster_based': []
        }


    def calculate_recommendation_score(self, user_id, recommendations, n_recommendations, model_type='item_based'):
        try:
            user_item_idx = self.item_df[self.item_df['Customer ID'] == user_id].index[0]
            user_item = self.item_df.iloc[user_item_idx]
            
            # Önerileri direkt kullan
            recommended_items = recommendations.head(n_recommendations)
            
            if model_type == 'cluster_based':
                # Cluster-based model için benzerlik skorlarını direkt al
                similarity_scores = recommended_items['Similarity'].values
            else:
                # Diğer modeller için cosine similarity kullan
                similarity_scores = []
                for _, rec in recommended_items.iterrows():
                    rec_idx = self.item_df[
                        (self.item_df['Item Purchased'] == rec['Item Purchased']) & 
                        (self.item_df['Category'] == rec['Category']) &
                        (self.item_df['Color'] == rec['Color']) &
                        (self.item_df['Season'] == rec['Season'])
                    ].index[0]
                    
                    base_similarity = self.similarity_matrix[user_item_idx][rec_idx]
                    
                    # Bonus skorlar
                    bonus = 0.0
                    if rec['Season'] == user_item['Season']:
                        bonus += 0.25
                    if rec['Color'] == user_item['Color']:
                        bonus += 0.15
                    if rec['Category'] == user_item['Category']:
                        bonus += 0.25
                    
                    final_similarity = min(1.0, base_similarity + bonus)
                    similarity_scores.append(final_similarity)
            
            # Ortalama benzerlik skoru
            return np.mean(similarity_scores)
                
        except Exception as e:
            print(f"Skor hesaplama hatası: {str(e)}")
            return 0


    def evaluate_models(self, test_users, recommendation_ranges=None):
        if recommendation_ranges is None:
            recommendation_ranges = range(10, 101, 10)
            
        total_users = len(test_users)
        total_ranges = len(recommendation_ranges)
        
        print("\nDeğerlendirme başlıyor...")
        print(f"Test edilecek kullanıcı sayısı: {total_users}")
        print(f"Test edilecek öneri sayısı aralıkları: {list(recommendation_ranges)}")
        print("="*50)
        
        for range_idx, n_recommendations in enumerate(recommendation_ranges, 1):
            print(f"\nİLERLEME: {range_idx}/{total_ranges} - Öneri sayısı: {n_recommendations}")
            
            model_scores = {
                'user_based': [],
                'item_based': [],
                'cluster_based': []
            }
            
            for idx, user_id in enumerate(test_users, 1):
                print(f"Test edilen kullanıcı: {idx}/{total_users}", end='\r')
                
                # User-based öneriler
                recommendations, _ = self.user_recommender.get_recommendations(
                    user_id, n_recommendations)
                if not recommendations.empty:
                    score = self.calculate_recommendation_score(user_id, recommendations, n_recommendations, 'user_based')
                    model_scores['user_based'].append(score)

                # Item-based öneriler            
                recommendations, _ = self.item_recommender.get_recommendations(
                    user_id, n_recommendations)
                if not recommendations.empty:
                    score = self.calculate_recommendation_score(user_id, recommendations, n_recommendations, 'item_based')
                    model_scores['item_based'].append(score)

                # Cluster-based öneriler
                recommendations, _ = self.cluster_recommender.get_cluster_recommendations(
                    user_id, n_recommendations)
                if not recommendations.empty:
                    score = self.calculate_recommendation_score(user_id, recommendations, n_recommendations, 'cluster_based')
                    model_scores['cluster_based'].append(score)
            
            print()  # Yeni satır
            
            # Her model için ortalama skorları kaydet
            for model_name in model_scores:
                if model_scores[model_name]:
                    avg_score = np.mean(model_scores[model_name])
                    self.results[model_name].append(avg_score)
                else:
                    self.results[model_name].append(0)
                print(f"{model_name}: {len(model_scores[model_name])} kullanıcı için ortalama skor = {self.results[model_name][-1]:.4f}")
                    
        return self.results, list(recommendation_ranges)


    def plot_results(self, results, recommendation_ranges):
        """
        Değerlendirme sonuçlarını görselleştirir
        """
        plt.figure(figsize=(12, 8))
        
        # Her modelin sonuçlarını ayrı ayrı çiz
        plt.plot(recommendation_ranges, self.results['user_based'], 
                marker='o', label='Kullanıcı Bazlı', linewidth=2)
        plt.plot(recommendation_ranges, self.results['item_based'], 
                marker='s', label='Ürün Bazlı', linewidth=2)
        plt.plot(recommendation_ranges, self.results['cluster_based'], 
                marker='^', label='Küme Bazlı', linewidth=2)
        
        plt.xlabel('Öneri Sayısı')
        plt.ylabel('Benzerlik Skoru')
        plt.title('Öneri Sistemleri Performans Karşılaştırması')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Y ekseni aralığını 0-1 arası sabitle
        plt.ylim(0, 1)
        
        plt.tight_layout()
                
        # results klasörü yoksa oluştur
        import os
        if not os.path.exists('results'):
            os.makedirs('results')
                
        # Grafiği kaydet
        plt.savefig('results/model_performance_comparison.png', 
                    dpi=300, 
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none')
        plt.close()  # Belleği temizle


def evaluate_recommenders(data_path, n_test_users=100, recommendation_ranges=None):
    """
    Öneri sistemlerini değerlendirir ve sonuçları görselleştirir
    """
    # Veriyi yükle
    user_df, item_df = split_dataset(data_path)
    
    # Rastgele test kullanıcıları seç
    test_users = np.random.choice(user_df['Customer ID'].unique(), 
                                size=min(n_test_users, len(user_df)), 
                                replace=False)
    
    # Değerlendiriciyi başlat
    evaluator = RecommenderEvaluator(user_df, item_df)
    
    # Modelleri değerlendir
    results, ranges = evaluator.evaluate_models(test_users, recommendation_ranges)
    
    # Sonuçları görselleştir
    evaluator.plot_results(results, ranges)
    
    return results, ranges