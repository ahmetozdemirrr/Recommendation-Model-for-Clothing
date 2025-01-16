# evaluation.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from data_preprocessing import encode_features, split_dataset
from recommender_models import UserBasedRecommender, ItemBasedRecommender
from clustering import ClusteringRecommender


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


    def calculate_recommendation_score(self, user_id, recommendations, n_recommendations):
        """
        Önerilen ürünlerin başarı skorunu hesaplar.
        Her model için tam olarak n_recommendations kadar öneri kullanır.
        """
        try:
            # Kullanıcının mevcut ürününün indeksini bul
            user_item_idx = self.item_df[self.item_df['Customer ID'] == user_id].index[0]
            user_item = self.item_df.iloc[user_item_idx]
            
            # Önerilen ürünlerin indekslerini bul
            recommended_indices = []
            for _, rec in recommendations.iterrows():
                # Daha esnek eşleştirme kriterleri
                potential_matches = self.item_df[
                    ((self.item_df['Item Purchased'] == rec['Item Purchased']) |
                     (self.item_df['Category'] == rec['Category'])) &
                    (self.item_df['Customer ID'] != user_id)  # Kendisini önermesin
                ]
                
                if not potential_matches.empty:
                    # Her potansiyel eşleşme için benzerlik skoru hesapla
                    match_scores = []
                    for idx in potential_matches.index:
                        base_similarity = self.similarity_matrix[user_item_idx][idx]
                        
                        # Ek özellik benzerlikleri için bonus skorlar
                        bonus = 0.0
                        if potential_matches.loc[idx, 'Season'] == user_item['Season']:
                            bonus += 0.1
                        if potential_matches.loc[idx, 'Color'] == user_item['Color']:
                            bonus += 0.05
                        
                        # Fiyat benzerliği için bonus
                        price_diff = abs(potential_matches.loc[idx, 'Purchase Amount (USD)'] - 
                                      user_item['Purchase Amount (USD)'])
                        if price_diff <= 20:  # 20$ fark için bonus
                            bonus += 0.05
                            
                        match_scores.append(base_similarity + bonus)
                    
                    # En yüksek skorlu eşleşmeyi seç
                    best_match_idx = potential_matches.index[np.argmax(match_scores)]
                    recommended_indices.append(best_match_idx)
                
                # İstenen sayıda öneriye ulaşınca dur
                if len(recommended_indices) >= n_recommendations:
                    break
            
            # Yeterli öneri yoksa 0 döndür
            if len(recommended_indices) < n_recommendations:
                return 0
            
            # Tam olarak n_recommendations kadar öneriyi değerlendir
            recommended_indices = recommended_indices[:n_recommendations]
            
            # Benzerlik skorlarını topla
            similarity_scores = [self.similarity_matrix[user_item_idx][rec_idx] 
                               for rec_idx in recommended_indices]
            
            # Ortalama benzerlik skoru
            return np.mean(similarity_scores)
            
        except Exception as e:
            print(f"Skor hesaplama hatası: {str(e)}")
            return 0


    def evaluate_models(self, test_users, recommendation_ranges=None):
        """
        Farklı öneri sayıları için modelleri değerlendirir
        """
        if recommendation_ranges is None:
            recommendation_ranges = range(10, 101, 10)
            
        results = {
            'user_based': [],
            'item_based': [],
            'cluster_based': []
        }
        
        total_users = len(test_users)
        total_ranges = len(recommendation_ranges)
        
        print("\nDeğerlendirme başlıyor...")
        print(f"Test edilecek kullanıcı sayısı: {total_users}")
        print(f"Test edilecek öneri sayısı aralıkları: {list(recommendation_ranges)}")
        print("="*50)
        
        for range_idx, n_recommendations in enumerate(recommendation_ranges, 1):
            print(f"\nİLERLEME: {range_idx}/{total_ranges} - Öneri sayısı: {n_recommendations}")
            
            # Her model için ortalama skorları hesapla
            model_scores = {
                'user_based': [],
                'item_based': [],
                'cluster_based': []
            }
            
            for idx, user_id in enumerate(test_users, 1):
                print(f"Test edilen kullanıcı: {idx}/{total_users}", end='\r')
                
                # User-based öneriler
                recommendations, _ = self.user_recommender.get_recommendations(
                    user_id, n_recommendations * 2)  # Daha fazla öneri al
                if not recommendations.empty:
                    score = self.calculate_recommendation_score(user_id, recommendations, n_recommendations)
                    model_scores['user_based'].append(score)
                
                # Item-based öneriler
                recommendations, _ = self.item_recommender.get_recommendations(
                    user_id, n_recommendations * 2)  # Daha fazla öneri al
                if not recommendations.empty:
                    score = self.calculate_recommendation_score(user_id, recommendations, n_recommendations)
                    model_scores['item_based'].append(score)
                
                # Cluster-based öneriler
                recommendations, _ = self.cluster_recommender.get_cluster_recommendations(
                    user_id, n_recommendations * 2)  # Daha fazla öneri al
                if not recommendations.empty:
                    score = self.calculate_recommendation_score(user_id, recommendations, n_recommendations)
                    model_scores['cluster_based'].append(score)
            
            print()  # Yeni satır
            
            # Her model için ortalama skorları kaydet
            for model_name in model_scores:
                if model_scores[model_name]:
                    avg_score = np.mean(model_scores[model_name])
                    results[model_name].append(avg_score)
                else:
                    results[model_name].append(0)
                print(f"{model_name}: {len(model_scores[model_name])} kullanıcı için ortalama skor = {results[model_name][-1]:.4f}")
            
        # Skorları normalize et
        scaler = MinMaxScaler()
        normalized_results = {}
        
        # Tüm skorları bir araya topla
        all_scores = []
        for scores in results.values():
            all_scores.extend(scores)
            
        # Tek bir scaler ile tüm skorları normalize et
        if len(all_scores) > 0:
            all_scores = np.array(all_scores).reshape(-1, 1)
            normalized_scores = scaler.fit_transform(all_scores).flatten()
            
            # Normalize edilmiş skorları modellere geri dağıt
            idx = 0
            for model_name in results:
                n_scores = len(results[model_name])
                normalized_results[model_name] = normalized_scores[idx:idx+n_scores]
                idx += n_scores
        else:
            for model_name in results:
                normalized_results[model_name] = np.zeros_like(results[model_name])
            
        return normalized_results, list(recommendation_ranges)


    def plot_results(self, results, recommendation_ranges):
        """
        Değerlendirme sonuçlarını görselleştirir
        """
        plt.figure(figsize=(12, 8))
        
        plt.plot(recommendation_ranges, results['user_based'], 
                marker='o', label='Kullanıcı Bazlı', linewidth=2)
        plt.plot(recommendation_ranges, results['item_based'], 
                marker='s', label='Ürün Bazlı', linewidth=2)
        plt.plot(recommendation_ranges, results['cluster_based'], 
                marker='^', label='Küme Bazlı', linewidth=2)
        
        plt.xlabel('Öneri Sayısı')
        plt.ylabel('Normalize Edilmiş Başarı Oranı')
        plt.title('Öneri Sistemleri Performans Karşılaştırması')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
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
