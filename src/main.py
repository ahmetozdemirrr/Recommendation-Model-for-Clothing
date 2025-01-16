# main.py

import argparse
import sys
from data_preprocessing import split_dataset
from recommender_models import UserBasedRecommender, ItemBasedRecommender
from clustering import ClusteringRecommender
from utils import RecommendationFormatter
from evaluation import evaluate_recommenders


def print_cluster_insights(insights):
    """Kümeleme analizi sonuçlarını formatlar ve ekrana basar"""
    print("\nKÜME ANALİZİ SONUÇLARI:")
    print("="*50)
    
    print("\nKullanıcı Kümeleri:")
    for cluster, data in insights['user_clusters'].items():
        print(f"\nKüme {cluster}:")
        print(f"  - Küme Büyüklüğü: {data['size']} kullanıcı")
        print(f"  - Ortalama Yaş: {data['avg_age']:.1f}")
        print(f"  - Baskın Cinsiyet: {data['common_gender']}")
        print(f"  - Ortalama Alışveriş: {data['avg_purchases']:.1f}")
    
    print("\nÜrün Kümeleri:")
    for cluster, data in insights['item_clusters'].items():
        print(f"\nKüme {cluster}:")
        print(f"  - Küme Büyüklüğü: {data['size']} ürün")
        print(f"  - Baskın Kategori: {data['common_category']}")
        print(f"  - Baskın Sezon: {data['common_season']}")
        print(f"  - Ortalama Fiyat: ${data['avg_price']:.2f}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Alışveriş Öneri Sistemi')
    
    # user_id artık sadece recommendation modunda zorunlu
    parser.add_argument('--user_id', type=int, 
                      help='Öneri yapılacak kullanıcı ID')
    parser.add_argument('--mode', choices=['user', 'item', 'cluster'], default='user', 
                      help='Öneri modu: user (kullanıcı bazlı), item (ürün bazlı) veya cluster (küme bazlı)')
    parser.add_argument('--num_recommendations', type=int, default=5, 
                      help='Önerilecek ürün sayısı')
    parser.add_argument('--evaluate', action='store_true',
                      help='Modelleri değerlendirme modunu aktifleştirir')
    parser.add_argument('--n_test_users', type=int, default=100,
                      help='Değerlendirme için kullanılacak test kullanıcısı sayısı')
    
    args = parser.parse_args()
    
    # Veriyi yükle ve böl
    data_path = "./data/shopping_trends_updated.csv"
    
    # Eğer değerlendirme modu seçildiyse
    if args.evaluate:
        print("\nModeller değerlendiriliyor...")
        results, ranges = evaluate_recommenders(
            data_path,
            n_test_users=args.n_test_users
        )
        return
        
    # Değerlendirme modu değilse, user_id zorunlu
    if not args.user_id:
        parser.error("Öneri modu için --user_id parametresi gereklidir")
    
    user_df, item_df = split_dataset(data_path)
    
    if args.mode == 'cluster':
        print("\nKüme bazlı öneriler hazırlanıyor...")
        recommender = ClusteringRecommender(user_df, item_df)
        
        # Küme içgörülerini göster
        insights = recommender.get_cluster_insights()
        print_cluster_insights(insights)
        
        # Önerileri al ve göster
        recommendations, target_item = recommender.get_cluster_recommendations(
            args.user_id,
            args.num_recommendations
        )
        
        if not recommendations.empty and target_item:
            print(RecommendationFormatter.format_target_info(item_info=target_item))
            print(RecommendationFormatter.format_recommendations(
                recommendations,
                include_user_info=True,
                mode='küme'
            ))
    
    elif args.mode == 'user':
        print("\nKullanıcı bazlı öneriler hazırlanıyor...")
        recommender = UserBasedRecommender(user_df, item_df)
        recommendations, target_info = recommender.get_recommendations(
            args.user_id,
            args.num_recommendations
        )
        
        if not recommendations.empty and target_info:
            print(RecommendationFormatter.format_target_info(user_info=target_info))
            print(RecommendationFormatter.format_recommendations(
                recommendations,
                include_user_info=True,
                mode='kullanıcı'
            ))
    
    else:  # item mode
        print("\nÜrün bazlı öneriler hazırlanıyor...")
        recommender = ItemBasedRecommender(user_df, item_df)
        recommendations, target_item = recommender.get_recommendations(
            args.user_id,
            args.num_recommendations
        )
        
        if not recommendations.empty and target_item:
            print(RecommendationFormatter.format_target_info(item_info=target_item))
            print(RecommendationFormatter.format_recommendations(
                recommendations,
                include_user_info=False,
                mode='ürün'
            ))


if __name__ == "__main__":
    main()
