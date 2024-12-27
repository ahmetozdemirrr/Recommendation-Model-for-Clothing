import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


def prepare_kmeans_data(data_path):
    """
    Bu fonksiyon, bir veri kümesini K-Means algoritmasına uygun bir şekilde hazırlamak için:
    1. Kategorik sütunları One-Hot Encoding ile dönüştürür.
    2. Numerik sütunları Min-Max Scaling ile ölçeklendirir.
    3. Tüm özellikleri tek bir matris halinde birleştirir.
    
    Parametre:
        data_path (str): Veri kümesinin dosya yolu.
    
    Dönüş:
        df (pd.DataFrame): Orijinal veri çerçevesi.
        feature_matrix (np.ndarray): Hazırlanmış özellik matrisi.
    """
    # Veriyi yükle
    # Veri kümesi pandas DataFrame olarak yüklenir. 
    # DataFrame, kategorik ve numerik sütunları ayrıştırmak için kullanılacak.
    df = pd.read_csv(data_path)

    # Kullanılacak sütunlar
    # Bu sütunlar, K-Means algoritması için veriyi dönüştürmekte kullanılacaktır.
    # categorical_cols: One-Hot Encoding ile dönüştürülecek kategorik sütunlar.
    # numeric_cols: Min-Max Scaling ile ölçeklendirilecek sayısal sütunlar.
    categorical_cols = ["Size", "Category", "Frequency of Purchases", "Item Purchased", "Color"]
    numeric_cols = ["Age", "Previous Purchases"]

# *******************************************************************
# önem sırası ata... (kullanacağımız sütunların önemleri farklı olsun ...weighted...)
# *******************************************************************
# oneHotEncoder ayarla... (tüm sütunlara ohe uygulamaya gerek olmayabilir, bazılarına kendimiz değer atayabiliriz)
# *******************************************************************
# accuracy ölç...
# *******************************************************************
# k-means: iki model üstünde çalışsın
#   1 - kullanıcı benzerliği üzerine kümeleme yapsın
#   2 - kullanıcının aldığı ürüne benzer ürünleri kümeleme yapsın
# ardından cosine_similarity ile en yüksek puanlı ürünleri listele
# dataseti kullanıcı ve ürün bazlı olarak ikiye ayıralım
# *******************************************************************

    # Kategorik sütunlar için One-Hot Encoding
    # One-Hot Encoding, kategorik değerleri 0 ve 1'den oluşan binary vektörlere dönüştürür.
    # Bu işlem, modellerin kategorik verilerle çalışabilmesi için gereklidir.
    # `sparse_output=False`: Çıktının yoğun matris (dense) olarak döndürülmesini sağlar.
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_categorical = ohe.fit_transform(df[categorical_cols])

    # Numerik sütunlar için Min-Max Scaling
    # Min-Max Scaling, her sayısal özelliği 0 ile 1 arasında bir ölçeğe dönüştürür.
    # Bu işlem, farklı birimlerdeki sayısal özelliklerin karşılaştırılabilir hale gelmesini sağlar.
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_numeric = scaler.fit_transform(df[numeric_cols])

    # Özellikleri birleştirme
    # Kategorik ve sayısal veriler birleştirilerek nihai özellik matrisi oluşturulur.
    # Bu matris, K-Means gibi algoritmaların girdi olarak kullanabileceği yapıdadır.
    feature_matrix = np.hstack([encoded_categorical, scaled_numeric])

    # Orijinal veri çerçevesi ve özellik matrisi döndürülür
    return df, feature_matrix



def recommend_kmeans(user_id, data_path, n_clusters=5, top_n=3):
    """
    Kullanıcı ID'sine göre K-Means kümeleme ile ürün önerisi yapan fonksiyon.
    1. Veri hazırlanır ve K-Means modeli eğitilir.
    2. Kullanıcının ait olduğu küme belirlenir.
    3. Aynı kümedeki diğer kullanıcılarla benzerlik hesaplanır.
    4. En benzer kullanıcıların satın aldığı ürünler önerilir.

    Parametreler:
        user_id (int): Öneri yapılacak kullanıcı ID'si.
        data_path (str): Veri kümesinin dosya yolu.
        n_clusters (int): K-Means modeli için kullanılacak küme sayısı (default=5).
        top_n (int): Önerilecek ürün sayısı (default=3).
    
    Dönüş:
        recommendations (pd.DataFrame): Kullanıcıya önerilen ürünler.
    """
    # 1. Veriyi hazırla
    # Veri hazırlığı için prepare_kmeans_data fonksiyonu çağrılır.
    # Bu fonksiyon, kategorik veriler için One-Hot Encoding ve sayısal veriler için Min-Max Scaling işlemlerini içerir.
    df, feature_matrix = prepare_kmeans_data(data_path)

    # 2. K-Means modelini eğit
    # KMeans algoritması, veriyi n_clusters kadar kümeye böler.
    # random_state=42 ile modelin tekrar eden çalışmalarda aynı sonuçları üretmesi sağlanır.
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(feature_matrix)

    # 3. Kullanıcının ait olduğu kümeyi bul
    # Kullanıcının küme bilgisi alınır. Eğer kullanıcı veri kümesinde yoksa hata fırlatılır.
    if user_id not in df["Customer ID"].values:
        raise ValueError(f"User ID {user_id} veri kümesinde bulunamadı.")
    
    # Kullanıcının bulunduğu küme ve veri kümesindeki indeksi belirlenir.
    user_cluster = df[df["Customer ID"] == user_id]["Cluster"].values[0]
    user_index = df[df["Customer ID"] == user_id].index[0]

    # 4. Aynı kümedeki diğer kullanıcıları bul
    # Kullanıcının bulunduğu kümede yer alan diğer kullanıcılar alınır.
    # cluster_users: Aynı kümede bulunan kullanıcıların bir alt veri çerçevesi.
    # cluster_features: Aynı kümedeki kullanıcıların özellik matrisleri.
    cluster_users = df[df["Cluster"] == user_cluster].copy()
    cluster_features = feature_matrix[df["Cluster"] == user_cluster]

    # 5. Kullanıcı ile diğer küme üyeleri arasındaki benzerlikleri hesapla
    # cosine_similarity ile kullanıcı ile diğer kullanıcılar arasındaki benzerlikler hesaplanır.
    # Benzerlik skorları, özellik matrisleri üzerinden yapılır.
    user_features = feature_matrix[user_index].reshape(1, -1)
    similarities = cosine_similarity(user_features, cluster_features).flatten()

    # 6. Benzerlik skorlarını kümeye ekle
    # cluster_users tablosuna Similarity sütunu eklenir.
    # Kullanıcı hariç diğer küme üyeleri benzerlik skorlarına göre sıralanır.
    cluster_users.loc[:, "Similarity"] = similarities
    cluster_users = cluster_users[cluster_users["Customer ID"] != user_id]
    cluster_users = cluster_users.sort_values(by="Similarity", ascending=False)

    # 7. En benzer kullanıcıların ürünlerini öner
    # Sıralanan küme üyelerinden ilk top_n kadar ürün alınır ve öneri olarak döndürülür.
    recommendations = cluster_users.head(top_n)
    return recommendations


def save_cluster_assignments(data_path, output_path="cluster_assignments.csv", n_clusters=5):
    """
    Verilen veri kümesindeki her kullanıcıyı K-Means modeliyle kümelere atayan fonksiyon.
    Küme atamaları bir CSV dosyasına kaydedilir.

    Parametreler:
        data_path (str): Veri kümesinin dosya yolu.
        output_path (str): Küme atamalarının kaydedileceği dosyanın yolu (varsayılan: "cluster_assignments.csv").
        n_clusters (int): K-Means modeli için kullanılacak küme sayısı (default=5).

    İşlem:
        - Veri hazırlanır.
        - K-Means modeliyle her kullanıcı bir kümeye atanır.
        - Küme bilgileriyle birlikte tüm veri bir CSV dosyasına kaydedilir.
    """
    # 1. Veriyi hazırla
    # prepare_kmeans_data fonksiyonu:
    # - Kategorik sütunlar için One-Hot Encoding
    # - Numerik sütunlar için Min-Max Scaling uygular
    df, encoded_data = prepare_kmeans_data(data_path)

    # 2. K-Means modelini eğit
    # KMeans modeline veriyi vererek her kullanıcıyı bir kümeye atar.
    # Atamalar, `Cluster` adında yeni bir sütunda tutulur.
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(encoded_data)

    # 3. Sonuçları CSV dosyasına kaydet
    # Veri çerçevesine küme bilgileri eklendikten sonra belirtilen dosya yoluna kaydedilir.
    df.to_csv(output_path, index=False)
    print(f"Küme atamaları '{output_path}' dosyasına kaydedildi.")



if __name__ == "__main__":
    import argparse

    # Komut satırı argümanlarını tanımlayan bir nesne oluştur
    parser = argparse.ArgumentParser(description="K-Means Clustering ile öneri")
    
    # Kullanıcı ID'sini alan bir argüman
    parser.add_argument("--user_id", type=int, help="Kullanıcı ID'si")
    # Önerilecek ürün sayısını belirten bir argüman (varsayılan: 3)
    parser.add_argument("--top_n", type=int, default=3, help="Öneri sayısı")
    # Küme atamalarını kaydetmek için bir bayrak (flag)
    parser.add_argument("--save-clusters", action="store_true", help="Küme atamalarını kaydet")
    
    # Komut satırı argümanlarını işle
    args = parser.parse_args()

    # Veri dosyasının yolu
    data_path = "./data/shopping_trends_updated.csv"

    # Kullanıcı seçimi: Küme atamaları kaydet veya öneri üret
    if args.save_clusters:
        # Küme atamaları kaydet
        save_cluster_assignments(data_path)
    elif args.user_id:
        # Kullanıcı ID'sine göre öneri üret
        recommendations = recommend_kmeans(args.user_id, data_path, top_n=args.top_n)
        print(f"Kullanıcı {args.user_id} için önerilen ürünler:")
        # Önerilen ürünlerin yalnızca belirli sütunlarını yazdır
        print(recommendations[["Item Purchased", "Category", "Purchase Amount (USD)", "Color"]])
