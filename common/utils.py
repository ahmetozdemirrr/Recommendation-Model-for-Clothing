# utils.py - düzeltilmiş versiyon

class RecommendationFormatter:
    """Öneri sistemlerinin çıktılarını formatlayan sınıf"""
    
    @staticmethod
    def _format_item_info(item_data, similarity_score=None, index=None):
        """Ürün bilgilerini formatlar"""
        
        header = f"\n{'='*20} Öneri {index+1 if index is not None else ''} {'='*20}" if index is not None else "\n" + "="*50
        
        output = [
            header,
            f"Ürün Bilgileri:",
            f"  - İsim: {item_data.get('Item Purchased', 'Bilinmiyor')}",
            f"  - Kategori: {item_data.get('Category', 'Bilinmiyor')}",
            f"  - Renk: {item_data.get('Color', 'Bilinmiyor')}",
            f"  - Sezon: {item_data.get('Season', 'Bilinmiyor')}",
            f"  - Fiyat: ${item_data.get('Purchase Amount', 0):.2f}"
        ]
        
        # Sadece user-based ve item-based modeller için benzerlik skorunu göster
        if 'Similarity' in item_data and similarity_score is not None:
            output.append(f"  - Benzerlik Skoru: {similarity_score:.4f}")
            
        if item_data.get('User_Cluster') is not None:
            output.extend([
                f"  - Kullanıcı Kümesi: {item_data.get('User_Cluster')}",
                f"  - Ürün Kümesi: {item_data.get('Item_Cluster')}"
            ])
            
        return "\n".join(output)


    @staticmethod
    def _format_user_info(user_data):
        """Kullanıcı bilgilerini formatlar"""
        
        return "\n".join([
            "\nKullanıcı Bilgileri:",
            f"  - Yaş: {user_data.get('User_Age', user_data.get('Age', 'Bilinmiyor'))}",
            f"  - Cinsiyet: {user_data.get('User_Gender', user_data.get('Gender', 'Bilinmiyor'))}",
            f"  - Konum: {user_data.get('User_Location', user_data.get('Location', 'Bilinmiyor'))}",
            f"  - Beden: {user_data.get('User_Size', user_data.get('Size', 'Bilinmiyor'))}",
            f"  - Önceki Alışverişler: {user_data.get('User_Previous_Purchases', user_data.get('Previous Purchases', 'Bilinmiyor'))}",
            f"  - Alışveriş Sıklığı: {user_data.get('User_Frequency', user_data.get('Frequency of Purchases', 'Bilinmiyor'))}"
        ])


    @classmethod
    def format_target_info(cls, user_info=None, item_info=None):
        """Hedef kullanıcı ve/veya ürün bilgilerini formatlar"""
        
        output = ["\n" + "="*50 + "\nHEDEF BİLGİLERİ:"]
        
        if item_info:
            output.append(cls._format_item_info(item_info))
        if user_info:
            output.append(cls._format_user_info(user_info))
            
        output.append("="*50)
        return "\n".join(output)


    @classmethod
    def format_recommendation(cls, recommendation, index=None, include_user_info=False):
        """Tek bir öneriyi formatlar"""
        
        similarity_score = recommendation.get('Similarity')
        output = [cls._format_item_info(recommendation, similarity_score, index)]
        
        if include_user_info:
            output.append(cls._format_user_info(recommendation))
            
        return "\n".join(output)


    @classmethod
    def format_recommendations(cls, recommendations_df, include_user_info=False, mode=''):
        """Tüm önerileri formatlar"""
        
        mode_str = f"{mode.upper()} BAZLI " if mode else ""
        output = [f"\n{'='*50}\n{mode_str}ÖNERİLER:"]
        
        for idx, recommendation in recommendations_df.iterrows():
            output.append(cls.format_recommendation(
                recommendation.to_dict(),
                index=idx,
                include_user_info=include_user_info
            ))
            
        output.append("="*50)
        return "\n".join(output)
