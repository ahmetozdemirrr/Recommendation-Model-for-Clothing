# Makefile

# Python interpreter
PYTHON = python3

# Proje dizinleri
MAIN_DIR = src
DATA_DIR = data
MODELS_DIR = models
COMMON_DIR = common

# Python path ayarları (ana dizini Python path'e ekler)
PYTHONPATH = PYTHONPATH=.

# Varsayılan değerler
USER_ID = 1
NUM_RECOMMENDATIONS = 3
N_TEST_USERS = 4


# Hedefler
.PHONY: setup clean collaborative_user collaborative_item kmeans_hybrid evaluate help


# Ortamı hazırlama
setup:
	@echo "-> Gerekli bağımlılıklar yükleniyor..."
	$(PYTHON) -m pip install -r requirements.txt
	@echo "-> Proje dizin yapısı oluşturuluyor..."
	mkdir -p $(MODELS_DIR)/{collaborative_user,collaborative_item,kmeans_hybrid} $(COMMON_DIR)


# Collaborative User-Based öneri
collaborative_user:
	@echo "-> Collaborative User-Based öneriler oluşturuluyor..."
	$(PYTHONPATH) $(PYTHON) $(MAIN_DIR)/main.py \
		--mode user \
		--user_id $(USER_ID) \
		--num_recommendations $(NUM_RECOMMENDATIONS)


# Collaborative Item-Based öneri
collaborative_item:
	@echo "-> Collaborative Item-Based öneriler oluşturuluyor..."
	$(PYTHONPATH) $(PYTHON) $(MAIN_DIR)/main.py \
		--mode item \
		--user_id $(USER_ID) \
		--num_recommendations $(NUM_RECOMMENDATIONS)


# KMeans Hybrid öneri
kmeans_hybrid:
	@echo "-> KMeans Hybrid öneriler oluşturuluyor..."
	$(PYTHONPATH) $(PYTHON) $(MAIN_DIR)/main.py \
		--mode cluster \
		--user_id $(USER_ID) \
		--num_recommendations $(NUM_RECOMMENDATIONS)


# Değerlendirme
evaluate:
	@echo "-> Öneri sistemleri değerlendiriliyor..."
	$(PYTHONPATH) $(PYTHON) $(MAIN_DIR)/main.py \
		--evaluate \
		--n_test_users=$(N_TEST_USERS)


# Temizlik
clean:
	@echo "-> Geçici dosyalar temizleniyor..."
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete


# Yardım mesajı
help:
	@echo "Kullanılabilir komutlar:"
	@echo "  make setup                                      - Gerekli bağımlılıkları yükler"
	@echo "  make collaborative_user USER_ID=123             - Collaborative User-Based öneriler oluşturur"
	@echo "  make collaborative_item USER_ID=123             - Collaborative Item-Based öneriler oluşturur"
	@echo "  make kmeans_hybrid USER_ID=123                 - KMeans Hybrid öneriler oluşturur"
	@echo "  make clean                                      - Geçici dosyaları temizler"
	@echo "  make evaluate N_TEST_USERS=100                  - Öneri sistemlerini değerlendirir"
	@echo ""
	@echo "Örnekler:"
	@echo "  make collaborative_user USER_ID=123"
	@echo "  make collaborative_user USER_ID=123 NUM_RECOMMENDATIONS=10"
	@echo "  make collaborative_item USER_ID=123"
	@echo "  make kmeans_hybrid USER_ID=123"


# Varsayılan hedef
.DEFAULT_GOAL := help