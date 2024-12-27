# Proje adı
PROJECT_NAME = Recommendation-Model

# Python interpreter
PYTHON = python3

# Gereksinim dosyası
REQUIREMENTS = requirements.txt

# Ana dizinler
SRC_DIR = src
DATA_DIR = data
TESTS_DIR = tests

# Hedefler

# Ortamı hazırlama (kütüphaneleri yükler)
.PHONY: setup
setup:
	@echo "-> Gerekli bağımlılıklar yükleniyor..."
	$(PYTHON) -m pip install -r $(REQUIREMENTS)


# Testleri çalıştırır
.PHONY: test
test:
	@echo "-> Testler çalıştırılıyor..."
	$(PYTHON) -m unittest discover $(TESTS_DIR)


# User-Based Collaborative Filtering modelini çalıştır
.PHONY: user-based
user-based:
	@echo "-> User-Based Collaborative Filtering modeli çalıştırılıyor..."
	$(PYTHON) src/user_based.py --user_id=$(USER_ID) --num_recommendations=$(NUM_RECOMMENDATIONS)


# Content-Based Filtering modelini çalıştır
.PHONY: content-based
content-based:
	@echo "-> Content-Based Filtering modeli çalıştırılıyor..."
	$(PYTHON) src/content_based.py --user_id=$(USER_ID) --top_n=$(TOP_N)


# K-Means Clustering modelini çalıştır
.PHONY: kmeans
kmeans:
	@echo "-> K-Means Clustering modeli çalıştırılıyor..."
	$(PYTHON) src/kmeans_clustering.py --user_id=$(USER_ID) --top_n=$(TOP_N)


# Kümeleri kaydet
.PHONY: save-clusters
save-clusters:
	@echo "-> Kümeler kaydediliyor..."
	$(PYTHON) src/kmeans_clustering.py --save-clusters



# Temizlik işlemleri (örneğin, geçici dosyaları siler)
.PHONY: clean
clean:
	@echo "-> Geçici dosyalar temizleniyor..."
	rm -rf __pycache__ */__pycache__
	rm -rf cluster_assignments.csv


# Yardım mesajı
.PHONY: help
help:
	@echo "Kullanılabilir komutlar:"
	@echo "  make setup    - Ortamı hazırlamak için gerekli bağımlılıkları yükler"
	@echo "  make test     - Testleri çalıştırır"
	@echo "  make run      - Öneri modelini çalıştırır"
	@echo "  make clean    - Geçici dosyaları temizler"
