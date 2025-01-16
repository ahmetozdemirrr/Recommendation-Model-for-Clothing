# Makefile

# Python interpreter
PYTHON = python3

# Proje dizinleri
SRC_DIR = src
DATA_DIR = data

# Varsayılan değerler
USER_ID = 1
NUM_RECOMMENDATIONS = 3


# Hedefler
.PHONY: setup clean user item cluster evaluate help


# Ortamı hazırlama
setup:
	@echo "-> Gerekli bağımlılıklar yükleniyor..."
	$(PYTHON) -m pip install -r requirements.txt


# Kullanıcı bazlı öneri
user:
	@echo "-> Kullanıcı bazlı öneriler oluşturuluyor..."
	$(PYTHON) $(SRC_DIR)/main.py \
		--mode user \
		--user_id $(USER_ID) \
		--num_recommendations $(NUM_RECOMMENDATIONS)


# Ürün bazlı öneri
item:
	@echo "-> Ürün bazlı öneriler oluşturuluyor..."
	$(PYTHON) $(SRC_DIR)/main.py \
		--mode item \
		--user_id $(USER_ID) \
		--num_recommendations $(NUM_RECOMMENDATIONS)


# Küme bazlı öneri
cluster:
	@echo "-> Küme bazlı öneriler oluşturuluyor..."
	$(PYTHON) $(SRC_DIR)/main.py \
		--mode cluster \
		--user_id $(USER_ID) \
		--num_recommendations $(NUM_RECOMMENDATIONS)


evaluate:
	@echo "-> Öneri sistemleri değerlendiriliyor..."
	$(PYTHON) $(SRC_DIR)/main.py --evaluate --n_test_users $(N_TEST_USERS)


# Temizlik
clean:
	@echo "-> Geçici dosyalar temizleniyor..."
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete


# Yardım mesajı
help:
	@echo "Kullanılabilir komutlar:"
	@echo "  make setup                               - Gerekli bağımlılıkları yükler"
	@echo "  make user USER_ID=123                    - Kullanıcı bazlı öneriler oluşturur"
	@echo "  make item USER_ID=123                    - Ürün bazlı öneriler oluşturur"
	@echo "  make cluster USER_ID=123                 - Küme bazlı öneriler oluşturur"
	@echo "  make clean                               - Geçici dosyaları temizler"
	@echo "  make evaluate N_TEST_USERS=100           - Öneri sistemlerini değerlendirir"
	@echo ""
	@echo "Örnekler:"
	@echo "  make user USER_ID=123"
	@echo "  make user USER_ID=123 NUM_RECOMMENDATIONS=10"
	@echo "  make item USER_ID=123"
	@echo "  make cluster USER_ID=123"


# Varsayılan hedef
.DEFAULT_GOAL := help