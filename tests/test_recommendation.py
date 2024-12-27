import sys
import os
import pytest

# sys.path ekleme (gerekebilir)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preparation import prepare_data
from utils import compute_similarity, recommend_items_for_user


@pytest.mark.parametrize("user_id, num_recs", [(569, 3), (10, 5)])
def test_recommendation(user_id, num_recs):
    df, feature_matrix, ohe, numeric_cols, categorical_cols = prepare_data()
    sim_matrix = compute_similarity(feature_matrix)
    items = recommend_items_for_user(user_id, df, sim_matrix, num_recs)
    assert len(items) <= num_recs
    
    # Her bir önerilen item string olmalıs
    for it in items:
        assert isinstance(it, str)
