import os

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

USER_DATA_PATH = os.path.join(DATA_DIR, "users.csv")
PRODUCT_DATA_PATH = os.path.join(DATA_DIR, "products.csv")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_data.csv")
PREPROCESSOR_PATH = os.path.join(CHECKPOINT_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "deepfm_model.pt")
TRAIN_HISTORY_PATH = os.path.join(CHECKPOINT_DIR, "training_history.json")
EMBEDDING_CACHE_PATH = os.path.join(CHECKPOINT_DIR, "product_embeddings.npy")

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TEXT_GEN_MODEL_NAME = "uer/gpt2-chinese-cluecorpussmall"

def _get_mysql_config():
    try:
        import streamlit as st
        return {
            "host": st.secrets["tidb_host"],
            "port": st.secrets.get("tidb_port", 4000),
            "user": st.secrets["tidb_user"],
            "password": st.secrets["tidb_password"],
            "database": st.secrets.get("tidb_database", "insurance_recommender"),
            "charset": "utf8mb4",
            "ssl": {
                "ca": os.path.join(BASE_DIR, "data", "tidb_ca.pem"),
            },
        }
    except Exception:
        pass

    env = os.environ
    host = env.get("TIDB_HOST", "gateway01.ap-northeast-1.prod.aws.tidbcloud.com")
    user = env.get("TIDB_USER", "4BaT2Z4EUDpj4XB.root")
    pw = env.get("TIDB_PASSWORD", "")

    return {
        "host": host,
        "port": int(env.get("TIDB_PORT", "4000")),
        "user": user,
        "password": pw,
        "database": env.get("TIDB_DATABASE", "insurance_recommender"),
        "charset": "utf8mb4",
        "ssl": {
            "ca": os.path.join(BASE_DIR, "data", "tidb_ca.pem"),
        },
    }

MYSQL_CONFIG = _get_mysql_config()

if _TORCH_AVAILABLE:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = None

DEFAULT_MODEL_PARAMS = {
    "embedding_dim": 16,
    "dnn_hidden_units": [256, 128, 64],
    "dnn_dropout": 0.2,
    "l2_reg": 1e-5,
    "learning_rate": 1e-3,
    "batch_size": 256,
    "epochs": 30,
    "early_stopping_patience": 5,
    "text_embedding_dim": 384,
}

TOP_N = 5
RANDOM_SEED = 42

DATA_SOURCE = "real"

USER_CATEGORICAL_FEATURES = [
    "gender", "family_structure", "health_status", "occupation",
    "city_tier", "has_house", "has_car", "education"
]

USER_NUMERICAL_FEATURES = [
    "age", "annual_income", "family_members"
]

REAL_USER_CATEGORICAL_FEATURES = [
    "social_category_idx", "age_group_idx", "religion_idx",
    "marital_status_idx"
]

REAL_USER_NUMERICAL_FEATURES = [
    "avg_income_level", "car_ownership", "household_size",
    "has_caravan"
] + [f"coil_socio_{i+1}" for i in range(43)]

PRODUCT_CATEGORICAL_FEATURES = [
    "category", "company", "coverage_type", "payment_method"
]

PRODUCT_NUMERICAL_FEATURES = [
    "coverage_amount", "annual_premium", "coverage_years",
    "waiting_period_days", "deductible"
]

REAL_PRODUCT_CATEGORICAL_FEATURES = ["category"]

REAL_PRODUCT_NUMERICAL_FEATURES = [
    "coverage_amount", "annual_premium"
]

PRODUCT_TEXT_FEATURES = ["product_name", "features", "coverage_detail"]

INSURANCE_CATEGORIES = ["医疗险", "重疾险", "意外险", "寿险", "年金险", "财产险", "旅行险"]

HEALTH_STATUS_MAP = {"健康": 0, "亚健康": 1, "有慢性病": 2, "有重大病史": 3}
