import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import (
    USER_CATEGORICAL_FEATURES, USER_NUMERICAL_FEATURES,
    PRODUCT_CATEGORICAL_FEATURES, PRODUCT_NUMERICAL_FEATURES,
    REAL_USER_CATEGORICAL_FEATURES, REAL_USER_NUMERICAL_FEATURES,
    REAL_PRODUCT_CATEGORICAL_FEATURES, REAL_PRODUCT_NUMERICAL_FEATURES,
    PRODUCT_TEXT_FEATURES, RANDOM_SEED, PREPROCESSOR_PATH, DATA_SOURCE
)


class DataPreprocessor:
    def __init__(self, use_real_data=True):
        self.use_real_data = use_real_data
        self.user_cat_encoders = {}
        self.user_num_scaler = StandardScaler()
        self.prod_cat_encoders = {}
        self.prod_num_scaler = StandardScaler()
        self.user_cat_dims = {}
        self.prod_cat_dims = {}
        self.fitted = False
        self._user_cat_features = REAL_USER_CATEGORICAL_FEATURES if use_real_data else USER_CATEGORICAL_FEATURES
        self._user_num_features = REAL_USER_NUMERICAL_FEATURES if use_real_data else USER_NUMERICAL_FEATURES
        self._prod_cat_features = REAL_PRODUCT_CATEGORICAL_FEATURES if use_real_data else PRODUCT_CATEGORICAL_FEATURES
        self._prod_num_features = REAL_PRODUCT_NUMERICAL_FEATURES if use_real_data else PRODUCT_NUMERICAL_FEATURES

    def fit(self, users_df, products_df):
        if not hasattr(self, "_user_cat_features"):
            self._user_cat_features = USER_CATEGORICAL_FEATURES
            self._user_num_features = USER_NUMERICAL_FEATURES
            self._prod_cat_features = PRODUCT_CATEGORICAL_FEATURES
            self._prod_num_features = PRODUCT_NUMERICAL_FEATURES
        for col in self._user_cat_features:
            if col not in users_df.columns:
                users_df[col] = 0
            le = LabelEncoder()
            le.fit(users_df[col].astype(str).values)
            self.user_cat_encoders[col] = le
            self.user_cat_dims[col] = len(le.classes_)

        for col in self._prod_cat_features:
            if col not in products_df.columns:
                products_df[col] = "unknown"
            le = LabelEncoder()
            le.fit(products_df[col].astype(str).values)
            self.prod_cat_encoders[col] = le
            self.prod_cat_dims[col] = len(le.classes_)

        user_num_cols = [c for c in self._user_num_features if c in users_df.columns]
        if user_num_cols:
            self.user_num_scaler.fit(users_df[user_num_cols].fillna(0).values)
        else:
            self.user_num_scaler.fit(np.zeros((len(users_df), len(self._user_num_features))))

        prod_num_cols = [c for c in self._prod_num_features if c in products_df.columns]
        if prod_num_cols:
            self.prod_num_scaler.fit(products_df[prod_num_cols].fillna(0).values)
        else:
            self.prod_num_scaler.fit(np.zeros((len(products_df), len(self._prod_num_features))))
        self.fitted = True

    def transform_user(self, user_row):
        if not hasattr(self, "_user_cat_features"):
            self._user_cat_features = USER_CATEGORICAL_FEATURES
            self._user_num_features = USER_NUMERICAL_FEATURES
            self._prod_cat_features = PRODUCT_CATEGORICAL_FEATURES
            self._prod_num_features = PRODUCT_NUMERICAL_FEATURES
        cat_encoded = []
        for col in self._user_cat_features:
            val = str(user_row.get(col, 0))
            le = self.user_cat_encoders.get(col)
            if le is None:
                cat_encoded.append(0)
                continue
            try:
                encoded = le.transform([val])[0]
            except ValueError:
                encoded = 0
            cat_encoded.append(encoded)
        user_vals = [float(user_row.get(col, 0) or 0) for col in self._user_num_features]
        num_scaled = self.user_num_scaler.transform(np.array(user_vals).reshape(1, -1))[0]
        return np.array(cat_encoded), num_scaled

    def transform_product(self, prod_row):
        if not hasattr(self, "_prod_cat_features"):
            self._user_cat_features = USER_CATEGORICAL_FEATURES
            self._user_num_features = USER_NUMERICAL_FEATURES
            self._prod_cat_features = PRODUCT_CATEGORICAL_FEATURES
            self._prod_num_features = PRODUCT_NUMERICAL_FEATURES
        cat_encoded = []
        for col in self._prod_cat_features:
            val = str(prod_row.get(col, "unknown"))
            le = self.prod_cat_encoders.get(col)
            if le is None:
                cat_encoded.append(0)
                continue
            try:
                encoded = le.transform([val])[0]
            except ValueError:
                encoded = 0
            cat_encoded.append(encoded)
        prod_vals = [float(prod_row.get(col, 0) or 0) for col in self._prod_num_features]
        num_scaled = self.prod_num_scaler.transform(np.array(prod_vals).reshape(1, -1))[0]
        return np.array(cat_encoded), num_scaled

    def save(self, path=None):
        path = path or PREPROCESSOR_PATH
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path=None):
        path = path or PREPROCESSOR_PATH
        with open(path, "rb") as f:
            return pickle.load(f)


def _load_data_mysql(user_path, product_path, train_path):
    from data.database import load_users, load_products, load_train_data, check_connection
    if not check_connection():
        return None, None, None
    users_df = load_users()
    products_df = load_products()
    train_df = load_train_data()
    if (users_df is not None and len(users_df) > 0 and
        products_df is not None and len(products_df) > 0 and
        train_df is not None and len(train_df) > 0):
        print("[Preprocessing] Data loaded from MySQL.")
        return users_df, products_df, train_df
    return None, None, None


def _load_data_csv(user_path, product_path, train_path):
    users_df = pd.read_csv(user_path, encoding="utf-8-sig")
    products_df = pd.read_csv(product_path, encoding="utf-8-sig")
    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    print("[Preprocessing] Data loaded from CSV.")
    return users_df, products_df, train_df


def load_and_preprocess_data(user_path=None, product_path=None, train_path=None):
    from config import USER_DATA_PATH, PRODUCT_DATA_PATH, TRAIN_DATA_PATH
    user_path = user_path or USER_DATA_PATH
    product_path = product_path or PRODUCT_DATA_PATH
    train_path = train_path or TRAIN_DATA_PATH

    users_df, products_df, train_df = _load_data_mysql(user_path, product_path, train_path)
    if users_df is None:
        users_df, products_df, train_df = _load_data_csv(user_path, product_path, train_path)

    for df in [users_df, products_df]:
        df.drop_duplicates(inplace=True)

    is_real_data = DATA_SOURCE == "real"
    use_real_data = is_real_data

    if use_real_data and "user_id" in train_df.columns and "product_id" in train_df.columns:
        user_cat_feats = REAL_USER_CATEGORICAL_FEATURES
        user_num_feats = REAL_USER_NUMERICAL_FEATURES
        prod_cat_feats = REAL_PRODUCT_CATEGORICAL_FEATURES
        prod_num_feats = REAL_PRODUCT_NUMERICAL_FEATURES

        all_user_cat = [c for c in user_cat_feats if c in users_df.columns]
        all_user_num = [c for c in user_num_feats if c in users_df.columns]
        all_prod_cat = [c for c in prod_cat_feats if c in products_df.columns]
        all_prod_num = [c for c in prod_num_feats if c in products_df.columns]

        train_df = train_df.merge(users_df[["user_id"] + all_user_cat + all_user_num], on="user_id", how="left")
        train_df = train_df.merge(products_df[["product_id"] + all_prod_cat + all_prod_num], on="product_id", how="left")

        for col in all_user_num + all_user_cat:
            train_df[col] = train_df[col].fillna(0)
        for col in all_prod_num + all_prod_cat:
            train_df[col] = train_df[col].fillna(0)
    else:
        is_real_data = False
        user_cat_feats = USER_CATEGORICAL_FEATURES
        user_num_feats = USER_NUMERICAL_FEATURES
        prod_cat_feats = PRODUCT_CATEGORICAL_FEATURES
        prod_num_feats = PRODUCT_NUMERICAL_FEATURES

    for col in user_num_feats + user_cat_feats:
        if col in users_df.columns and pd.api.types.is_numeric_dtype(users_df[col]):
            users_df[col] = users_df[col].fillna(users_df[col].median())
        elif col in users_df.columns:
            mode_vals = users_df[col].mode()
            users_df[col] = users_df[col].fillna(mode_vals[0] if len(mode_vals) > 0 else "未知")

    for col in prod_num_feats + prod_cat_feats:
        if col in products_df.columns and pd.api.types.is_numeric_dtype(products_df[col]):
            products_df[col] = products_df[col].fillna(products_df[col].median())
        elif col in products_df.columns:
            mode_vals = products_df[col].mode()
            products_df[col] = products_df[col].fillna(mode_vals[0] if len(mode_vals) > 0 else "未知")

    preprocessor = DataPreprocessor(use_real_data=is_real_data)
    preprocessor.fit(users_df, products_df)

    active_user_cat = [c for c in user_cat_feats if c in preprocessor.user_cat_encoders]
    active_user_num = [c for c in user_num_feats if c in users_df.columns]
    active_prod_cat = [c for c in prod_cat_feats if c in preprocessor.prod_cat_encoders]
    active_prod_num = [c for c in prod_num_feats if c in products_df.columns]

    user_cat_map = {}
    user_num_map = {}
    for _, urow in users_df.iterrows():
        uid = str(urow.get("user_id", ""))
        u_cat, u_num = preprocessor.transform_user({
            **{c: urow.get(c, 0) for c in active_user_cat},
            **{c: urow.get(c, 0) for c in active_user_num},
        })
        user_cat_map[uid] = u_cat
        user_num_map[uid] = u_num

    prod_cat_map = {}
    prod_num_map = {}
    for _, prow in products_df.iterrows():
        pid = str(prow.get("product_id", ""))
        p_cat, p_num = preprocessor.transform_product({
            **{c: prow.get(c, 0) for c in active_prod_cat},
            **{c: prow.get(c, 0) for c in active_prod_num},
        })
        prod_cat_map[pid] = p_cat
        prod_num_map[pid] = p_num

    max_samples = 50000
    if len(train_df) > max_samples:
        train_df = train_df.sample(n=max_samples, random_state=RANDOM_SEED)
        print(f"[Preprocessing] Sampled {max_samples} from {len(train_df) if 'original_len' not in dir() else '?'} training samples")

    X_cat, X_num_user, X_num_prod, y = [], [], [], []

    for _, row in train_df.iterrows():
        uid = str(row.get("user_id", ""))
        pid = str(row.get("product_id", ""))
        u_cat = user_cat_map.get(uid, np.zeros(len(active_user_cat), dtype=np.int64))
        u_num = user_num_map.get(uid, np.zeros(len(active_user_num), dtype=np.float32))
        p_cat = prod_cat_map.get(pid, np.zeros(len(active_prod_cat), dtype=np.int64))
        p_num = prod_num_map.get(pid, np.zeros(len(active_prod_num), dtype=np.float32))
        X_cat.append(np.concatenate([u_cat, p_cat]))
        X_num_user.append(u_num)
        X_num_prod.append(p_num)
        y.append(float(row.get("label", 0)))

    X_cat = np.array(X_cat, dtype=np.int64)
    X_num_user = np.array(X_num_user, dtype=np.float32)
    X_num_prod = np.array(X_num_prod, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=(y > 0.5).astype(int), random_state=RANDOM_SEED)

    result = {
        "X_cat_train": X_cat[train_idx], "X_num_user_train": X_num_user[train_idx],
        "X_num_prod_train": X_num_prod[train_idx], "y_train": y[train_idx],
        "X_cat_test": X_cat[test_idx], "X_num_user_test": X_num_user[test_idx],
        "X_num_prod_test": X_num_prod[test_idx], "y_test": y[test_idx],
        "preprocessor": preprocessor, "users_df": users_df, "products_df": products_df,
    }
    return result
