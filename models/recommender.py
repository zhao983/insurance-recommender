import os
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from config import (
    DEVICE, TOP_N, MODEL_PATH, PREPROCESSOR_PATH, TRAIN_HISTORY_PATH,
    USER_CATEGORICAL_FEATURES, USER_NUMERICAL_FEATURES,
    PRODUCT_CATEGORICAL_FEATURES, PRODUCT_NUMERICAL_FEATURES,
    REAL_USER_CATEGORICAL_FEATURES, REAL_USER_NUMERICAL_FEATURES,
    REAL_PRODUCT_CATEGORICAL_FEATURES, REAL_PRODUCT_NUMERICAL_FEATURES,
    PRODUCT_TEXT_FEATURES, DEFAULT_MODEL_PARAMS
)
from data.preprocessing import load_and_preprocess_data, DataPreprocessor
from data.generator import generate_users, generate_products, generate_training_data
from models.deepfm import DeepFM, DeepFMTrainer, RecommendDataset
from models.embedding import TextEmbedder
from models.text_generator import TextGenerator


class InsuranceRecommender:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.text_embedder = None
        self.text_generator = None
        self.users_df = None
        self.products_df = None
        self.product_embeddings = None
        self.user_cat_dims = []
        self.prod_cat_dims = []
        self.model_params = dict(DEFAULT_MODEL_PARAMS)
        self.training_history = []
        self._initialized = False

    def initialize(self, force_retrain=False):
        print("=" * 60)
        print("Insurance Recommender System - Initialization")
        print("=" * 60)

        self._ensure_data_exists()
        self._load_data()

        self.text_embedder = TextEmbedder()
        self.product_embeddings = self.text_embedder.encode_products(self.products_df)

        self.text_generator = TextGenerator()

        if not force_retrain and os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
            print("Loading pre-trained model (fast path)...")
            self.preprocessor = DataPreprocessor.load()
            self.user_cat_dims = list(self.preprocessor.user_cat_dims.values())
            self.prod_cat_dims = list(self.preprocessor.prod_cat_dims.values())
            all_cat_dims = self.user_cat_dims + self.prod_cat_dims

            state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            saved_dnn_input_dim = state["dnn.0.weight"].shape[1]
            saved_num_input_dim = saved_dnn_input_dim - len(all_cat_dims) * self.model_params["embedding_dim"]

            self.model = DeepFM(all_cat_dims, saved_num_input_dim, self.model_params)
            self.model.load_state_dict(state)
            self.model.to(DEVICE)
            self.model.eval()

            if os.path.exists(TRAIN_HISTORY_PATH):
                with open(TRAIN_HISTORY_PATH, "r", encoding="utf-8") as f:
                    self.training_history = json.load(f)
            else:
                self._load_training_from_db()
        else:
            print("Training new model...")
            data = load_and_preprocess_data()
            self.preprocessor = data["preprocessor"]
            self.users_df = data["users_df"]
            self.products_df = data["products_df"]
            self.user_cat_dims = list(self.preprocessor.user_cat_dims.values())
            self.prod_cat_dims = list(self.preprocessor.prod_cat_dims.values())
            all_cat_dims = self.user_cat_dims + self.prod_cat_dims
            num_input_dim = (data["X_num_user_train"].shape[1] +
                             data["X_num_prod_train"].shape[1] +
                             self.text_embedder.dim)

            n_user_cat = len(self.preprocessor.user_cat_dims)
            train_emb = self._get_text_embeddings_for_samples(data["X_cat_train"], n_user_cat, data["X_num_prod_train"])
            test_emb = self._get_text_embeddings_for_samples(data["X_cat_test"], n_user_cat, data["X_num_prod_test"])
            self._train_model(data, all_cat_dims, num_input_dim, train_emb, test_emb)

        self._initialized = True
        print("Initialization complete!\n")

    def _ensure_data_exists(self):
        from config import USER_DATA_PATH, PRODUCT_DATA_PATH, TRAIN_DATA_PATH
        try:
            from data.database import check_connection, table_has_data
            if check_connection() and table_has_data("users_real"):
                return
        except Exception:
            pass
        if not all(os.path.exists(p) for p in [USER_DATA_PATH, PRODUCT_DATA_PATH, TRAIN_DATA_PATH]):
            print("Real data not found. Run: python -c 'from data.generator import main; main()'")
            print("Falling back to generating synthetic data...")
            try:
                from data.real_loader import (
                    load_uci_coil_training_data, build_user_profiles_from_coil,
                    build_products_from_coil, build_training_data_from_coil
                )
                coil_df = load_uci_coil_training_data()
                users = build_user_profiles_from_coil(coil_df)
                products = build_products_from_coil()
                train_data = build_training_data_from_coil(coil_df, users, products)
                users.to_csv(USER_DATA_PATH, index=False, encoding="utf-8-sig")
                products.to_csv(PRODUCT_DATA_PATH, index=False, encoding="utf-8-sig")
                train_data.to_csv(TRAIN_DATA_PATH, index=False, encoding="utf-8-sig")
                print("Real data downloaded and saved.")
            except Exception as e:
                print(f"Real data download failed: {e}, using synthetic data...")
                users = generate_users(5000)
                users.to_csv(USER_DATA_PATH, index=False, encoding="utf-8-sig")
                products = generate_products(80)
                products.to_csv(PRODUCT_DATA_PATH, index=False, encoding="utf-8-sig")
                train_data = generate_training_data(users, products, 30000)
                train_data.to_csv(TRAIN_DATA_PATH, index=False, encoding="utf-8-sig")
            print("Data generated.")

    def _load_data(self):
        from config import USER_DATA_PATH, PRODUCT_DATA_PATH
        try:
            from data.database import load_users, load_products, check_connection
            if check_connection():
                self.users_df = load_users()
                self.products_df = load_products()
                if (self.users_df is not None and len(self.users_df) > 0 and
                    self.products_df is not None and len(self.products_df) > 0):
                    print("[Recommender] Data loaded from MySQL.")
                    return
        except Exception:
            pass
        self.users_df = pd.read_csv(USER_DATA_PATH, encoding="utf-8-sig")
        self.products_df = pd.read_csv(PRODUCT_DATA_PATH, encoding="utf-8-sig")
        print("[Recommender] Data loaded from CSV.")

    def _get_text_embeddings_for_samples(self, X_cat, n_user_cat, X_num_prod):
        if self.product_embeddings is None:
            return np.zeros((len(X_cat), self.text_embedder.dim), dtype=np.float32)
        n_prod_cat = len(PRODUCT_CATEGORICAL_FEATURES)
        prod_cat = X_cat[:, n_user_cat:n_user_cat + n_prod_cat]
        idx = np.arange(len(X_cat)) % len(self.product_embeddings)
        return self.product_embeddings[idx]

    def _train_model(self, data, all_cat_dims, num_input_dim, train_emb, test_emb):
        train_dataset = RecommendDataset(
            data["X_cat_train"], data["X_num_user_train"],
            data["X_num_prod_train"], data["y_train"], train_emb
        )
        test_dataset = RecommendDataset(
            data["X_cat_test"], data["X_num_user_test"],
            data["X_num_prod_test"], data["y_test"], test_emb
        )
        train_loader = DataLoader(train_dataset, batch_size=self.model_params["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.model_params["batch_size"], shuffle=False)

        self.model = DeepFM(all_cat_dims, num_input_dim, self.model_params)
        trainer = DeepFMTrainer(self.model, self.model_params)
        history = trainer.fit(train_loader, test_loader)

        torch.save(self.model.state_dict(), MODEL_PATH)
        self.preprocessor.save()

        record = {
            "model_name": "DeepFM",
            "params": self.model_params,
            "train_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "final_val_auc": history["val_auc"][-1] if history["val_auc"] else None,
            "epochs_trained": len(history["train_loss"]),
        }
        self.training_history = [record]
        with open(TRAIN_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(self.training_history, f, ensure_ascii=False, indent=2)
        self._save_training_to_db(record)
        print(f"Model saved to {MODEL_PATH}")
        print(f"Training history saved to {TRAIN_HISTORY_PATH}")

    def _save_training_to_db(self, record):
        try:
            from data.database import insert_training_record, check_connection
            if check_connection():
                insert_training_record(record)
        except Exception:
            pass

    def _load_training_from_db(self):
        try:
            from data.database import load_training_history, check_connection
            if check_connection():
                records = load_training_history()
                if records:
                    self.training_history = records
        except Exception:
            pass

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.model_params:
                self.model_params[k] = v

    def _map_form_to_coil(self, form_profile, cat_feats, num_feats):
        import hashlib

        age = form_profile.get("age", 30)
        income = form_profile.get("annual_income", 80000)
        has_car_val = 1 if form_profile.get("has_car") == "有" else 0
        has_house_val = 1 if form_profile.get("has_house") == "有" else 0

        health = form_profile.get("health_status", "健康")
        health_score = {"健康": 1.0, "亚健康": 0.7, "有慢性病": 0.4, "有重大病史": 0.1}.get(health, 0.5)

        family = form_profile.get("family_structure", "单身")
        family_size = {"单身": 1, "单亲家庭": 2, "已婚无子女": 2, "已婚有子女": 4, "三代同堂": 5}.get(family, 2)
        household_size = form_profile.get("family_members", family_size)

        gender = form_profile.get("gender", "男")
        city = form_profile.get("city_tier", "二线城市")
        city_mult = {"一线城市": 1.5, "二线城市": 1.0, "三线城市": 0.7, "四线及以下": 0.5}.get(city, 1.0)
        occupation = form_profile.get("occupation", "白领")

        user_seed = int(hashlib.md5(
            f"{age}{income}{gender}{city}{occupation}{health}{family}{household_size}{has_car_val}{has_house_val}".encode()
        ).hexdigest(), 16) % 10000

        np.random.seed(user_seed)

        age_bucket = 0 if age < 30 else (1 if age < 40 else (2 if age < 50 else (3 if age < 60 else (4 if age < 70 else 5))))

        religion_idxs = list(range(5))
        np.random.shuffle(religion_idxs)
        religion_idx = religion_idxs[0]

        marital_idxs = list(range(4))
        np.random.shuffle(marital_idxs)
        if family in ("单身", "单亲家庭"):
            marital_idx = marital_idxs.index(3) if 3 in marital_idxs else 3
        elif family in ("已婚有子女", "已婚无子女"):
            marital_idx = marital_idxs.index(0) if 0 in marital_idxs else 0
        else:
            marital_idx = marital_idxs[0]

        social_category_idx = int(np.clip(int(income / 50000) + (has_car_val * 2) + (has_house_val * 2), 0, 11))

        result = {}
        for col in cat_feats:
            if col == "social_category_idx":
                result[col] = social_category_idx
            elif col == "age_group_idx":
                result[col] = age_bucket
            elif col == "religion_idx":
                result[col] = religion_idx
            elif col == "marital_status_idx":
                result[col] = marital_idx
            else:
                result[col] = int(np.random.randint(0, 5))

        income_level = income / (100000 * city_mult) * health_score
        car_own = has_car_val + np.random.normal(0, 0.1)
        hh_size = household_size + np.random.normal(0, 0.1)
        caravan = 0

        for col in num_feats:
            if col == "avg_income_level":
                result[col] = float(np.clip(income_level, 0, 200))
            elif col == "car_ownership":
                result[col] = float(np.clip(car_own, 0, 5))
            elif col == "household_size":
                result[col] = float(np.clip(hh_size, 1, 10))
            elif col == "has_caravan":
                result[col] = caravan
            elif col.startswith("coil_socio_"):
                result[col] = float(np.clip(np.random.normal(0.5, 0.2), 0, 1))
            else:
                result[col] = float(np.random.normal(0, 1))

        np.random.seed()
        return result

    def recommend_for_user(self, user_profile):
        if not self._initialized:
            raise RuntimeError("Recommender not initialized. Call initialize() first.")

        from config import DATA_SOURCE
        if DATA_SOURCE == "real":
            user_cat_feats = [c for c in REAL_USER_CATEGORICAL_FEATURES if c in self.preprocessor.user_cat_encoders]
            user_num_feats = [c for c in REAL_USER_NUMERICAL_FEATURES if c in self.users_df.columns]
            prod_cat_feats = [c for c in REAL_PRODUCT_CATEGORICAL_FEATURES if c in self.preprocessor.prod_cat_encoders]
            prod_num_feats = [c for c in REAL_PRODUCT_NUMERICAL_FEATURES if c in self.products_df.columns]
            user_row = self._map_form_to_coil(user_profile, user_cat_feats, user_num_feats)
        else:
            user_cat_feats = USER_CATEGORICAL_FEATURES
            user_num_feats = USER_NUMERICAL_FEATURES
            prod_cat_feats = PRODUCT_CATEGORICAL_FEATURES
            prod_num_feats = PRODUCT_NUMERICAL_FEATURES
            user_row = {col: user_profile.get(col, "") for col in USER_CATEGORICAL_FEATURES + USER_NUMERICAL_FEATURES}

        u_cat, u_num = self.preprocessor.transform_user(user_row)

        results = []
        for idx, (_, prod_row) in enumerate(self.products_df.iterrows()):
            p_cat, p_num = self.preprocessor.transform_product(prod_row)

            cat_input = torch.LongTensor(np.concatenate([u_cat, p_cat])).unsqueeze(0).to(DEVICE)
            num_input = torch.FloatTensor(
                np.concatenate([u_num, p_num, self.product_embeddings[idx]])
            ).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                score = self.model(cat_input, num_input).item()

            results.append({
                "product_id": prod_row.get("product_id", ""),
                "product_name": prod_row.get("product_name", ""),
                "category": prod_row.get("category", ""),
                "company": prod_row.get("company", ""),
                "annual_premium": prod_row.get("annual_premium", 0),
                "coverage_amount": prod_row.get("coverage_amount", 0),
                "features": prod_row.get("features", ""),
                "coverage_detail": prod_row.get("coverage_detail", ""),
                "match_score": round(score, 4),
            })

        results.sort(key=lambda x: x["match_score"], reverse=True)
        top_results = results[:TOP_N]

        for r in top_results:
            r["reason"] = self.text_generator.generate_reason(
                user_profile, r, r["match_score"]
            )

        return top_results

    def recommend_batch(self, users_list):
        all_results = []
        for user in users_list:
            recs = self.recommend_for_user(user)
            for rank, rec in enumerate(recs, 1):
                rec["rank"] = rank
                rec["user_id"] = user.get("user_id", "unknown")
                rec["age"] = user.get("age", "")
                rec["gender"] = user.get("gender", "")
                rec["health_status"] = user.get("health_status", "")
                rec["annual_income"] = user.get("annual_income", "")
            all_results.extend(recs)
        return all_results

    def get_model_performance(self):
        if not self.training_history:
            return None
        return self.training_history[-1]

    def get_category_stats(self):
        stats = {}
        for cat in self.products_df["category"].unique():
            cat_products = self.products_df[self.products_df["category"] == cat]
            stats[str(cat)] = {
                "product_count": len(cat_products),
                "avg_premium": float(cat_products["annual_premium"].mean()),
                "avg_coverage": float(cat_products["coverage_amount"].mean()),
            }
        return stats
