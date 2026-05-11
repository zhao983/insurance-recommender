import pymysql
import numpy as np
import pandas as pd
import json
import time
from contextlib import contextmanager
from config import MYSQL_CONFIG, USER_CATEGORICAL_FEATURES, USER_NUMERICAL_FEATURES
from config import PRODUCT_CATEGORICAL_FEATURES, PRODUCT_NUMERICAL_FEATURES

_db = None


def _connect():
    cfg = dict(MYSQL_CONFIG)
    db_name = cfg.pop("database")
    ssl_cfg = cfg.pop("ssl", {})
    cfg_with_db = dict(cfg)
    cfg_with_db["database"] = db_name
    try:
        conn = pymysql.connect(
            **cfg_with_db,
            ssl=ssl_cfg if ssl_cfg else None,
            autocommit=False
        )
    except pymysql.err.OperationalError:
        conn = pymysql.connect(
            **cfg,
            ssl=ssl_cfg if ssl_cfg else None,
            autocommit=False
        )
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        conn.select_db(db_name)
    return conn


def _ensure_connection():
    global _db
    try:
        if _db is not None:
            _db.ping(reconnect=False)
        else:
            _db = _connect()
    except Exception:
        _db = _connect()


@contextmanager
def get_cursor():
    _ensure_connection()
    cur = _db.cursor()
    try:
        yield cur
        _db.commit()
    except Exception:
        _db.rollback()
        raise
    finally:
        cur.close()


def init_database():
    with get_cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users_real (
                user_id VARCHAR(16) PRIMARY KEY,
                history_purchases TEXT,
                social_category_idx INT,
                age_group_idx INT,
                religion_idx INT,
                marital_status_idx INT,
                avg_income_level DOUBLE,
                car_ownership DOUBLE,
                household_size DOUBLE,
                has_caravan INT,
                coil_socio_data JSON
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS products_real (
                product_id VARCHAR(8) PRIMARY KEY,
                product_name VARCHAR(64),
                category VARCHAR(16),
                company VARCHAR(32),
                coverage_amount INT,
                annual_premium INT,
                coverage_years INT DEFAULT 1,
                waiting_period_days INT DEFAULT 0,
                deductible INT DEFAULT 0,
                features TEXT,
                coverage_type VARCHAR(16) DEFAULT '标准版',
                payment_method VARCHAR(8) DEFAULT '年交',
                rating DOUBLE DEFAULT 4.0,
                coverage_detail TEXT
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS train_data_real (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(16),
                product_id VARCHAR(8),
                label INT,
                INDEX idx_user (user_id),
                INDEX idx_product (product_id),
                INDEX idx_label (label),
                UNIQUE KEY uk_user_product (user_id, product_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS training_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                model_name VARCHAR(64),
                params JSON,
                train_time DATETIME,
                final_train_loss DOUBLE,
                final_val_loss DOUBLE,
                final_val_auc DOUBLE,
                epochs_trained INT,
                data_source VARCHAR(32) DEFAULT 'real'
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
    print("[Database] Tables initialized successfully (real data schema).")


def table_has_data(table_name):
    try:
        with get_cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM `{table_name}`")
            return cur.fetchone()[0] > 0
    except Exception:
        return False


def table_exists(table_name):
    try:
        with get_cursor() as cur:
            cur.execute("SHOW TABLES LIKE %s", (table_name,))
            return cur.fetchone() is not None
    except Exception:
        return False


def insert_users(users_df):
    if not isinstance(users_df, pd.DataFrame) or len(users_df) == 0:
        return
    simple_cols = [c for c in users_df.columns
                   if not c.startswith("coil_socio_") and c != "history_purchases"]
    coil_cols = [c for c in users_df.columns if c.startswith("coil_socio_")]

    if coil_cols:
        placeholders = ", ".join(["%s"] * (len(simple_cols) + 1))
        sql = f"REPLACE INTO users_real ({', '.join(simple_cols)}, coil_socio_data) VALUES ({placeholders})"
        rows = []
        for _, row in users_df.iterrows():
            coil_data = {col: float(row.get(col, 0) or 0) for col in coil_cols}
            vals = [row.get(c, 0) if not isinstance(row.get(c, 0), float) or pd.notna(row.get(c, 0)) else 0 for c in simple_cols]
            vals.append(json.dumps(coil_data))
            rows.append(tuple(vals))
    else:
        placeholders = ", ".join(["%s"] * len(simple_cols))
        sql = f"REPLACE INTO users_real ({', '.join(simple_cols)}) VALUES ({placeholders})"
        rows = [tuple(row.get(c, 0) if not isinstance(row.get(c, 0), float) or pd.notna(row.get(c, 0)) else 0 for c in simple_cols)
                for _, row in users_df.iterrows()]

    chunk = 500
    with get_cursor() as cur:
        for i in range(0, len(rows), chunk):
            cur.executemany(sql, rows[i:i + chunk])
    print(f"[Database] Inserted {len(users_df)} users (TiDB, {chunk}/batch).")


def insert_products(products_df):
    if not isinstance(products_df, pd.DataFrame) or len(products_df) == 0:
        return
    available = [c for c in products_df.columns]
    placeholders = ", ".join(["%s"] * len(available))
    sql = f"REPLACE INTO products_real ({', '.join(available)}) VALUES ({placeholders})"
    rows = [tuple(row.get(c, 0) if not isinstance(row.get(c, 0), float) or pd.notna(row.get(c, 0)) else 0 for c in available)
            for _, row in products_df.iterrows()]
    with get_cursor() as cur:
        cur.executemany(sql, rows)
    print(f"[Database] Inserted {len(products_df)} products (TiDB).")


def insert_training_data(df, batch_size=1000):
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return
    if "label" not in df.columns:
        return
    sql = "REPLACE INTO train_data_real (user_id, product_id, label) VALUES (%s, %s, %s)"
    with get_cursor() as cur:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            rows = [(str(row["user_id"]), str(row["product_id"]), int(row["label"])) for _, row in batch.iterrows()]
            cur.executemany(sql, rows)
    print(f"[Database] Inserted {len(df)} training samples (real).")


def load_users():
    try:
        with get_cursor() as cur:
            cur.execute("SELECT * FROM users_real")
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
        df = pd.DataFrame(rows, columns=cols)
        if "coil_socio_data" in df.columns:
            for idx, val in df["coil_socio_data"].items():
                if isinstance(val, str):
                    try:
                        data = json.loads(val)
                        for k, v in data.items():
                            df.at[idx, k] = v
                    except Exception:
                        pass
        return df
    except Exception:
        try:
            with get_cursor() as cur:
                cur.execute("SELECT * FROM users")
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
            return pd.DataFrame(rows, columns=cols)
        except Exception:
            return None


def load_products():
    try:
        with get_cursor() as cur:
            cur.execute("SELECT * FROM products_real")
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        try:
            with get_cursor() as cur:
                cur.execute("SELECT * FROM products")
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
            return pd.DataFrame(rows, columns=cols)
        except Exception:
            return None


def load_train_data():
    try:
        with get_cursor() as cur:
            cur.execute("SELECT * FROM train_data_real")
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
        df = pd.DataFrame(rows, columns=cols)
        if len(df) > 0:
            users_df = load_users()
            products_df = load_products()
            if users_df is not None and products_df is not None:
                df = df.merge(users_df, on="user_id", how="left")
                df = df.merge(products_df, on="product_id", how="left")
        return df
    except Exception:
        try:
            with get_cursor() as cur:
                cur.execute("SELECT * FROM train_data")
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
            return pd.DataFrame(rows, columns=cols)
        except Exception:
            return None


def count_train_data():
    try:
        with get_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM train_data_real")
            return cur.fetchone()[0]
    except Exception:
        try:
            with get_cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM train_data")
                return cur.fetchone()[0]
        except Exception:
            return 0


def insert_training_record(record):
    try:
        with get_cursor() as cur:
            cur.execute(
                "INSERT INTO training_history (model_name, params, train_time, final_train_loss, final_val_loss, final_val_auc, epochs_trained) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (record.get("model_name"),
                 json.dumps(record.get("params", {}), ensure_ascii=False),
                 record.get("train_time", time.strftime("%Y-%m-%d %H:%M:%S")),
                 record.get("final_train_loss"),
                 record.get("final_val_loss"),
                 record.get("final_val_auc"),
                 record.get("epochs_trained"))
            )
        print("[Database] Training record inserted.")
    except Exception as e:
        print(f"[Database] Failed to insert training record: {e}")


def load_training_history():
    try:
        with get_cursor() as cur:
            cur.execute("SELECT * FROM training_history ORDER BY train_time DESC")
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
        results = []
        for row in rows:
            record = dict(zip(cols, row))
            if isinstance(record.get("params"), str):
                record["params"] = json.loads(record["params"])
            if hasattr(record.get("train_time"), "isoformat"):
                record["train_time"] = record["train_time"].strftime("%Y-%m-%d %H:%M:%S")
            results.append(record)
        return results
    except Exception:
        return []


def check_connection():
    try:
        _ensure_connection()
        with get_cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False
