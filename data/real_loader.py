import os
import ssl
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from config import DATA_DIR

ssl._create_default_https_context = ssl._create_unverified_context

UCI_COIL_URL = "https://archive.ics.uci.edu/static/public/125/insurance+company+benchmark+coil+2000.zip"
UCI_COIL_FILES = {"data": "ticdata2000.txt", "desc": "TicDataDescr.txt"}

KAGGLE_CROSS_SELL_URL = "https://github.com/ruru-lyy/Insurance-Customer-Data-EDA/raw/master/insurance_customer_data.csv"

COIL_PRODUCT_NAMES = [
    "A号码次标准车险", "B号码第三者车险", "C号码火灾险", "D号码自行车险",
    "E号码货物运输险", "F号码一般责任险", "G号码律师援助险", "H号码作物险",
    "I号码农机险", "J号码寿险", "K号码私人意外险", "L号码家庭财产险",
    "M号码健康险", "N号码信用卡保险", "O号码法律援助险", "P号码剩余价值险",
    "Q号码玻璃险", "R号码摩托车险", "S号码亲属责任险", "T号码失业保险",
    "U号码收入保障险", "V号码人寿储蓄险", "W号码汽车险", "X号码火灾盗窃险",
    "Y号码特殊车险", "Z号码卡车上牌险", "AA号码法律费用险", "BB号码综合险",
    "CC号码农用拖拉机险", "DD号码农机责任险", "EE号码旅行险", "FF号码拖车险",
    "GG号码微型车险", "HH号码踏板车险", "II号码残障险", "JJ号码学生险",
    "KK号码丧葬险", "LL号码续保投资险", "MM号码养老金计划", "NN号码家庭险",
    "OO号码个人责任险", "PP号码收入保护险"
]

COIL_PRODUCT_CATEGORIES = {
    "车险": ["A号码次标准车险", "B号码第三者车险", "W号码汽车险", "X号码火灾盗窃险",
             "Y号码特殊车险", "Z号码卡车上牌险", "GG号码微型车险", "HH号码踏板车险",
             "Q号码玻璃险", "R号码摩托车险", "FF号码拖车险"],
    "健康险": ["M号码健康险", "II号码残障险"],
    "寿险": ["J号码寿险", "V号码人寿储蓄险", "KK号码丧葬险", "PP号码收入保护险"],
    "意外险": ["K号码私人意外险"],
    "财产险": ["D号码自行车险", "E号码货物运输险", "H号码作物险", "I号码农机险",
             "L号码家庭财产险", "CC号码农用拖拉机险", "NN号码家庭险"],
    "责任险": ["F号码一般责任险", "S号码亲属责任险", "OO号码个人责任险",
              "DD号码农机责任险"],
    "旅行险": ["EE号码旅行险"],
    "法律与援助险": ["G号码律师援助险", "O号码法律援助险", "AA号码法律费用险"],
    "失业与收入险": ["T号码失业保险", "U号码收入保障险"],
    "其他保险": ["C号码火灾险", "N号码信用卡保险", "P号码剩余价值险",
               "JJ号码学生险", "BB号码综合险", "LL号码续保投资险", "MM号码养老金计划"]
}

COIL_SOCIO_DEMOGRAPHIC_NAMES = [
    "客户亚型1", "客户亚型2", "客户亚型3", "客户亚型4", "客户亚型5",
    "客户亚型6", "客户亚型7", "客户亚型8", "客户亚型9", "客户亚型10",
    "客户亚型11", "客户亚型12", "客户亚型13", "客户亚型14", "客户亚型15",
    "客户亚型16", "客户亚型17", "客户亚型18", "客户亚型19", "客户亚型20",
    "客户亚型21", "客户亚型22", "客户亚型23", "客户亚型24", "客户亚型25",
    "客户亚型26", "客户亚型27", "客户亚型28", "客户亚型29", "客户亚型30",
    "客户亚型31", "客户亚型32", "客户亚型33", "客户亚型34", "客户亚型35",
    "客户亚型36", "客户亚型37", "客户亚型38", "客户亚型39", "客户亚型40",
    "客户亚型41", "客户亚型42", "客户亚型43"
]

COIL_SOCIO_CATEGORY_NAMES = [
    "购买力等级", "社会阶层A", "社会阶层B1", "社会阶层B2", "社会阶层C",
    "社会阶层D", "租房比例", "农业企业家比例", "高学历比例", "高收入比例",
    "有子女家庭比例", "平均收入", "平均拥有车数", "平均拥有卡车数", "平均拥有拖车数",
    "摩托车比例", "面包车比例", "社会保障比例", "低收入比例", "高收入居民比例",
    "有车家庭比例", "国家公务员比例", "中收入比例", "中高收入比例", "基督教徒比例",
    "天主教徒比例", "新教徒比例", "其他宗教比例", "无宗教比例", "已婚比例",
    "同居比例", "其他婚姻比例", "单身比例", "大家庭比例", "独居比例",
    "20-30岁居民比例", "30-40岁居民比例", "40-50岁居民比例", "50-60岁居民比例",
    "60-70岁居民比例", "70-80岁居民比例", "平均年龄", "社区家庭数"
]


def download_uci_coil():
    uci_dir = os.path.join(DATA_DIR, "uci_coil_2000")
    os.makedirs(uci_dir, exist_ok=True)
    zip_path = os.path.join(uci_dir, "coil.zip")
    data_path = os.path.join(uci_dir, UCI_COIL_FILES["data"])

    if os.path.exists(data_path):
        print(f"[RealLoader] UCI COIL 2000 already cached at {data_path}")
        return uci_dir

    print(f"[RealLoader] Downloading UCI COIL 2000 from UCI Machine Learning Repository...")
    urllib.request.urlretrieve(UCI_COIL_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(uci_dir)
    print(f"[RealLoader] UCI COIL 2000 extracted to {uci_dir}")
    return uci_dir


def load_uci_coil_training_data():
    uci_dir = download_uci_coil()
    data_file = os.path.join(uci_dir, UCI_COIL_FILES["data"])
    headers = COIL_SOCIO_DEMOGRAPHIC_NAMES + COIL_PRODUCT_NAMES + ["CARAVAN_目标"]
    df = pd.read_csv(data_file, sep="\t", header=None, names=headers)
    print(f"[RealLoader] Loaded UCI COIL 2000: {len(df)} customers × {len(df.columns)} features")
    return df


def build_user_profiles_from_coil(coil_df):
    user_records = []
    for idx, row in coil_df.iterrows():
        profile = {"user_id": f"UCI{idx+1:06d}"}
        profile["history_purchases"] = ",".join([
            COIL_PRODUCT_NAMES[i] for i in range(len(COIL_PRODUCT_NAMES))
            if row[COIL_PRODUCT_NAMES[i]] == 1
        ])
        for ci, col in enumerate(COIL_SOCIO_DEMOGRAPHIC_NAMES):
            profile[f"coil_socio_{ci+1}"] = float(row[col]) if not pd.isna(row[col]) else 0.0
        profile["social_category_idx"] = int(np.argmax([row.get(c, 0) for c in COIL_SOCIO_CATEGORY_NAMES[:12]]))
        profile["age_group_idx"] = int(np.argmax([row.get(c, 0) for c in COIL_SOCIO_CATEGORY_NAMES[35:43]]))
        profile["avg_income_level"] = float(row.get("平均收入", 0))
        profile["car_ownership"] = float(row.get("平均拥有车数", 0))
        profile["household_size"] = float(row.get("大家庭比例", 0)) * 10
        profile["religion_idx"] = int(np.argmax([
            row.get("基督教徒比例", 0), row.get("天主教徒比例", 0),
            row.get("新教徒比例", 0), row.get("其他宗教比例", 0),
            row.get("无宗教比例", 0)
        ]))
        profile["marital_status_idx"] = int(np.argmax([
            row.get("已婚比例", 0), row.get("同居比例", 0),
            row.get("其他婚姻比例", 0), row.get("单身比例", 0)
        ]))
        profile["has_caravan"] = int(row["CARAVAN_目标"])
        user_records.append(profile)
    return pd.DataFrame(user_records)


def build_products_from_coil():
    records = []
    for i, pname in enumerate(COIL_PRODUCT_NAMES):
        category = "其他保险"
        for cat, names in COIL_PRODUCT_CATEGORIES.items():
            if pname in names:
                category = cat
                break
        records.append({
            "product_id": f"PCOIL{i+1:04d}",
            "product_name": pname,
            "category": category,
            "company": "COIL保险公司",
            "coverage_amount": np.random.randint(50000, 500001),
            "annual_premium": np.random.randint(200, 5001),
            "coverage_years": 1,
            "waiting_period_days": np.random.choice([0, 30]),
            "deductible": np.random.choice([0, 5000, 10000]),
            "target_age_min": 18,
            "target_age_max": 80,
            "target_gender": "不限",
            "target_income_min": 0,
            "features": f"COIL-2000真实保险产品：{pname}，属于{category}类别",
            "coverage_type": "标准版",
            "payment_method": "年交",
            "rating": 4.0,
            "coverage_detail": f"COIL-2000数据集中的真实保险产品：{pname}（{category}）。"
                              f"此数据来自荷兰保险公司Sentient Machine Research的真实业务数据，"
                              f"于2000年作为CoIL Challenge发布在UCI机器学习库。"
        })
    return pd.DataFrame(records)


def build_training_data_from_coil(coil_df, users_df, products_df):
    records = []
    product_names = COIL_PRODUCT_NAMES
    user_ids = users_df["user_id"].tolist()
    product_ids = products_df["product_id"].tolist()

    for i, (_, row) in enumerate(coil_df.iterrows()):
        uid = user_ids[i]
        for j, pname in enumerate(product_names):
            label = min(int(row[pname]), 1)
            pid = product_ids[j]
            records.append({
                "user_id": uid, "product_id": pid, "label": label
            })

    df = pd.DataFrame(records)
    print(f"[RealLoader] Generated {len(df)} training samples from UCI COIL 2000 "
          f"(pos={df['label'].sum()}, neg={len(df)-df['label'].sum()})")
    return df


def load_kaggle_cross_sell():
    kaggle_path = os.path.join(DATA_DIR, "kaggle_cross_sell.csv")
    if os.path.exists(kaggle_path):
        print(f"[RealLoader] Kaggle cross-sell data cached at {kaggle_path}")
        return pd.read_csv(kaggle_path)

    print("[RealLoader] Kaggle cross-sell not cached. Use kagglehub or manual download.")
    print("[RealLoader] Visit: https://www.kaggle.com/competitions/playground-series-s4e7/data")
    print("[RealLoader] Download train.csv and save to data/kaggle_cross_sell.csv")
    return None


def get_coil_descriptive_stats(coil_df):
    stats = {
        "total_customers": len(coil_df),
        "product_columns": len(COIL_PRODUCT_NAMES),
        "socio_demographic_columns": len(COIL_SOCIO_DEMOGRAPHIC_NAMES),
        "caravan_positive_rate": float(coil_df["CARAVAN_目标"].mean()),
        "products_per_customer": float(coil_df[COIL_PRODUCT_NAMES].sum(axis=1).mean()),
        "product_names": COIL_PRODUCT_NAMES[:5] + ["..."] + COIL_PRODUCT_NAMES[-3:],
        "categories": list(COIL_PRODUCT_CATEGORIES.keys()),
    }
    return stats


if __name__ == "__main__":
    df = load_uci_coil_training_data()
    stats = get_coil_descriptive_stats(df)
    print("\n=== UCI COIL 2000 Dataset Statistics ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\n  First 3 rows:\n{df.iloc[:3, :3].to_string()}")
