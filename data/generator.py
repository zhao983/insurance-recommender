import numpy as np
import pandas as pd
from config import RANDOM_SEED, INSURANCE_CATEGORIES

np.random.seed(RANDOM_SEED)

GENDERS = ["男", "女"]
FAMILY_STRUCTURES = ["单身", "已婚无子女", "已婚有子女", "单亲家庭", "三代同堂"]
HEALTH_STATUSES = ["健康", "亚健康", "有慢性病", "有重大病史"]
OCCUPATIONS = ["白领", "蓝领", "自由职业", "学生", "退休", "无业"]
CITY_TIERS = ["一线城市", "二线城市", "三线城市", "四线及以下"]
EDUCATIONS = ["高中及以下", "大专", "本科", "硕士及以上"]
HOUSE_CAR = ["有", "无"]

COMPANIES = [
    "中国人寿", "中国平安", "太平洋保险", "新华保险", "泰康保险",
    "阳光保险", "大地保险", "中华联合", "天安保险", "华泰保险"
]

PRODUCT_NAMES = {
    "医疗险": [
        "安心百万医疗险", "普惠住院医疗险", "尊享e生医疗险",
        "健康保门诊医疗险", "悦享医疗补充险", "康护住院医疗险",
        "全民医保通"
    ],
    "重疾险": [
        "康惠保重疾险", "守护神终身重疾险", "安康保重疾险",
        "健康源重疾险", "一生保重大疾病险", "康宁终身重疾险",
        "如意保重疾险"
    ],
    "意外险": [
        "平安意外险", "无忧意外保", "百万意外险",
        "出行保综合意外险", "守护保意外险", "安行天下意外险",
        "随身保意外险"
    ],
    "寿险": [
        "福寿安康终身寿险", "鑫享人生寿险", "金生相伴寿险",
        "如意终身寿险", "传世经典寿险", "幸福人生定期寿险",
        "颐养天年终身寿险"
    ],
    "年金险": [
        "乐享晚年年金险", "金裕人生年金险", "盛世尊享年金险",
        "如意养老年金险", "鑫福年年年金险", "金彩一生年金险",
        "盛世年华教育年金险"
    ],
    "财产险": [
        "安居保家财险", "金锁家庭财产险", "安居乐业财险",
        "家和万事兴财险", "安居卫士家庭险"
    ],
    "旅行险": [
        "畅游全球旅行险", "境内旅游意外险", "海外旅行综合险",
        "自驾游保险", "环球旅行保障险"
    ]
}

PRODUCT_FEATURES = {
    "医疗险": [
        "住院医疗费用报销", "门诊医疗", "特殊门诊",
        "住院前后门急诊", "质子重离子治疗", "外购药报销",
        "就医绿色通道", "住院垫付服务"
    ],
    "重疾险": [
        "100种重疾保障", "50种轻症保障", "中症额外赔付",
        "恶性肿瘤二次赔付", "心脑血管二次赔付", "身故返还保费",
        "重疾豁免保费", "特定疾病额外赔"
    ],
    "意外险": [
        "意外身故伤残", "意外医疗", "猝死保障",
        "交通意外额外赔", "意外住院津贴", "意外骨折保障",
        "救护车费用", "航空意外高保额"
    ],
    "寿险": [
        "身故全残保障", "终身保障", "保额递增",
        "保单贷款", "减额交清", "年金转换权",
        "分红收益", "万能账户"
    ],
    "年金险": [
        "保证领取20年", "万能账户增值", "教育金储备",
        "养老年金领取", "祝寿金", "身故给付",
        "保单贷款", "灵活领取"
    ],
    "财产险": [
        "房屋主体保障", "室内财产", "盗抢保障",
        "水管爆裂", "第三者责任", "家电维修",
        "临时住所费用", "自然灾害保障"
    ],
    "旅行险": [
        "意外身故伤残", "医疗运送", "行李丢失",
        "航班延误", "旅行取消", "紧急救援",
        "个人责任", "证件丢失"
    ]
}


def generate_users(n_users=5000):
    records = []
    for i in range(n_users):
        age = np.random.randint(18, 71)
        gender = np.random.choice(GENDERS)
        if age < 22:
            family_structure = np.random.choice(["单身", "三代同堂"], p=[0.9, 0.1])
        elif age < 30:
            family_structure = np.random.choice(FAMILY_STRUCTURES, p=[0.45, 0.25, 0.15, 0.05, 0.10])
        elif age < 50:
            family_structure = np.random.choice(FAMILY_STRUCTURES, p=[0.10, 0.15, 0.50, 0.10, 0.15])
        else:
            family_structure = np.random.choice(FAMILY_STRUCTURES, p=[0.10, 0.25, 0.30, 0.05, 0.30])

        if age < 22:
            occupation = "学生"
            income_base = np.random.randint(0, 30001)
        elif age > 60:
            occupation = np.random.choice(["退休", "自由职业", "无业"], p=[0.7, 0.2, 0.1])
            income_base = np.random.randint(20000, 150001)
        else:
            occupation = np.random.choice(OCCUPATIONS, p=[0.35, 0.25, 0.15, 0.02, 0.03, 0.20])
            income_map = {"白领": (80000, 500001), "蓝领": (40000, 150001),
                          "自由职业": (30000, 300001), "学生": (0, 30001),
                          "退休": (30000, 150001), "无业": (0, 50001)}
            lo, hi = income_map[occupation]
            income_base = np.random.randint(lo, hi)

        city_tier = np.random.choice(CITY_TIERS, p=[0.20, 0.30, 0.30, 0.20])
        city_mult = {"一线城市": 1.5, "二线城市": 1.0, "三线城市": 0.7, "四线及以下": 0.5}
        annual_income = int(income_base * city_mult[city_tier])

        education_probs = {
            "高中及以下": {18: 0.6, 30: 0.3, 50: 0.5, 71: 0.7},
            "大专": {18: 0.3, 30: 0.3, 50: 0.25, 71: 0.2},
            "本科": {18: 0.09, 30: 0.3, 50: 0.2, 71: 0.08},
            "硕士及以上": {18: 0.01, 30: 0.1, 50: 0.05, 71: 0.02}
        }
        age_bracket = 18 if age < 30 else (30 if age < 50 else (50 if age < 60 else 71))
        edu_probs = [education_probs[e][age_bracket] for e in EDUCATIONS]
        edu_probs = np.array(edu_probs) / sum(edu_probs)
        education = np.random.choice(EDUCATIONS, p=edu_probs)

        health_probs = [0.55, 0.30, 0.10, 0.05] if age < 40 else (
            [0.30, 0.35, 0.25, 0.10] if age < 60 else [0.15, 0.30, 0.35, 0.20]
        )
        health_status = np.random.choice(HEALTH_STATUSES, p=health_probs)

        has_house = np.random.choice(HOUSE_CAR, p=[0.55, 0.45] if age > 30 else [0.25, 0.75])
        has_car = np.random.choice(HOUSE_CAR, p=[0.40, 0.60] if annual_income > 100000 else [0.10, 0.90])

        family_map = {"单身": 1, "单亲家庭": 2, "已婚无子女": 2, "已婚有子女": np.random.choice([3, 4, 5]),
                       "三代同堂": np.random.choice([4, 5, 6])}
        family_members = family_map[family_structure]

        history_purchases = _generate_history(age, health_status, family_structure, annual_income)

        records.append({
            "user_id": f"U{i+1:06d}",
            "age": age, "gender": gender,
            "family_structure": family_structure,
            "annual_income": annual_income,
            "occupation": occupation, "city_tier": city_tier,
            "education": education, "health_status": health_status,
            "has_house": has_house, "has_car": has_car,
            "family_members": family_members,
            "history_purchases": history_purchases
        })
    return pd.DataFrame(records)


def _generate_history(age, health, family, income):
    num = np.random.randint(0, 3)
    if num == 0:
        return ""
    candidates = []
    if age > 25:
        candidates.extend(["医疗险", "意外险"])
    if age > 30 and income > 80000:
        candidates.append("重疾险")
    if age > 35 and family in ("已婚有子女", "三代同堂"):
        candidates.append("寿险")
    if age > 40 and income > 150000:
        candidates.append("年金险")
    if income > 200000:
        candidates.append("财产险")
    chosen = list(np.random.choice(candidates, size=min(num, len(candidates)), replace=False))
    return ",".join(chosen)


def generate_products(n_products=80):
    records = []
    pid = 1
    for cat in INSURANCE_CATEGORIES:
        cat_count = np.random.randint(8, 13) if cat != "旅行险" else np.random.randint(5, 8)
        for _ in range(cat_count):
            pname = np.random.choice(PRODUCT_NAMES[cat])
            coverage_amounts = {"医疗险": [50, 100, 200, 300, 400, 500, 600],
                                "重疾险": [10, 20, 30, 50, 80, 100],
                                "意外险": [10, 30, 50, 100, 200, 500],
                                "寿险": [20, 50, 100, 200, 300, 500],
                                "年金险": [10, 20, 30, 50, 100],
                                "财产险": [50, 100, 200, 300, 500],
                                "旅行险": [10, 30, 50, 100, 200]}
            coverage_amount = np.random.choice(coverage_amounts[cat]) * 10000

            premium_ranges = {"医疗险": (200, 3000), "重疾险": (1000, 15000),
                              "意外险": (100, 2000), "寿险": (500, 20000),
                              "年金险": (5000, 50000), "财产险": (300, 3000),
                              "旅行险": (50, 1000)}
            pmin, pmax = premium_ranges[cat]
            premium = np.random.randint(pmin, pmax + 1)

            coverage_years = np.random.choice([1, 5, 10, 20, 30, 99]) if cat != "旅行险" else np.random.choice([1, 7, 15, 30, 90])
            waiting_days = np.random.choice([0, 30, 60, 90, 180]) if cat in ("医疗险", "重疾险") else 0
            deductible = np.random.choice([0, 5000, 10000, 20000, 50000]) if cat == "医疗险" else 0

            target_min = np.random.randint(0, 25) if cat in ("意外险", "旅行险") else np.random.randint(18, 40)
            target_max = np.random.randint(50, 76) if cat != "旅行险" else np.random.randint(60, 81)
            target_gender = np.random.choice(["不限", "男", "女"], p=[0.8, 0.1, 0.1])
            target_income_min = np.random.choice([0, 30000, 50000, 80000, 100000], p=[0.2, 0.2, 0.2, 0.2, 0.2])

            features_list = PRODUCT_FEATURES[cat]
            num_features = np.random.randint(3, min(6, len(features_list) + 1))
            selected = list(np.random.choice(features_list, size=num_features, replace=False))
            features_text = "；".join(selected)

            coverage_type = np.random.choice(["基础版", "标准版", "升级版", "尊享版"], p=[0.2, 0.4, 0.25, 0.15])
            payment_method = np.random.choice(["趸交", "年交", "月交"], p=[0.1, 0.7, 0.2])
            company = np.random.choice(COMPANIES)

            coverage_detail = f"{pname}提供{coverage_amount//10000}万元{'保额' if cat != '医疗险' else '医疗保障'}，" \
                              f"保障期限{coverage_years}{'年' if coverage_years < 99 else '（终身）'}，" \
                              f"年缴保费{premium}元。主要保障：{features_text}"

            rating = round(np.random.uniform(3.5, 5.0), 1)

            records.append({
                "product_id": f"P{pid:04d}",
                "product_name": pname, "category": cat,
                "company": company, "coverage_amount": coverage_amount,
                "annual_premium": premium, "coverage_years": coverage_years,
                "waiting_period_days": waiting_days, "deductible": deductible,
                "target_age_min": target_min, "target_age_max": target_max,
                "target_gender": target_gender, "target_income_min": target_income_min,
                "features": features_text, "coverage_type": coverage_type,
                "payment_method": payment_method, "rating": rating,
                "coverage_detail": coverage_detail
            })
            pid += 1
    return pd.DataFrame(records)


def generate_training_data(users_df, products_df, n_samples=30000):
    records = []
    users = users_df.to_dict("records")
    products = products_df.to_dict("records")
    n_users = len(users)
    n_prods = len(products)

    for _ in range(n_samples):
        u = users[np.random.randint(0, n_users)]
        p = products[np.random.randint(0, n_prods)]
        label = _compute_match_score(u, p)
        records.append({**{f"u_{k}": v for k, v in u.items()},
                        **{f"p_{k}": v for k, v in p.items()},
                        "label": label})
    return pd.DataFrame(records)


def _compute_match_score(user, product):
    score = 0.0

    age = user["age"]
    if product["target_age_min"] <= age <= product["target_age_max"]:
        score += 0.15
    elif abs(age - product["target_age_min"]) <= 5 or abs(age - product["target_age_max"]) <= 5:
        score += 0.05

    if product["target_gender"] == "不限" or product["target_gender"] == user["gender"]:
        score += 0.10

    if user["annual_income"] >= product["target_income_min"]:
        score += 0.10

    premium_ratio = product["annual_premium"] / max(user["annual_income"], 1)
    if premium_ratio < 0.05:
        score += 0.15
    elif premium_ratio < 0.10:
        score += 0.10
    elif premium_ratio < 0.20:
        score += 0.05

    cat = product["category"]
    health = user["health_status"]
    if cat == "重疾险" and health in ("有慢性病", "有重大病史"):
        score -= 0.15
    elif cat == "医疗险" and health in ("健康", "亚健康"):
        score += 0.05
    elif cat == "意外险":
        score += 0.05
    elif cat == "寿险" and user["family_structure"] in ("已婚有子女", "三代同堂"):
        score += 0.10
    elif cat == "年金险" and age > 35 and user["annual_income"] > 100000:
        score += 0.10
    elif cat == "旅行险" and user["annual_income"] > 80000 and age < 55:
        score += 0.08
    elif cat == "财产险" and user["has_house"] == "有":
        score += 0.10

    history = user.get("history_purchases", "")
    if history and cat in history:
        score += 0.05

    product_rating = product.get("rating", 3.5) / 5.0
    score += product_rating * 0.10

    noise = np.random.normal(0, 0.08)
    score += noise

    if user["has_car"] == "有" and cat == "财产险":
        score += 0.03

    if user["family_members"] >= 4 and cat == "寿险":
        score += 0.05

    score = np.clip(score, 0.0, 1.0)
    if score > 0.5:
        score = 1.0
    elif score > 0.3:
        score = 1.0 if np.random.random() < (score - 0.3) / 0.2 else 0.0
    else:
        score = 0.0

    return float(score)


def main():
    print("=" * 60)
    print("Real Insurance Data Pipeline")
    print("=" * 60)

    from data.real_loader import (
        load_uci_coil_training_data, build_user_profiles_from_coil,
        build_products_from_coil, build_training_data_from_coil,
        get_coil_descriptive_stats
    )

    print("\n[1/4] Downloading UCI COIL 2000 dataset...")
    print("      Source: UCI Machine Learning Repository")
    print("      URL: https://archive.ics.uci.edu/dataset/125/")
    coil_df = load_uci_coil_training_data()

    stats = get_coil_descriptive_stats(coil_df)
    print(f"\n[2/4] Dataset statistics:")
    print(f"      Customers: {stats['total_customers']}")
    print(f"      Insurance products: {stats['product_columns']}")
    print(f"      Socio-demographic features: {stats['socio_demographic_columns']}")
    print(f"      Avg products per customer: {stats['products_per_customer']:.1f}")
    print(f"      Categories: {', '.join(stats['categories'])}")

    print("\n[3/4] Building user profiles and product catalog...")
    users_df = build_user_profiles_from_coil(coil_df)
    products_df = build_products_from_coil()
    train_data = build_training_data_from_coil(coil_df, users_df, products_df)

    users_df.to_csv("d:/VsPython/insurance_recommender/data/users.csv", index=False, encoding="utf-8-sig")
    products_df.to_csv("d:/VsPython/insurance_recommender/data/products.csv", index=False, encoding="utf-8-sig")
    train_data.to_csv("d:/VsPython/insurance_recommender/data/train_data.csv", index=False, encoding="utf-8-sig")

    print(f"\n[4/4] Saving to CSV and MySQL...")
    print(f"      users:      {len(users_df)} records -> data/users.csv")
    print(f"      products:   {len(products_df)} records -> data/products.csv")
    print(f"      train_data: {len(train_data)} records -> data/train_data.csv "
          f"(pos={train_data['label'].sum()}, neg={len(train_data)-train_data['label'].sum()})")

    _write_to_mysql(users_df, products_df, train_data)

    print("\nData pipeline complete!")
    print("=" * 60)


def _write_to_mysql(users, products, train_data):
    try:
        from data.database import init_database, insert_users, insert_products
        init_database()
        insert_users(users)
        insert_products(products)
        print("      MySQL sync complete! (users + products only, training data stays local)")
    except Exception as e:
        print(f"      MySQL sync skipped: {e}")
        print("      CSV files are still usable as fallback.")


if __name__ == "__main__":
    main()
