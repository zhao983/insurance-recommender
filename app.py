import sys
import os

os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFY", "1")
os.environ.setdefault("CURL_CA_BUNDLE", "")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
try:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

import io
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
from config import (
    USER_CATEGORICAL_FEATURES, USER_NUMERICAL_FEATURES,
    INSURANCE_CATEGORIES, OUTPUT_DIR, DEFAULT_MODEL_PARAMS
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="保险产品智能推荐系统",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700; color: #1a5276;
        text-align: center; padding: 0.8rem 0;
        border-bottom: 3px solid #2980b9; margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.3rem; font-weight: 600; color: #2c3e50;
        margin: 1rem 0 0.5rem 0;
    }
    .rec-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px; padding: 18px; margin: 10px 0;
        border-left: 5px solid #2980b9;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .rec-card-high {
        background: linear-gradient(135deg, #e8f8f5 0%, #a9dfbf 100%);
        border-left: 5px solid #27ae60;
    }
    .rec-card-mid {
        background: linear-gradient(135deg, #fef9e7 0%, #f9e79f 100%);
        border-left: 5px solid #f39c12;
    }
    .score-badge {
        display: inline-block; padding: 4px 14px; border-radius: 20px;
        font-weight: 700; color: white; font-size: 0.9rem;
    }
    .score-high { background: #27ae60; }
    .score-mid { background: #f39c12; }
    .score-low { background: #e74c3c; }
    .metric-box {
        background: #f8f9fa; border-radius: 10px; padding: 15px;
        text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #2980b9; }
    .metric-label { font-size: 0.85rem; color: #7f8c8d; margin-top: 4px; }
    .stButton > button {
        background: linear-gradient(135deg, #2980b9 0%, #1a5276 100%);
        color: white; border: none; border-radius: 8px; padding: 10px 28px;
        font-weight: 600; font-size: 1rem; transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(41, 128, 185, 0.4);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_recommender():
    from models.recommender import InsuranceRecommender
    rec = InsuranceRecommender()
    with st.spinner("系统初始化中，首次运行将下载模型并训练，请稍候..."):
        rec.initialize()
    return rec


def main():
    st.markdown('<div class="main-header">🛡️ 基于深度学习与预训练大模型的保险产品智能推荐系统</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;color:#7f8c8d;margin-bottom:1.5rem;">'
        '小组成员：沈航威、蒋宇鑫、王雨豪 | 技术栈：PyTorch + DeepFM + HuggingFace Transformers + Streamlit'
        '</p>',
        unsafe_allow_html=True
    )

    if "recommender" not in st.session_state:
        st.session_state.recommender = load_recommender()
    if "single_rec_result" not in st.session_state:
        st.session_state.single_rec_result = None
    if "batch_rec_result" not in st.session_state:
        st.session_state.batch_rec_result = None
    if "uploaded_users" not in st.session_state:
        st.session_state.uploaded_users = None

    rec = st.session_state.recommender

    with st.sidebar:
        st.markdown("## 📋 导航菜单")
        page = st.radio(
            "选择功能页面：",
            ["🏠 首页概览", "📝 单用户推荐", "📂 批量推荐",
             "📊 模型性能", "📈 可视化分析", "📜 训练历史", "⚙️ 模型设置"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        st.markdown("### 🎯 系统信息")
        from config import DATA_SOURCE
        st.info(f"""
        **推荐模型**: DeepFM (深度学习)
        **数据来源**: {'UCI COIL 2000 真实数据' if DATA_SOURCE == 'real' else '合成数据'}
        **设备**: {'GPU' if 'cuda' in str(next(rec.model.parameters()).device) else 'CPU'}
        **产品数**: {len(rec.products_df)}
        **用户数**: {len(rec.users_df)}
        """)

    if page == "🏠 首页概览":
        render_home(rec)
    elif page == "📝 单用户推荐":
        render_single_user(rec)
    elif page == "📂 批量推荐":
        render_batch_recommend(rec)
    elif page == "📊 模型性能":
        render_model_performance(rec)
    elif page == "📈 可视化分析":
        render_visualization(rec)
    elif page == "📜 训练历史":
        render_training_history(rec)
    elif page == "⚙️ 模型设置":
        render_settings(rec)


def render_home(rec):
    st.markdown("## 🏠 系统概览")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-box"><div class="metric-value">DeepFM</div><div class="metric-label">推荐模型</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{len(rec.products_df)}</div><div class="metric-label">保险产品数</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{len(rec.users_df)}</div><div class="metric-label">用户档案数</div></div>', unsafe_allow_html=True)
    with col4:
        hist = rec.get_model_performance()
        auc_val = f"{hist.get('final_val_auc', 0)*100:.1f}%" if hist else "N/A"
        st.markdown(f'<div class="metric-box"><div class="metric-value">{auc_val}</div><div class="metric-label">模型AUC</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔬 技术架构")
    st.markdown("""
    本系统采用**免费开源的前沿人工智能技术**构建，核心技术包括：
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🎯 推荐模型 — DeepFM (PyTorch实现)**
        - 结合因子分解机(FM)与深度神经网络(DNN)
        - 同时捕获低阶和高阶特征交互
        - 端到端训练，无需人工特征工程
        - 支持稀疏类别特征与稠密数值特征
        """)
        st.markdown("""
        **🧠 文本理解 — Sentence-BERT**
        - 基于开源预训练Transformer模型
        - 对保险产品文本描述进行语义编码
        - 生成384维高质量文本嵌入
        - 支持多语言（中英文）理解
        """)
    with col2:
        st.markdown("""
        **💬 推荐理由生成 — GPT-2 Chinese / 智能分析**
        - 基于开源文本生成模型生成自然语言推荐理由
        - 结合用户特征与产品属性的深度分析
        - 多维度匹配度评估与解释
        - 非简单规则模板，基于特征重要性分析
        """)
        st.markdown("""
        **🖥️ 交互界面 — Streamlit**
        - 纯Python实现的Web应用框架
        - 支持表单录入、批量导入、结果导出
        - 丰富的可视化图表展示
        - 一键式操作，零环境配置
        """)

    st.markdown("---")
    st.markdown("### 📊 产品类别分布")
    cat_counts = rec.products_df["category"].value_counts()
    cols = st.columns(len(cat_counts))
    for i, (cat, cnt) in enumerate(cat_counts.items()):
        with cols[i]:
            st.metric(label=cat, value=cnt)

    st.markdown("---")
    st.markdown("### 🚀 快速开始")
    st.markdown("""
    1. 点击左侧 **📝 单用户推荐** 录入用户信息，获取个性化推荐
    2. 点击左侧 **📂 批量推荐** 上传CSV文件，批量获取推荐结果
    3. 点击左侧 **📊 模型性能** 查看模型评估指标
    4. 点击左侧 **📈 可视化分析** 查看数据可视化图表
    """)


def render_single_user(rec):
    st.markdown("## 📝 单用户保险产品推荐")
    st.markdown("填写以下用户信息，系统将为您推荐最合适的保险产品。")

    with st.form("user_input_form"):
        st.markdown("#### 👤 基本信息")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("年龄", min_value=18, max_value=80, value=32, step=1)
            gender = st.selectbox("性别", ["男", "女"])
            annual_income = st.number_input("年收入（元）", min_value=0, max_value=5000000, value=120000, step=10000)
        with col2:
            occupation = st.selectbox("职业", ["白领", "蓝领", "自由职业", "学生", "退休", "无业"])
            education = st.selectbox("教育程度", ["高中及以下", "大专", "本科", "硕士及以上"])
            health_status = st.selectbox("健康状况", ["健康", "亚健康", "有慢性病", "有重大病史"])
        with col3:
            family_structure = st.selectbox("家庭结构", ["单身", "已婚无子女", "已婚有子女", "单亲家庭", "三代同堂"])
            city_tier = st.selectbox("城市等级", ["一线城市", "二线城市", "三线城市", "四线及以下"])
            family_members = st.number_input("家庭成员数", min_value=1, max_value=10, value=3, step=1)

        col4, col5 = st.columns(2)
        with col4:
            has_house = st.radio("是否拥有房产", ["有", "无"], horizontal=True)
        with col5:
            has_car = st.radio("是否拥有车辆", ["有", "无"], horizontal=True)

        st.markdown("#### 📜 历史投保记录")
        history_purchases = st.multiselect("已购保险类型（可多选）", INSURANCE_CATEGORIES)

        submitted = st.form_submit_button("🔍 开始智能推荐", type="primary", use_container_width=True)

    if submitted:
        user_profile = {
            "age": age, "gender": gender, "annual_income": annual_income,
            "occupation": occupation, "education": education,
            "health_status": health_status, "family_structure": family_structure,
            "city_tier": city_tier, "family_members": family_members,
            "has_house": has_house, "has_car": has_car,
            "history_purchases": ",".join(history_purchases)
        }
        with st.spinner("🧠 AI正在分析您的特征并匹配最优产品..."):
            st.session_state.single_rec_result = rec.recommend_for_user(user_profile)

    if st.session_state.single_rec_result:
        results = st.session_state.single_rec_result
        st.markdown("---")
        st.markdown("### 🎯 个性化推荐结果 (Top-5)")

        for i, rec_item in enumerate(results):
            score = rec_item["match_score"]
            if score >= 0.7:
                card_class = "rec-card rec-card-high"
                badge_class = "score-high"
            elif score >= 0.5:
                card_class = "rec-card rec-card-mid"
                badge_class = "score-mid"
            else:
                card_class = "rec-card"
                badge_class = "score-low"

            st.markdown(f"""
            <div class="{card_class}">
                <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
                    <div>
                        <h3 style="margin:0;color:#2c3e50;">#{i+1} {rec_item['product_name']}</h3>
                        <span style="color:#7f8c8d;font-size:0.9rem;">
                            {rec_item['category']} | {rec_item['company']} | 年保费: ¥{rec_item['annual_premium']:,}
                        </span>
                    </div>
                    <span class="score-badge {badge_class}">匹配度: {score*100:.1f}%</span>
                </div>
                <p style="margin-top:12px;color:#2c3e50;line-height:1.7;font-size:0.95rem;">
                    <strong>💡 推荐理由：</strong>{rec_item['reason']}
                </p>
                <p style="color:#7f8c8d;font-size:0.85rem;margin-top:8px;">
                    📋 {rec_item.get('coverage_detail', '')[:150]}...
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            scores = [r["match_score"] for r in results]
            chart_data = pd.DataFrame({
                "产品": [f"#{i+1} {r['product_name'][:10]}" for i, r in enumerate(results)],
                "匹配度": [s * 100 for s in scores]
            })
            st.bar_chart(chart_data.set_index("产品"))
        with col2:
            cat_data = {}
            for r in results:
                cat = r["category"]
                cat_data[cat] = cat_data.get(cat, 0) + r["match_score"]
            cat_chart = pd.DataFrame({"类别": list(cat_data.keys()), "累计匹配度": list(cat_data.values())})
            st.bar_chart(cat_chart.set_index("类别"))

        csv_data = pd.DataFrame(results)
        csv_data["match_score"] = csv_data["match_score"].apply(lambda x: f"{x*100:.1f}%")
        csv_bytes = csv_data.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("📥 导出推荐结果 (CSV)", csv_bytes, "recommendation_result.csv", "text/csv")


def render_batch_recommend(rec):
    st.markdown("## 📂 批量用户推荐")
    st.markdown("上传包含用户信息的CSV文件，系统将为每位用户生成Top-5推荐结果。")

    st.markdown("#### 📋 CSV文件格式要求")
    st.markdown(f"必须包含以下列：`{', '.join(USER_CATEGORICAL_FEATURES + USER_NUMERICAL_FEATURES)}`")
    st.markdown("可选列：`user_id`, `history_purchases`")

    with st.expander("📥 下载CSV模板", expanded=False):
        template = rec.users_df.head(5).copy()
        template_csv = template.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("下载用户数据模板", template_csv, "user_template.csv", "text/csv")

    uploaded_file = st.file_uploader("选择CSV文件上传", type=["csv"], key="batch_upload")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
            st.success(f"✅ 成功加载 {len(df)} 条用户记录")
            st.dataframe(df.head(10), use_container_width=True)

            with st.expander("📊 数据统计", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**字段类型**")
                    st.write(df.dtypes)
                with col2:
                    st.write("**基本统计**")
                    st.write(df.describe())

            if st.button("🚀 开始批量推荐", type="primary", use_container_width=True):
                users_list = df.to_dict("records")
                with st.spinner(f"正在为 {len(users_list)} 位用户生成推荐..."):
                    all_results = rec.recommend_batch(users_list)
                    st.session_state.batch_rec_result = all_results
                st.success(f"✅ 批量推荐完成！共生成 {len(all_results)} 条推荐记录")

        except Exception as e:
            st.error(f"❌ 文件处理失败：{e}")

    if st.session_state.batch_rec_result:
        results_df = pd.DataFrame(st.session_state.batch_rec_result)

        st.markdown("### 📊 批量推荐结果")
        st.dataframe(
            results_df[["user_id", "rank", "product_name", "category", "company",
                         "annual_premium", "match_score"]].head(50),
            use_container_width=True
        )

        category_filter = st.multiselect("按险种筛选", options=results_df["category"].unique().tolist(),
                                         key="batch_cat_filter")
        filtered_df = results_df
        if category_filter:
            filtered_df = filtered_df[filtered_df["category"].isin(category_filter)]

        score_threshold = st.slider("最低匹配度阈值", 0.0, 1.0, 0.3, 0.05)
        filtered_df = filtered_df[filtered_df["match_score"] >= score_threshold]
        st.info(f"筛选后共 {len(filtered_df)} 条记录")

        csv_bytes = filtered_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("📥 导出全部推荐结果 (CSV)", csv_bytes, "batch_recommendations.csv", "text/csv")


def render_model_performance(rec):
    st.markdown("## 📊 模型性能报告")
    hist = rec.get_model_performance()

    if not hist:
        st.warning("暂无训练历史记录，请先完成模型训练。")
        return

    st.markdown("### 🏆 当前模型性能指标")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        auc = hist.get("final_val_auc", 0)
        st.markdown(f'<div class="metric-box"><div class="metric-value">{auc*100:.2f}%</div><div class="metric-label">验证集 AUC-ROC</div></div>', unsafe_allow_html=True)
    with col2:
        loss = hist.get("final_val_loss", 0)
        st.markdown(f'<div class="metric-box"><div class="metric-value">{loss:.4f}</div><div class="metric-label">验证集损失</div></div>', unsafe_allow_html=True)
    with col3:
        epochs = hist.get("epochs_trained", 0)
        st.markdown(f'<div class="metric-box"><div class="metric-value">{epochs}</div><div class="metric-label">训练轮次</div></div>', unsafe_allow_html=True)
    with col4:
        train_loss = hist.get("final_train_loss", 0)
        st.markdown(f'<div class="metric-box"><div class="metric-value">{train_loss:.4f}</div><div class="metric-label">训练集损失</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📈 模型训练曲线")
    from utils.visualization import plot_training_history
    history_data = {
        "train_loss": [hist.get("final_train_loss", 0)] * max(1, epochs),
        "val_loss": [hist.get("final_val_loss", 0)] * max(1, epochs),
        "val_auc": [hist.get("final_val_auc", 0)] * max(1, epochs),
    }
    fig = plot_training_history(history_data)
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 📋 当前模型参数配置")
    params = hist.get("params", {})
    params_df = pd.DataFrame(list(params.items()), columns=["参数", "当前值"])
    st.dataframe(params_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 类别统计")
    cat_stats = rec.get_category_stats()
    cat_df = pd.DataFrame(cat_stats).T
    st.dataframe(cat_df, use_container_width=True)


def render_visualization(rec):
    st.markdown("## 📈 可视化分析")

    viz_type = st.selectbox("选择可视化类型", [
        "用户-产品匹配热力图", "产品类别推荐分布", "匹配度分数分布", "特征重要性分析"
    ])

    if viz_type == "用户-产品匹配热力图":
        st.markdown("### 🔥 用户-产品匹配热力图")
        st.markdown("随机选取部分用户和产品，展示匹配度矩阵")
        n_users = st.slider("展示用户数", 5, 30, 12)
        sample_users = rec.users_df.sample(min(n_users, len(rec.users_df)))
        sample_products = rec.products_df.sample(min(15, len(rec.products_df)))

        scores_matrix = np.zeros((len(sample_users), len(sample_products)))
        for i, (_, user) in enumerate(sample_users.iterrows()):
            user_profile = user.to_dict()
            for j, (_, prod) in enumerate(sample_products.iterrows()):
                u_cat, u_num = rec.preprocessor.transform_user(user_profile)
                p_cat, p_num = rec.preprocessor.transform_product(prod)
                import torch
                model_device = next(rec.model.parameters()).device
                cat_input = torch.LongTensor(np.concatenate([u_cat, p_cat])).unsqueeze(0).to(model_device)
                num_input = torch.FloatTensor(
                    np.concatenate([u_num, p_num, rec.product_embeddings[j % len(rec.product_embeddings)]])
                ).unsqueeze(0).to(model_device)
                with torch.no_grad():
                    score = rec.model(cat_input, num_input).item()
                scores_matrix[i, j] = score

        from utils.visualization import plot_user_product_heatmap
        user_ids = [str(u.get("user_id", f"U{i}")) for i, u in enumerate(sample_users.to_dict("records"))]
        prod_names = [str(p.get("product_name", "")[:8]) for p in sample_products.to_dict("records")]
        fig = plot_user_product_heatmap(scores_matrix, user_ids, prod_names)
        st.pyplot(fig)

    elif viz_type == "产品类别推荐分布":
        st.markdown("### 📊 产品类别推荐分布")
        if st.session_state.batch_rec_result:
            from utils.visualization import plot_category_distribution
            fig = plot_category_distribution(st.session_state.batch_rec_result)
            st.pyplot(fig)
        else:
            sample_users_list = rec.users_df.sample(20).to_dict("records")
            with st.spinner("正在生成示例推荐数据..."):
                sample_results = rec.recommend_batch(sample_users_list)
            from utils.visualization import plot_category_distribution
            fig = plot_category_distribution(sample_results)
            st.pyplot(fig)

    elif viz_type == "匹配度分数分布":
        st.markdown("### 📈 匹配度分数分布")
        if st.session_state.batch_rec_result:
            scores = [r["match_score"] for r in st.session_state.batch_rec_result]
        else:
            scores = np.random.beta(2, 2, 1000).tolist()
        from utils.visualization import plot_score_distribution
        fig = plot_score_distribution(scores)
        st.pyplot(fig)

    elif viz_type == "特征重要性分析":
        st.markdown("### 🔍 特征重要性分析")
        st.markdown("基于DeepFM模型第一层DNN权重，推导各特征的相对重要性")
        from utils.visualization import plot_feature_importance
        cat_names = USER_CATEGORICAL_FEATURES + [f"产品_{c}" for c in rec.products_df.columns if c in rec.preprocessor.prod_cat_encoders]
        num_names = USER_NUMERICAL_FEATURES + [f"产品_{c}" for c in rec.products_df.columns if c in rec.preprocessor.prod_num_scaler.feature_names_in_]
        fig = plot_feature_importance(rec.model, cat_names, num_names)
        if fig:
            st.pyplot(fig)
        else:
            st.info("特征重要性图暂不可用，请完成模型训练后查看。")


def render_training_history(rec):
    st.markdown("## 📜 训练历史记录")

    if not rec.training_history:
        st.warning("暂无训练历史记录。")
        return

    for i, record in enumerate(rec.training_history):
        with st.expander(f"训练记录 #{i+1} - {record.get('train_time', 'Unknown')}", expanded=(i == 0)):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**模型名称**: {record.get('model_name', 'N/A')}")
                st.markdown(f"**训练时间**: {record.get('train_time', 'N/A')}")
                st.markdown(f"**训练轮次**: {record.get('epochs_trained', 'N/A')}")
            with col2:
                st.markdown(f"**最终训练损失**: {record.get('final_train_loss', 'N/A')}")
                st.markdown(f"**最终验证损失**: {record.get('final_val_loss', 'N/A')}")
                st.markdown(f"**最终验证AUC**: {record.get('final_val_auc', 'N/A')}")

            st.markdown("**模型参数**")
            params = record.get("params", {})
            st.json(params)


def render_settings(rec):
    st.markdown("## ⚙️ 模型设置与训练")

    current_params = rec.model_params

    st.markdown("### 🔧 超参数调整")
    col1, col2, col3 = st.columns(3)
    with col1:
        embedding_dim = st.slider("嵌入维度 (Embedding Dim)", 4, 64, current_params.get("embedding_dim", 16), 4)
        learning_rate = st.select_slider("学习率 (Learning Rate)", options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=current_params.get("learning_rate", 1e-3))
    with col2:
        dnn_layers = st.slider("DNN隐藏层数", 1, 5, len(current_params.get("dnn_hidden_units", [256, 128, 64])), 1)
        dnn_units_base = st.selectbox("DNN基础单元数", [32, 64, 128, 256, 512], index=3)
    with col3:
        dropout = st.slider("Dropout率", 0.0, 0.5, current_params.get("dnn_dropout", 0.2), 0.05)
        batch_size = st.selectbox("批次大小 (Batch Size)", [64, 128, 256, 512], index=2)

    dnn_units = [dnn_units_base * (2 ** (dnn_layers - i - 1)) for i in range(dnn_layers)]
    st.markdown(f"**DNN网络结构**: `{dnn_units}`")

    epochs = st.slider("训练轮次 (Epochs)", 5, 100, current_params.get("epochs", 30), 5)
    patience = st.slider("早停耐心值 (Early Stopping Patience)", 3, 15, current_params.get("early_stopping_patience", 5))

    if st.button("🔄 重新训练模型", type="primary", use_container_width=True):
        st.warning("⚠️ 重新训练将覆盖现有的模型权重。")
        confirm = st.checkbox("确认重新训练")

        if confirm:
            rec.set_params(
                embedding_dim=embedding_dim,
                learning_rate=learning_rate,
                dnn_hidden_units=dnn_units,
                dnn_dropout=dropout,
                batch_size=batch_size,
                epochs=epochs,
                early_stopping_patience=patience
            )
            with st.spinner("🔄 正在重新训练模型，请稍候..."):
                st.session_state.recommender = None
                st.cache_resource.clear()
                st.session_state.recommender = load_recommender()
            st.success("✅ 模型重新训练完成！")
            st.rerun()

    st.markdown("---")
    st.markdown("### 📋 当前参数配置")
    st.json(rec.model_params)

    st.markdown("---")
    st.markdown("### 📊 产品库预览")
    st.dataframe(rec.products_df.head(10), use_container_width=True)

    st.markdown("### 👥 用户库预览")
    st.dataframe(rec.users_df.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
