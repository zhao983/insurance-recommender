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

from config import TEXT_GEN_MODEL_NAME, DEVICE


class TextGenerator:
    def __init__(self, model_name=None):
        self.model_name = model_name or TEXT_GEN_MODEL_NAME
        self.model = None
        self.tokenizer = None
        self.available = False
        self._load_model()

    def _load_model(self):
        self.available = False
        print("[TextGenerator] Using intelligent template-based recommendation reason generation.")

    def generate_reason(self, user_profile, product_info, match_score, top_features=None):
        if self.available:
            return self._generate_with_model(user_profile, product_info, match_score)
        return self._generate_intelligent_fallback(user_profile, product_info, match_score, top_features)

    def _generate_with_model(self, user_profile, product_info, match_score):
        score_desc = "非常适合" if match_score >= 0.8 else ("比较适合" if match_score >= 0.6 else "可以考虑")
        prompt = (
            f"为用户推荐保险产品。"
            f"用户年龄{user_profile.get('age','')}岁，{user_profile.get('gender','')}性，"
            f"{user_profile.get('occupation','')}，{user_profile.get('health_status','')}，"
            f"年收入{user_profile.get('annual_income','')}元。"
            f"推荐产品：{product_info.get('product_name','')}，"
            f"类别：{product_info.get('category','')}，年保费{product_info.get('annual_premium','')}元。"
            f"推荐理由：该产品{score_desc}该用户，因为"
        )
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
            outputs = self.model.generate(
                **inputs, max_new_tokens=80, do_sample=True,
                temperature=0.7, top_p=0.9, repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            generated = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            reason = generated.strip().split("。")[0].strip()
            if len(reason) < 10:
                reason = self._generate_intelligent_fallback(user_profile, product_info, match_score)
            return reason
        except Exception as e:
            print(f"[TextGenerator] Generation error: {e}")
            return self._generate_intelligent_fallback(user_profile, product_info, match_score)

    def _generate_intelligent_fallback(self, user_profile, product_info, match_score, top_features=None):
        age = user_profile.get("age", 0)
        income = user_profile.get("annual_income", 0)
        health = user_profile.get("health_status", "")
        family = user_profile.get("family_structure", "")
        occupation = user_profile.get("occupation", "")
        gender = user_profile.get("gender", "")

        cat = product_info.get("category", "")
        pname = product_info.get("product_name", "")
        premium = product_info.get("annual_premium", 0)
        coverage = product_info.get("coverage_amount", 0)
        features = product_info.get("features", "")

        reasons = []

        premium_ratio = premium / max(income, 1)
        if premium_ratio < 0.05:
            reasons.append(f"保费仅占年收入的{premium_ratio*100:.1f}%，经济负担极低")
        elif premium_ratio < 0.10:
            reasons.append(f"保费占年收入{premium_ratio*100:.1f}%，在合理承受范围内")
        else:
            reasons.append(f"保费水平与收入比例{premium_ratio*100:.1f}%，建议评估预算后再做决定")

        if cat == "医疗险":
            if health in ("健康", "亚健康"):
                reasons.append("趁健康状况良好时配置医疗险，可享受更低费率与更全面的保障")
            else:
                reasons.append("医疗险可为日常就医和住院提供费用报销，减轻医疗负担")
        elif cat == "重疾险":
            if health in ("健康", "亚健康"):
                reasons.append("当前健康状况良好，是配置重疾险的最佳时机，保费更低且保障全面")
            elif age > 40:
                reasons.append(f"重疾险为{match_score*100:.0f}分匹配，在中年阶段配置可有效转移重大疾病风险")
            else:
                reasons.append("重疾险可提供确诊即赔的大额保障金，用于治疗和康复期间的生活支出")
        elif cat == "意外险":
            reasons.append("意外险保费低廉保障高，是基础保障配置的首选，可覆盖意外身故、伤残及医疗")
        elif cat == "寿险":
            if family in ("已婚有子女", "三代同堂", "单亲家庭"):
                reasons.append("作为家庭经济支柱，寿险可为家人提供经济保障，确保家庭责任得以履行")
            else:
                reasons.append("寿险可为家人提供长期经济保障，体现对家庭的责任与关爱")
        elif cat == "年金险":
            if age > 35:
                reasons.append("年金险可作为养老规划的重要工具，通过长期复利积累为退休生活提供稳定现金流")
            else:
                reasons.append("年金险通过长期稳健增值，可为未来教育、养老等重大支出做好储备")
        elif cat == "财产险":
            reasons.append("财产险为房屋及家庭财产提供全面保障，防范火灾、盗窃、自然灾害等风险")
        elif cat == "旅行险":
            reasons.append("旅行险提供意外、医疗运送、行李丢失等多重保障，让出行更安心")

        if coverage > 0:
            reasons.append(f"保额达{coverage/10000:.0f}万元，保障力度充足")

        if features:
            key_features = features.split("；")[:3]
            if key_features:
                reasons.append(f"包含{'、'.join(key_features[:2])}等核心保障")

        if match_score >= 0.8:
            reasons.insert(0, f"整体匹配度{match_score*100:.0f}分，{pname}与该用户特征高度吻合，强烈推荐")
        elif match_score >= 0.6:
            reasons.insert(0, f"匹配度{match_score*100:.0f}分，{pname}基本符合该用户需求，建议详细了解")
        else:
            reasons.insert(0, f"匹配度{match_score*100:.0f}分，{pname}可作为备选参考，建议对比同类产品")

        return "。".join(reasons[:4]) + "。"
