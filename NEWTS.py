import pandas as pd
import re
from openai import OpenAI
from langchain.prompts import PromptTemplate

# ===== 文件路径设置 =====
raw_path = r"C:\Users\OEM\Desktop\glucosedata\Test_data.csv"
df_raw = pd.read_csv(raw_path, parse_dates=["time"])
df_raw["time"] = df_raw["time"].dt.tz_localize(None)

# ===== 初始化 OpenAI GPT 客户端（替换为你的 key） =====
client = OpenAI(api_key="sk-")  # ← 替换为你的 OpenAI API Key

# ===== 分析函数：评估患者的糖尿病风险 =====
def analyze_patient_glucose(target_id: int, optional_input: str = ""):
    global df_raw

    if target_id not in df_raw["ID"].values:
        raise ValueError(f"Invalid patient ID: {target_id}")

    # 可选添加一个新血糖数据点
    if optional_input.strip():
        try:
            time_str, gl_str = optional_input.split(",")
            new_time = pd.to_datetime(time_str.strip())
            new_gl = float(gl_str.strip())
            new_row = pd.DataFrame([{"time": new_time, "gl": new_gl, "ID": target_id}])
            df_raw = pd.concat([df_raw, new_row], ignore_index=True)
            df_raw.to_csv(raw_path, index=False)
        except Exception as e:
            print(f"⚠️ Failed to add record: {e}")

    # 获取该患者数据
    user_data = df_raw[df_raw["ID"] == target_id].sort_values(by="time").drop_duplicates(subset=["time", "gl"])
    record_count = len(user_data)
    recent_data_str = user_data[["time", "gl"]].tail(min(5, record_count)).to_string(index=False)

    if record_count < 10:
        # 少于10条数据：仅根据最近血糖评估
        template = """
You are a professional medical assistant. Based on the recent glucose records provided below, briefly assess the patient's risk of diabetes. Note that normal glucose levels are generally considered to be between 70 and 200 mg/dL.

[Recent Glucose Records]
{recent_data}

Return only one short paragraph. Do not add any summary, follow-up, or extra explanation. Do not output anything before or after the paragraph.
"""
        prompt_text = PromptTemplate.from_template(template).format(recent_data=recent_data_str)
        metrics = None

    else:
        # ≥10条数据：提供统计指标
        glucose = user_data["gl"]
        mean_gl = glucose.mean()
        std_gl = glucose.std()
        cv = std_gl / mean_gl if mean_gl > 0 else 0
        max_gl = glucose.max()
        pct_high = (glucose > 200).mean() * 100
        pct_low = (glucose < 70).mean() * 100

        metrics = {
            "mean_glucose": f"{mean_gl:.2f}",
            "std_dev": f"{std_gl:.2f}",
            "cv": f"{cv:.3f}",
            "max_glucose": f"{max_gl:.2f}",
            "pct_high": f"{pct_high:.2f}",
            "pct_low": f"{pct_low:.2f}"
        }

        template = """
You are a professional medical assistant. Based on the recent glucose records and statistical profile provided below, briefly assess the patient's risk of diabetes. Note that normal glucose levels are generally considered to be between 70 and 200 mg/dL.

[Recent Glucose Records]
{recent_data}

[Glucose Profile]
- Mean glucose: {mean_glucose} mg/dL
- Std deviation: {std_dev}
- Coefficient of variation (CV): {cv}
- Max glucose: {max_glucose} mg/dL
- % time > 200 mg/dL: {pct_high}%
- % time < 70 mg/dL: {pct_low}%

Return only one short paragraph. Do not add any summary, follow-up, or extra explanation. Do not output anything before or after the paragraph.
"""
        prompt_text = PromptTemplate.from_template(template).format(
            recent_data=recent_data_str, **metrics
        )

    # 调用 GPT 模型生成回答
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful and professional medical assistant."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.2,
            top_p=0.8,
            max_tokens=300
        )
        ts_result = response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ GPT API error: {e}", recent_data_str, metrics

    return ts_result, recent_data_str, metrics

# ===== 程序入口 =====
if __name__ == "__main__":
    print("🚀 Glucose-based Diabetes Risk Assessment System is running.")

    try:
        # ✅ 在这里修改你的目标 ID，例如 1012
        result_text, latest_data, stat_summary = analyze_patient_glucose(target_id=10)

        print("\n✅ GPT Medical Evaluation:\n" + result_text)
        print("\n🧾 Recent Glucose Records:\n" + latest_data)
        if stat_summary:
            print("\n📊 Statistical Summary:\n" + str(stat_summary))

    except Exception as e:
        print(f"\n❌ Runtime error: {e}")
