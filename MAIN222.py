from datetime import datetime
import pandas as pd
import sys
from openai import OpenAI

# ========== Configuration ==========
client = OpenAI(api_key="sk-")  # 请替换为你的API Key

# 读取血糖数据
raw_path = r"C:\Users\OEM\Desktop\glucosedata\Test_data.csv"
df_raw = pd.read_csv(raw_path, parse_dates=["time"])
df_raw["time"] = df_raw["time"].dt.tz_localize(None)


# ========== 时间序列提示模板 ==========
def build_ts_prompt(recent_data_str, metrics=None):
    if metrics is None:
        template = f"""
You are a professional medical assistant. Based on the recent glucose records provided below, briefly assess the patient's risk of diabetes. Note that normal glucose levels are generally considered to be between 70 and 200 mg/dL.

[Recent Glucose Records]
{recent_data_str}

Return only one short paragraph. Do not add any summary, follow-up, or extra explanation.
"""
    else:
        template = f"""
You are a professional medical assistant. Based on the recent glucose records and statistical profile provided below, briefly assess the patient's risk of diabetes. Note that normal glucose levels are generally considered to be between 70 and 200 mg/dL.

[Recent Glucose Records]
{recent_data_str}

[Glucose Profile]
- Mean glucose: {metrics['mean_glucose']} mg/dL
- Std deviation: {metrics['std_dev']}
- Coefficient of variation (CV): {metrics['cv']}
- Max glucose: {metrics['max_glucose']} mg/dL
- % time > 200 mg/dL: {metrics['pct_high']}%
- % time < 70 mg/dL: {metrics['pct_low']}%

Return only one short paragraph. Do not add any summary, follow-up, or extra explanation.
"""
    return template.strip()


# ========== 调用 GPT ==========
def ask_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


# ========== 分析血糖数据 ==========
def analyze_patient_glucose(target_id: int, optional_input: str = ""):
    global df_raw

    if target_id not in df_raw["ID"].values:
        raise ValueError(f"Invalid patient ID: {target_id}")

    if optional_input.strip():
        try:
            time_str, gl_str = optional_input.split(",")
            new_time = pd.to_datetime(time_str.strip())
            new_gl = float(gl_str.strip())
            new_row = pd.DataFrame([{"time": new_time, "gl": new_gl, "ID": target_id}])
            df_raw = pd.concat([df_raw, new_row], ignore_index=True)
            df_raw.to_csv(raw_path, index=False)
        except Exception as e:
            raise ValueError(f"Failed to parse new input: {e}")

    user_data = df_raw[df_raw["ID"] == target_id].sort_values(by="time")
    record_count = len(user_data)
    recent_data_str = user_data[["time", "gl"]].tail(min(5, record_count)).to_string(index=False)

    if record_count < 10:
        prompt = build_ts_prompt(recent_data_str)
        metrics = None
    else:
        glucose = user_data["gl"]
        metrics = {
            "mean_glucose": f"{glucose.mean():.2f}",
            "std_dev": f"{glucose.std():.2f}",
            "cv": f"{glucose.std() / glucose.mean():.3f}" if glucose.mean() > 0 else "0",
            "max_glucose": f"{glucose.max():.2f}",
            "pct_high": f"{(glucose > 200).mean() * 100:.2f}",
            "pct_low": f"{(glucose < 70).mean() * 100:.2f}"
        }
        prompt = build_ts_prompt(recent_data_str, metrics)

    ts_result = ask_gpt(prompt)
    return ts_result, recent_data_str, metrics


# ========== 主程序入口 ==========
if __name__ == "__main__":
    print("🚀 Glucose-based Diabetes Risk Assessment System is running.")

    # === 用户输入 ===
    question = input("❓ Enter your medical question:\n> ").strip()
    target_id = int(input("\n🔎 Enter patient ID:\n> "))
    optional_input = input("📝 Optional: enter a new record as time,gl (e.g. 2024-04-01 08:00,120), or press Enter to skip:\n> ")

    # === 构造模拟 RAG 回答 ===
    rag_prompt = f"""
You are a professional medical assistant. You can use the provided context to help answer the user's question, but you are not limited to it. Feel free to incorporate general medical knowledge where appropriate.

[User Question]
{question}

[Context]
(The answer should assume relevant medical background knowledge is included here.)
""".strip()

    rag_output = ask_gpt(rag_prompt)
    print("\n📚 RAG Answer:\n")
    print(rag_output)

    # === 时间序列分析 ===
    try:
        ts_result, recent_data_str, metrics = analyze_patient_glucose(target_id, optional_input)
    except ValueError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    print("\n🧾 Recent Glucose Records:")
    print(recent_data_str)

    if metrics:
        print("\n📊 Statistical Summary:")
        print(metrics)
    else:
        print("\nℹ️ Not enough data to compute metrics (requires ≥10 records).")

    print("\n✅ GPT Medical Evaluation:")
    print(ts_result)

    # === 综合分析推荐 ===
    combined_prompt = f"""
You are a professional medical assistant.

A user has asked the following medical question, which should remain the primary focus of your response:
[User Question]
{question}

This question has already been answered using a document-augmented retrieval system:
[RAG Answer]
{rag_output[:800]}

In addition, the user has also provided recent glucose data and its analysis:
[Time Series Analysis]
{ts_result}

👉 Your task is to synthesize all this information and provide a clear, medically sound recommendation or insight focused on the original question: "{question}"
""".strip()

    final_response = ask_gpt(combined_prompt)

    print("\n💡 Final LLM Recommendation:\n")
    print(final_response)
