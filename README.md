# Hybrid AI Framework for Personalized Diabetes Risk Assessment

This project implements a hybrid medical AI framework for assessing diabetes risk. It combines three core components:

- **Retrieval-Augmented Generation (RAG):** Answers diabetes-related questions using embedded clinical guidelines.
- **Glucose Time-Series Analysis:** Interprets individual patient blood glucose records.
- **GPT-4o Reasoning:** Synthesizes medical knowledge and patient data to generate personalized recommendations.

This project was developed by Peidong Liu as part of a final-year research project at the University of Auckland.

---

## Features

- Medical Q&A based on New Zealand clinical guidelines
- Patient glucose risk profiling via statistical analysis
- Integrated response combining LLM-based knowledge and patient data
- Simple, modular structure with three independent Python scripts

---

## System Requirements

- Python 3.8 or higher
- OpenAI API Key (GPT-4 or GPT-4o access required)
- CSV-format glucose data file (columns: `ID`, `time`, `gl`)
- Internet connection for GPT queries

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/padarate88/Diabetes-Risk-Assessment.git
cd Diabetes-Risk-Assessment
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```
---

## Usage

### Full Pipeline: Medical Q&A + Glucose Analysis + Final Recommendation

```bash
python MAIN222.py
```

You will be prompted to:

- Enter a medical question
- Select a patient ID from the glucose data
- (Optional) Add a new glucose reading

The system will:

- Generate a medical answer using RAG 
- Analyze the glucose pattern for that patient
- Use GPT to combine both into a professional, contextual final assessment

---

### RAG Module Only 

```bash
python LLM_rag.py
```

- Retrieves relevant medical guideline content using FAISS
- Uses GPT to answer the user's diabetes-related question

---

### Glucose Analysis Module Only

```bash
python NEWTS.py
```

- Reads time-series glucose data from `Test_data.csv`
- Computes: mean, max, % above 200, % below 70, coefficient of variation
- Uses GPT to interpret results and assess risk level

---

## Project Structure

```
├── LLM_rag.py           # Medical Q&A module using GPT (simulated RAG)
├── NEWTS.py             # Glucose time-series risk analysis
├── MAIN222.py           # Main pipeline: question + data → final output
├── Test_data.csv        # Example anonymized glucose records
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
├── .gitignore           # Git exclusions
```

---

## Methodology

- Embedding model: Sentence-BERT (MiniLM-L6-v2)
- Retrieval: FAISS (local index, optional in simulation)
- Large Language Model: GPT-4o (via OpenAI API)
- Statistical metrics: mean, max, % high (>200), % low (<70), CV
- Integration: Prompt engineering to combine guideline + patient data

---

## Sample Output

```
User question:
Can stress contribute to higher glucose levels?

Glucose profile (Patient ID 003):
Mean: 182.6 mg/dL
Max: 248 mg/dL
% Above 200: 28.4%
% Below 70: 1.2%
CV: 0.22

Final recommendation:
Based on the patient's glucose profile and established clinical evidence regarding the physiological impact of stress, it is plausible that psychological stress is contributing to the observed hyperglycemic episodes. Elevated stress levels can increase cortisol and other counter-regulatory hormones, which may impair insulin sensitivity and lead to sustained high blood glucose levels. In this context, implementing lifestyle interventions—such as structured physical activity, mindfulness-based stress reduction, or cognitive behavioral strategies—may help reduce glycemic variability and improve overall metabolic control.

```

---

## License

```
This project is provided for academic or educational reference only.

You may not copy, modify, redistribute, or use this code or any derivative work for commercial or public deployment without written permission from the author.

All rights reserved.
```

---

## Contact

Peidong Liu 3199519996@qq.com
