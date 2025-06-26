# Hybrid AI for Personalized Diabetes Risk Assessment

This project implements a hybrid medical AI framework for assessing diabetes risk. It combines three core components:

- Retrieval-Augmented Generation (RAG) to answer diabetes-related questions using embedded clinical guidelines
- Glucose Time-Series Analysis to interpret individual patient data
- GPT-4o-based reasoning to synthesize responses and provide personalized recommendations

This project was developed by Peidong Liu as part of a final-year research project at the University of Auckland.

## Project Structure

The main components of this project include:

- `LLM_rag.py`: A document-based question-answering module that uses FAISS and GPT to retrieve and respond to diabetes-related medical questions
- `NEWTS.py`: A glucose time-series analysis module that uses GPT to assess diabetes risk based on patient blood glucose data
- `MAIN222.py`: A full pipeline that combines user questions, guideline-based answers, and glucose data to provide personalized medical recommendations
- `data/Test_data.csv`: A sample anonymized glucose dataset for testing and demonstration purposes
