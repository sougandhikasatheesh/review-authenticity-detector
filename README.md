# AI-Powered Review Authenticity Detection System

Binary classification: **0 = Human-written**, **1 = AI-generated**

## Project Structure

```
reviewanalyzer/
├── data/
│   ├── raw/                    ← place your original dataset files here
│   │   ├── fake reviews dataset.csv
│   │   └── Dataset of products reviews with annotated Discourses (1).xlsx
│   ├── processed/              ← auto-created by prepare.py
│   └── prepare.py              ← Phase 3: data cleaning + splits
│
├── eda/
│   ├── plots/                  ← auto-created by explore.py
│   └── explore.py              ← Phase 4: EDA plots
│
├── models/
│   ├── plots/                  ← confusion matrices, training curves
│   ├── train_lr.py             ← Phase 5: Logistic Regression baseline
│   └── train_bert.py           ← Phase 6: BERT fine-tuning
│
├── bert_finetuned/             ← saved after running train_bert.py
├── lr_model.pkl                ← saved after running train_lr.py
├── app.py                      ← Phase 7: Streamlit demo
└── requirements.txt
```

## Results

| Model               | Test Accuracy | F1   |
|---------------------|---------------|------|
| Logistic Regression | 93.84%        | —    |
| BERT (fine-tuned)   | 96.60%        | 0.97 |

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt
```

## Running Each Phase

```bash
# Phase 3 — Data preparation (run first)
python data/prepare.py

# Phase 4 — EDA
python eda/explore.py

# Phase 5 — Train Logistic Regression
python models/train_lr.py

# Phase 6 — Fine-tune BERT (needs GPU for reasonable speed)
python models/train_bert.py

# Phase 7 — Run Streamlit app
streamlit run app.py
```

## Notes

- **BERT training** requires a GPU. If you don't have one locally, train in Google Colab and copy the `bert_finetuned/` folder into this project.
- `lr_model.pkl` and `bert_finetuned/` must be in the **project root** for `app.py` to find them.
- Dataset: 42,932 rows, 50/50 class balance.
