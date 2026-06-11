"""
Phase 7 — Streamlit Demo
Review Authenticity Detector: predicts if a review is Human or AI-generated.

Run locally:
    streamlit run app.py
"""

import pickle
import numpy as np
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Review Authenticity Detector",
    page_icon="🔍",
    layout="centered",
)

# ── Custom CSS — dark premium theme ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #080808;
    background-image:
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px),
        radial-gradient(ellipse 70% 50% at 50% 0%, rgba(180,180,180,0.12) 0%, transparent 60%);
    background-size: 20px 20px, 20px 20px, 100% 100%;
    color: #e8e8e8;
}

/* Hide default streamlit header */
#MainMenu, footer, header { visibility: hidden; }

/* Force all containers transparent so grid shows through */
.stApp > div, .stApp > div > div, .stApp > div > div > div,
[data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"],
[data-testid="block-container"], [data-testid="stVerticalBlock"],
[data-testid="stVerticalBlockBorderWrapper"],
.main, .main > div, .block-container,
section[tabindex="0"], section.main > div {
    background-color: transparent !important;
    background: transparent !important;
    color: #e8e8e8 !important;
}

/* Hero section */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem 1rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 999px;
    padding: 0.3rem 1rem;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1.15;
    color: #ffffff;
    margin-bottom: 1rem;
    letter-spacing: -0.03em;
}
.hero-title span {
    background: linear-gradient(135deg, #ffffff 0%, #888888 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1rem;
    color: #aaa;
    font-weight: 400;
    max-width: 420px;
    margin: 0 auto 2.5rem auto;
    line-height: 1.6;
}

/* Stat cards */
.stats-row {
    display: flex;
    gap: 1rem;
    justify-content: center;
    margin-bottom: 2.5rem;
}
.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
    min-width: 130px;
}
.stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    letter-spacing: -0.02em;
}
.stat-label {
    font-size: 0.72rem;
    color: #aaaaaa !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.2rem;
}

/* Input card */
.input-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}
.input-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.75rem;
}

/* textarea — identical to stat cards */
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextArea"] > div > div > textarea,
.stTextArea textarea,
textarea {
    background: #1a1a1a !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    resize: none !important;
    box-shadow: none !important;
    caret-color: #ffffff !important;
}
div[data-testid="stTextArea"] textarea::placeholder,
div[data-testid="stTextArea"] > div > div > textarea::placeholder,
.stTextArea textarea::placeholder,
textarea::placeholder {
    color: rgba(255,255,255,0.3) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.3) !important;
    opacity: 1 !important;
}

/* Radio buttons */
.stRadio > div {
    gap: 0.75rem;
}
.stRadio label {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    color: #cccccc !important;
    -webkit-text-fill-color: #cccccc !important;
    font-size: 0.85rem !important;
    cursor: pointer;
    transition: all 0.15s;
}
.stRadio label:hover {
    border-color: rgba(255,255,255,0.2) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
.stRadio p, .stRadio span, .stRadio div {
    color: #cccccc !important;
    -webkit-text-fill-color: #cccccc !important;
}

/* Analyse button */
.stButton > button {
    background: #ffffff !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 2rem !important;
    letter-spacing: 0.01em !important;
    transition: opacity 0.15s !important;
    width: 100% !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
}

/* Result card */
.result-card {
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1.5rem;
    text-align: center;
}
.result-card.ai {
    background: rgba(255, 59, 59, 0.06);
    border: 1px solid rgba(255, 59, 59, 0.2);
}
.result-card.human {
    background: rgba(59, 255, 130, 0.05);
    border: 1px solid rgba(59, 255, 130, 0.18);
}
.result-icon {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}
.result-label {
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
}
.result-label.ai { color: #ff6b6b; }
.result-label.human { color: #69ffb0; }
.result-sub {
    font-size: 0.82rem;
    color: #555;
}

/* Trust score */
.trust-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1rem;
    text-align: center;
}
.trust-title {
    font-size: 0.72rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 1rem;
}
.trust-gauge-bg {
    height: 6px;
    background: rgba(255,255,255,0.07);
    border-radius: 99px;
    margin: 0.75rem 0;
    overflow: hidden;
}
.trust-gauge-fill {
    height: 6px;
    border-radius: 99px;
    transition: width 0.4s ease;
}
.trust-score {
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    line-height: 1;
}
.trust-verdict {
    font-size: 0.8rem;
    margin-top: 0.4rem;
    font-weight: 500;
}

/* Probability bars */
.prob-row {
    display: flex;
    gap: 1rem;
    margin-top: 1.5rem;
}
.prob-card {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.25rem;
}
.prob-title {
    font-size: 0.72rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.4rem;
}
.prob-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.03em;
}
.prob-bar-bg {
    height: 3px;
    background: rgba(255,255,255,0.07);
    border-radius: 99px;
    margin-top: 0.6rem;
}
.prob-bar-fill {
    height: 3px;
    border-radius: 99px;
}

/* Model tag */
.model-tag {
    display: inline-block;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.72rem;
    color: #555;
    margin-top: 1rem;
    letter-spacing: 0.04em;
}

/* Section label — mid-gray so it reads in both light and dark */
.section-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 0.6rem;
    margin-top: 1.25rem;
}

/* New result card */
.result-card-new {
    background: rgba(255,255,255,0.02);
    border: 0.5px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.5rem;
    margin-top: 1rem;
}
.top-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem; }
.top-left { flex: 1; }
.res-label { font-size: 1.1rem; font-weight: 600; color: #2ea865; letter-spacing: -0.02em; }
.res-sub { font-size: 0.72rem; color: #888; margin-top: 3px; }
.bars { display: flex; flex-direction: column; gap: 0.6rem; margin-top: 0.85rem; }
.bar-row { display: flex; align-items: center; gap: 0.75rem; }
.bar-label { font-size: 11px; color: #aaa; width: 52px; }
.bar-bg { flex: 1; height: 2px; background: rgba(255,255,255,0.05); border-radius: 99px; overflow: hidden; }
.bar-fill { height: 2px; border-radius: 99px; }
.bar-val { font-size: 11px; color: #ccc; width: 32px; text-align: right; }
.circle { width: 72px; height: 72px; position: relative; flex-shrink: 0; }
.circle-inner { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); text-align: center; }
.circle-val { font-size: 1.25rem; font-weight: 700; color: #2ea865; line-height: 1; }
.circle-lbl { font-size: 9px; color: #888; text-transform: uppercase; letter-spacing: 0.06em; }
.steps-sep { border: none; border-top: 0.5px solid rgba(255,255,255,0.06); margin-bottom: 1.25rem; }
.steps { display: flex; }
.step { flex: 1; text-align: center; position: relative; }
.step:not(:last-child)::after { content: ''; position: absolute; top: 11px; left: 56%; width: 88%; height: 0.5px; background: rgba(255,255,255,0.07); }
.dot { width: 22px; height: 22px; border-radius: 50%; margin: 0 auto 0.4rem; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 600; }
.dot.done { background: rgba(46,168,101,0.1); border: 0.5px solid rgba(46,168,101,0.2); color: #2ea865; }
.dot.active { background: #2ea865; color: #000; font-size: 9px; }
.step-lbl { font-size: 9px; color: #888; text-transform: uppercase; letter-spacing: 0.06em; }
.step-val { font-size: 10px; color: #bbb; margin-top: 2px; }

/* Divider */
.custom-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Load models (lazy — only when needed) ─────────────────────────────────
@st.cache_resource
def load_lr():
    with open("lr_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_bert():
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    tokenizer  = BertTokenizer.from_pretrained("bert_finetuned/")
    bert_model = BertForSequenceClassification.from_pretrained("bert_finetuned/")
    bert_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    return tokenizer, bert_model, device


# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI Detection · NLP · Binary Classification</div>
    <div class="hero-title">Is this review <span>real?</span></div>
    <div class="hero-sub">Paste any product review below. Our models will tell you whether it's a genuine review or bot-generated.</div>
</div>

<div class="stats-row">
    <div class="stat-card">
        <div class="stat-value">97%</div>
        <div class="stat-label">BERT F1</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">93.8%</div>
        <div class="stat-label">LR Accuracy</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">42K</div>
        <div class="stat-label">Training rows</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Input ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Review text</div>', unsafe_allow_html=True)
review_text = st.text_area(
    label="review_text",
    label_visibility="collapsed",
    height=160,
    placeholder="Paste a product review here - e.g. This arrived two days early and works exactly as described...",
)

st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
model_choice = st.radio(
    label="model_choice",
    label_visibility="collapsed",
    options=["BERT (more accurate)", "Logistic Regression (faster)"],
    horizontal=True,
)

st.markdown("<br>", unsafe_allow_html=True)
analyse_btn = st.button("Analyse review", use_container_width=True)


# ── Prediction ─────────────────────────────────────────────────────────────
if analyse_btn:
    if not review_text.strip():
        st.warning("Paste a review above to analyse it.")
    else:
        with st.spinner(""):
            if "Logistic Regression" in model_choice:
                lr_pipeline = load_lr()
                prob  = lr_pipeline.predict_proba([review_text])[0]
                label = int(lr_pipeline.predict([review_text])[0])
            else:
                tokenizer, bert_model, device = load_bert()
                inputs = tokenizer(
                    review_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=256,
                ).to(device)
                import torch
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                    prob    = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                label = int(np.argmax(prob))

        card_class   = "ai" if label == 1 else "human"
        icon         = "🤖" if label == 1 else "✅"
        result_label = "Bot-Generated" if label == 1 else "Genuine Review"
        result_sub   = "This review shows strong signals of bot authorship." if label == 1 else "This review reads as a genuine, human-written review."
        result_color = "#cc4444" if label == 1 else "#2ea865"
        genuine_pct  = f"{prob[0]*100:.1f}%"
        bot_pct      = f"{prob[1]*100:.1f}%"
        genuine_w    = f"{prob[0]*100:.0f}%"
        bot_w        = f"{prob[1]*100:.0f}%"
        genuine_color = "#2ea865"
        bot_color     = "#cc4444"
        model_name   = "BERT" if "BERT" in model_choice else "Logistic Regression"

        # Trust score
        trust_score = int(round(prob[0] * 100))
        if trust_score >= 80:
            trust_color   = "#69ffb0"
            trust_verdict = "Highly Trustworthy"
        elif trust_score >= 55:
            trust_color   = "#ffe066"
            trust_verdict = "Likely Genuine"
        elif trust_score >= 35:
            trust_color   = "#ffaa33"
            trust_verdict = "Uncertain"
        else:
            trust_color   = "#ff6b6b"
            trust_verdict = "Likely Bot-Generated"

        word_count = len(review_text.split())
        stroke_offset = int(176 - (trust_score / 100) * 176)

        st.markdown(f"""
        <hr class="custom-divider">
        <div class="result-card-new">
            <div class="top-row">
                <div class="top-left">
                    <div class="res-label" style="color:{result_color};">{icon} {result_label}</div>
                    <div class="res-sub">{result_sub}</div>
                    <div class="bars">
                        <div class="bar-row">
                            <span class="bar-label">Genuine</span>
                            <div class="bar-bg"><div class="bar-fill" style="width:{genuine_w};background:#2ea865;"></div></div>
                            <span class="bar-val">{genuine_pct}</span>
                        </div>
                        <div class="bar-row">
                            <span class="bar-label">Bot</span>
                            <div class="bar-bg"><div class="bar-fill" style="width:{bot_w};background:#aa3333;"></div></div>
                            <span class="bar-val">{bot_pct}</span>
                        </div>
                    </div>
                </div>
                <div class="circle">
                    <svg viewBox="0 0 72 72" style="transform:rotate(-90deg);width:72px;height:72px;">
                        <circle cx="36" cy="36" r="28" fill="none" stroke="rgba(255,255,255,0.04)" stroke-width="5"/>
                        <circle cx="36" cy="36" r="28" fill="none" stroke="{result_color}" stroke-width="5" stroke-dasharray="176" stroke-dashoffset="{stroke_offset}" stroke-linecap="round"/>
                    </svg>
                    <div class="circle-inner">
                        <div class="circle-val" style="color:{result_color};">{trust_score}</div>
                        <div class="circle-lbl">Trust</div>
                    </div>
                </div>
            </div>
            <div class="steps-sep"></div>
            <div class="steps">
                <div class="step">
                    <div class="dot done">✓</div>
                    <div class="step-lbl">Analysed</div>
                    <div class="step-val">{word_count} words</div>
                </div>
                <div class="step">
                    <div class="dot done">✓</div>
                    <div class="step-lbl">Model</div>
                    <div class="step-val">{model_name}</div>
                </div>
                <div class="step">
                    <div class="dot done">✓</div>
                    <div class="step-lbl">Result</div>
                    <div class="step-val">{genuine_pct}</div>
                </div>
                <div class="step">
                    <div class="dot active">{trust_score}</div>
                    <div class="step-lbl">Trust</div>
                    <div class="step-val">{trust_verdict}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)