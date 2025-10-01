import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import time
import re
from io import StringIO
from urllib.parse import quote, unquote
from typing import Optional

# Optional dependency: SciPy
try:
    from scipy import stats  # type: ignore
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# Optional dependency: Document parsing
try:
    from pypdf import PdfReader  # type: ignore
    HAS_PYPDF = True
except Exception:
    HAS_PYPDF = False

try:
    import fitz  # PyMuPDF  # type: ignore
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

try:
    from pdf2image import convert_from_bytes  # type: ignore
    HAS_PDF2IMAGE = True
except Exception:
    HAS_PDF2IMAGE = False

try:
    import pytesseract  # type: ignore
    HAS_PYTESSERACT = True
except Exception:
    HAS_PYTESSERACT = False

try:
    from docx import Document  # type: ignore
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

def extract_text_from_pdf_with_fallbacks(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes with robust fallbacks: PyPDF → PyMuPDF → OCR.
    Returns empty string if all methods fail.
    """
    text_chunks = []

    # 1) PyPDF (fast)
    if HAS_PYPDF:
        try:
            from io import BytesIO
            reader = PdfReader(BytesIO(pdf_bytes))
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    text_chunks.append(page_text)
        except Exception:
            pass

    if not "".join(text_chunks).strip() and HAS_PYMUPDF:
        # 2) PyMuPDF (better encoding support)
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page in doc:
                page_text = page.get_text() or ""
                if page_text:
                    text_chunks.append(page_text)
        except Exception:
            pass

    if not "".join(text_chunks).strip() and HAS_PYTESSERACT:
        # 3) OCR fallback for scanned PDFs
        try:
            images = []
            if HAS_PDF2IMAGE:
                images = convert_from_bytes(pdf_bytes)
            elif HAS_PYMUPDF:
                # Render pages to images via PyMuPDF if pdf2image/poppler not available
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                for page in doc:
                    pix = page.get_pixmap(dpi=200)
                    from PIL import Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images.append(img)
            for img in images:
                try:
                    page_text = pytesseract.image_to_string(img) or ""
                    if page_text:
                        text_chunks.append(page_text)
                except Exception:
                    continue
        except Exception:
            pass

    return "\n".join(text_chunks).strip()

def extract_text_from_docx_bytes(docx_bytes: bytes) -> str:
    if not HAS_DOCX:
        return ""
    try:
        from io import BytesIO
        doc = Document(BytesIO(docx_bytes))
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""

# Optional dependency: HuggingFace AI-text detector
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
    import torch  # type: ignore
    HAS_HF = True
except Exception:
    HAS_HF = False

@st.cache_resource(show_spinner=False)
def load_ai_text_detector():
    if not HAS_HF:
        return None
    try:
        model_name = "roberta-base-openai-detector"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        return tokenizer, model
    except Exception:
        return None

def detect_ai_generated_with_model(text: str) -> Optional[dict]:
    detector = load_ai_text_detector()
    if detector is None:
        return None
    tokenizer, model = detector
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            return {"human_prob": float(probs[0]), "ai_prob": float(probs[1])}
    except Exception:
        return None

# Page configuration
st.set_page_config(
    page_title="AI vs Human Resume Bias Study",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for animations and styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        animation: slideInDown 1s ease-out;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin: 0;
    }
    
    .metric-card {
        background: rgba(2, 6, 23, 0.68);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        animation: fadeInUp 0.8s ease-out;
        box-shadow: 0 12px 24px rgba(2,6,23,0.45);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        backdrop-filter: blur(12px) saturate(120%);
        -webkit-backdrop-filter: blur(12px) saturate(120%);
        color: #e2e8f0;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
        animation: countUp 2s ease-out;
    }
    
    .stat-label {
        color: #64748b;
        font-size: 0.9rem;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .section-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        animation: slideInLeft 0.8s ease-out;
    }
    
    .survey-card {
        background: rgba(2, 6, 23, 0.6);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        animation: fadeIn 1s ease-out;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 12px 28px rgba(2,6,23,0.45);
        backdrop-filter: blur(14px) saturate(120%);
        -webkit-backdrop-filter: blur(14px) saturate(120%);
        color: #e5e7eb;
    }
    
    .resume-preview {
        background: rgba(2, 6, 23, 0.68); /* deep slate with opacity */
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.14);
        margin: 1rem 0;
        animation: slideInRight 0.8s ease-out;
        box-shadow: 0 12px 32px rgba(2,6,23,0.45);
        backdrop-filter: blur(14px) saturate(120%);
        -webkit-backdrop-filter: blur(14px) saturate(120%);
        color: #f8fafc;
    }
    .resume-preview h4, .resume-preview h5, .resume-preview h6 { color: #f1f5f9; margin-top: .6rem; text-shadow: 0 1px 2px rgba(0,0,0,.35); }
    .resume-preview p, .resume-preview li { color: #e2e8f0; line-height: 1.6; text-shadow: 0 1px 2px rgba(0,0,0,.35); }
    .resume-preview ul { margin-left: 1rem; }
    
    .upload-area {
        border: 2px dashed rgba(102, 126, 234, 0.6);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(2,6,23,0.5);
        margin: 1rem 0;
        animation: pulse 2s infinite;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        color: #e2e8f0;
    }
    
    .ai-result {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        animation: slideInUp 0.8s ease-out;
    }
    
    .human-result {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        animation: slideInUp 0.8s ease-out;
    }
    
    .confidence-bar {
        background: #e2e8f0;
        border-radius: 10px;
        height: 20px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 2s ease-out;
        animation: fillBar 2s ease-out;
    }
    
    .realtime-metric {
        background: linear-gradient(145deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        animation: bounce 2s infinite;
    }
    
    @keyframes slideInDown {
        from { transform: translateY(-50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes fadeInUp {
        from { transform: translateY(30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInRight {
        from { transform: translateX(50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInUp {
        from { transform: translateY(50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes countUp {
        from { opacity: 0; transform: scale(0.5); }
        to { opacity: 1; transform: scale(1); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    @keyframes fillBar {
        from { width: 0%; }
        to { width: var(--target-width); }
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    
    .stSelectbox > div > div { background: rgba(2,6,23,0.55); border-radius: 8px; color: #e2e8f0; border: 1px solid rgba(255,255,255,0.12); }
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea { background: rgba(2,6,23,0.55); color: #e2e8f0; border: 1px solid rgba(255,255,255,0.12); }
    .stSlider > div[data-baseweb="slider"] { background: rgba(148,163,184,0.15); }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Glassmorphism sidebar */
    [data-testid="stSidebar"] > div {
        background: rgba(15, 23, 42, 0.45); /* slate-900 tint */
        backdrop-filter: blur(16px) saturate(120%);
        -webkit-backdrop-filter: blur(16px) saturate(120%);
        border-right: 1px solid rgba(255, 255, 255, 0.12);
    }
    
    /* Glowing navigation button styles */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 16px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 
            0 0 20px rgba(102, 126, 234, 0.6),
            0 0 40px rgba(102, 126, 234, 0.4),
            0 0 60px rgba(102, 126, 234, 0.2);
        animation: glowPulse 2s ease-in-out infinite alternate;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button[kind="primary"]:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button[kind="primary"]:hover:before {
        left: 100%;
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 0 25px rgba(102, 126, 234, 0.8),
            0 0 50px rgba(102, 126, 234, 0.6),
            0 0 75px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.08);
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 12px 16px;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.12);
        border-color: rgba(255, 255, 255, 0.25);
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    @keyframes glowPulse {
        0% {
            box-shadow: 
                0 0 20px rgba(102, 126, 234, 0.6),
                0 0 40px rgba(102, 126, 234, 0.4),
                0 0 60px rgba(102, 126, 234, 0.2);
        }
        100% {
            box-shadow: 
                0 0 25px rgba(102, 126, 234, 0.8),
                0 0 50px rgba(102, 126, 234, 0.6),
                0 0 75px rgba(102, 126, 234, 0.4);
        }
    }
</style>
""", unsafe_allow_html=True)

# AI Detection Function
def analyze_resume_ai_probability(text):
    """Analyze resume text to determine if it's AI-generated"""
    ai_indicators = 0
    total_checks = 0
    
    # Check for AI-typical patterns
    ai_phrases = [
        'results-driven', 'detail-oriented', 'team player', 'excellent communication skills',
        'proven track record', 'dynamic professional', 'innovative solutions',
        'cross-functional teams', 'scalable', 'optimization', 'streamlined',
        'leveraged', 'spearheaded', 'orchestrated', 'facilitated'
    ]
    
    # Check for repetitive patterns
    for phrase in ai_phrases:
        if phrase.lower() in text.lower():
            ai_indicators += 1
        total_checks += 1
    
    # Check sentence structure (AI tends to use more uniform sentence lengths)
    sentences = re.split(r'[.!?]+', text)
    sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
    
    if sentence_lengths:
        avg_length = np.mean(sentence_lengths)
        std_length = np.std(sentence_lengths)
        
        # AI resumes tend to have more consistent sentence lengths
        if std_length < 3 and avg_length > 10:
            ai_indicators += 2
        total_checks += 2
    
    # Check for bullet point consistency (AI tends to be very consistent)
    bullet_lines = [line for line in text.split('\n') if line.strip().startswith(('•', '-', '*'))]
    if len(bullet_lines) > 3:
        bullet_lengths = [len(line.split()) for line in bullet_lines]
        if bullet_lengths and np.std(bullet_lengths) < 2:
            ai_indicators += 1
        total_checks += 1
    
    # Calculate probability
    if total_checks > 0:
        ai_probability = (ai_indicators / total_checks) * 100
        # Add some randomness to make it more realistic
        ai_probability += np.random.normal(0, 10)
        ai_probability = max(0, min(100, ai_probability))
    else:
        ai_probability = 50  # Default if no analysis possible
    
    return ai_probability

# Filename-based classifier (requested rule)
def classify_resume_by_filename(file_name: str) -> str:
    """Classify resume as 'AI' or 'Human' based on special symbols in the filename.
    If filename contains any of: _ # $ ! @ & * => 'AI', else 'Human'.
    """
    special_symbols = set("_#$!@&*")
    return "AI" if any(ch in file_name for ch in special_symbols) else "Human"

# Generate sample data
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    
    # Sample recruiter responses
    industries = ['Technology', 'Finance', 'Healthcare', 'Marketing', 'Education', 'Consulting']
    job_levels = ['Entry Level', 'Mid Level', 'Senior Level', 'Executive']
    resume_types = ['AI Generated', 'Human Written']
    
    data = []
    for _ in range(500):
        data.append({
            'resume_type': np.random.choice(resume_types),
            'industry': np.random.choice(industries),
            'job_level': np.random.choice(job_levels),
            'suitability_score': np.random.normal(7.2, 1.5),
            'professionalism_score': np.random.normal(7.8, 1.2),
            'clarity_score': np.random.normal(7.5, 1.3),
            'hireability_score': np.random.normal(7.0, 1.6),
            'recruiter_experience': np.random.randint(1, 20),
            'date_reviewed': datetime.now() - timedelta(days=np.random.randint(0, 90))
        })
    
    df = pd.DataFrame(data)
    
    # Add slight bias - AI resumes score slightly lower in creative fields
    creative_industries = ['Marketing', 'Education']
    mask = (df['industry'].isin(creative_industries)) & (df['resume_type'] == 'AI Generated')
    df.loc[mask, 'suitability_score'] *= 0.9
    df.loc[mask, 'hireability_score'] *= 0.85
    
    # Ensure scores are within reasonable range
    score_columns = ['suitability_score', 'professionalism_score', 'clarity_score', 'hireability_score']
    for col in score_columns:
        df[col] = np.clip(df[col], 1, 10)
    
    return df

# Generate real-time data
def generate_realtime_data():
    """Generate real-time data points"""
    current_time = datetime.now()
    
    # Simulate incoming resume reviews
    new_reviews = np.random.poisson(5)  # Average 5 new reviews per update
    ai_bias = np.random.normal(0.15, 0.05)  # AI resumes score 15% lower on average
    
    data = {
        'timestamp': current_time,
        'new_reviews': new_reviews,
        'ai_score': max(1, min(10, np.random.normal(6.8, 1.2))),
        'human_score': max(1, min(10, np.random.normal(7.3, 1.1))),
        'bias_percentage': ai_bias * 100,
        'total_reviews': np.random.randint(450, 550)
    }
    
    return data

# Main header
st.markdown("""
<div class="main-header">
    <h1>🤖 AI vs Human Resume Bias Study</h1>
    <p>Analyzing Recruiter Preferences in the Age of AI</p>
</div>
""", unsafe_allow_html=True)

# Load data
df = generate_sample_data()

# Sidebar navigation using Streamlit buttons
st.sidebar.title("📊 Navigation")

NAV_ITEMS = [
    "🏠 Dashboard Overview",
    "📈 Detailed Analysis",
    "📝 Resume Comparison",
    "🔍 Survey Interface",
    "📊 Statistical Results",
    "📄 Resume Upload",
    "📡 Real Time Analysis",
]

# Initialize session state for current page
if 'current_page' not in st.session_state:
    st.session_state.current_page = NAV_ITEMS[0]

# Navigation buttons
for item in NAV_ITEMS:
    is_active = item == st.session_state.current_page
    button_type = "primary" if is_active else "secondary"
    
    if st.sidebar.button(
        item, 
        key=f"nav_{item}", 
        type=button_type,
        use_container_width=True
    ):
        st.session_state.current_page = item
        st.rerun()

# Set the current page
page = st.session_state.current_page

if page == "🏠 Dashboard Overview":
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <p class="stat-number">500</p>
            <p class="stat-label">Total Reviews</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        ai_avg = df[df['resume_type'] == 'AI Generated']['hireability_score'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <p class="stat-number">{ai_avg:.1f}</p>
            <p class="stat-label">AI Resume Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        human_avg = df[df['resume_type'] == 'Human Written']['hireability_score'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <p class="stat-number">{human_avg:.1f}</p>
            <p class="stat-label">Human Resume Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        bias_score = ((human_avg - ai_avg) / human_avg * 100)
        st.markdown(f"""
        <div class="metric-card">
            <p class="stat-number">{bias_score:.1f}%</p>
            <p class="stat-label">Bias Difference</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">📊 Overall Performance Comparison</h2>', unsafe_allow_html=True)
    
    # Overall comparison chart
    score_cols = ['suitability_score', 'professionalism_score', 'clarity_score', 'hireability_score']
    comparison_data = []
    
    for score_type in score_cols:
        ai_mean = df[df['resume_type'] == 'AI Generated'][score_type].mean()
        human_mean = df[df['resume_type'] == 'Human Written'][score_type].mean()
        
        comparison_data.extend([
            {'Score Type': score_type.replace('_score', '').title(), 'Resume Type': 'AI Generated', 'Average Score': ai_mean},
            {'Score Type': score_type.replace('_score', '').title(), 'Resume Type': 'Human Written', 'Average Score': human_mean}
        ])
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig = px.bar(comparison_df, x='Score Type', y='Average Score', color='Resume Type',
                 barmode='group', 
                 color_discrete_map={'AI Generated': '#ff6b6b', 'Human Written': '#4ecdc4'},
                 title="Average Scores by Resume Type")
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family="Inter",
        title_font_size=20,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Industry breakdown
    st.markdown('<h2 class="section-header">🏢 Industry Analysis</h2>', unsafe_allow_html=True)
    
    industry_avg = df.groupby(['industry', 'resume_type'])['hireability_score'].mean().reset_index()
    
    fig2 = px.sunburst(industry_avg, path=['industry', 'resume_type'], values='hireability_score',
                       color='hireability_score', 
                       color_continuous_scale='RdYlBu_r',
                       title="Hireability Scores by Industry and Resume Type")
    
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family="Inter"
    )
    
    st.plotly_chart(fig2, use_container_width=True)

elif page == "📈 Detailed Analysis":
    st.markdown('<h2 class="section-header">📈 Detailed Statistical Analysis</h2>', unsafe_allow_html=True)
    
    # Time series analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily trend
        daily_scores = df.groupby([df['date_reviewed'].dt.date, 'resume_type'])['hireability_score'].mean().reset_index()
        daily_scores['date_reviewed'] = pd.to_datetime(daily_scores['date_reviewed'])
        
        fig3 = px.line(daily_scores, x='date_reviewed', y='hireability_score', color='resume_type',
                       title="Daily Hireability Trends",
                       color_discrete_map={'AI Generated': '#ff6b6b', 'Human Written': '#4ecdc4'})
        
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family="Inter"
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Score distribution
        fig4 = go.Figure()
        
        for resume_type in ['AI Generated', 'Human Written']:
            scores = df[df['resume_type'] == resume_type]['hireability_score']
            fig4.add_trace(go.Histogram(
                x=scores,
                name=resume_type,
                opacity=0.7,
                nbinsx=20
            ))
        
        fig4.update_layout(
            title="Score Distribution Comparison",
            xaxis_title="Hireability Score",
            yaxis_title="Frequency",
            barmode='overlay',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family="Inter"
        )
        
        st.plotly_chart(fig4, use_container_width=True)
    
    # Correlation heatmap
    st.markdown('<h3 class="section-header">🔗 Score Correlations</h3>', unsafe_allow_html=True)
    
    correlation_data = df[['suitability_score', 'professionalism_score', 'clarity_score', 'hireability_score', 'recruiter_experience']].corr()
    
    fig5 = px.imshow(correlation_data, 
                     text_auto=True, 
                     aspect="auto",
                     color_continuous_scale='RdBu_r',
                     title="Correlation Matrix of Scoring Factors")
    
    fig5.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family="Inter"
    )
    
    st.plotly_chart(fig5, use_container_width=True)

elif page == "📝 Resume Comparison":
    st.markdown('<h2 class="section-header">📝 Resume Comparison Tool</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="resume-preview">
            <h3 style="color: #ff6b6b;">🤖 AI-Generated Resume</h3>
            <hr>
            <h4>John Smith</h4>
            <p><strong>Software Engineer</strong></p>
            <p>📧 john.smith@email.com | 📱 (555) 123-4567</p>
            
            <h5>Professional Summary</h5>
            <p>Results-driven software engineer with 5+ years of experience developing scalable web applications. Proficient in JavaScript, Python, and cloud technologies. Demonstrated ability to lead cross-functional teams and deliver high-quality solutions.</p>
            
            <h5>Technical Skills</h5>
            <ul>
                <li>Programming: JavaScript, Python, Java, C++</li>
                <li>Frameworks: React, Node.js, Django, Spring</li>
                <li>Databases: PostgreSQL, MongoDB, Redis</li>
                <li>Cloud: AWS, Docker, Kubernetes</li>
            </ul>
            
            <h5>Experience</h5>
            <p><strong>Senior Software Engineer</strong> - TechCorp (2020-Present)</p>
            <ul>
                <li>Led development of microservices architecture serving 1M+ users</li>
                <li>Improved application performance by 40% through optimization</li>
                <li>Mentored junior developers and conducted code reviews</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="resume-preview">
            <h3 style="color: #4ecdc4;">👤 Human-Written Resume</h3>
            <hr>
            <h4>Jane Doe</h4>
            <p><strong>Software Developer</strong></p>
            <p>📧 jane.doe@email.com | 📱 (555) 987-6543</p>
            
            <h5>About Me</h5>
            <p>I'm a passionate software developer who loves creating innovative solutions. With a background in computer science and a knack for problem-solving, I've spent the last 4 years building web applications that make a difference. I thrive in collaborative environments and enjoy learning new technologies.</p>
            
            <h5>Skills</h5>
            <ul>
                <li>Languages: JavaScript, Python, HTML/CSS</li>
                <li>Tools: React, Express.js, Flask</li>
                <li>Databases: MySQL, SQLite</li>
                <li>Other: Git, Agile methodologies</li>
            </ul>
            
            <h5>Work Experience</h5>
            <p><strong>Software Developer</strong> - StartupXYZ (2019-Present)</p>
            <ul>
                <li>Built and maintained the company's main web platform</li>
                <li>Collaborated with designers to create user-friendly interfaces</li>
                <li>Participated in daily standups and sprint planning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Rating interface
    st.markdown('<h3 class="section-header">⭐ Rate These Resumes</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("AI Resume Rating")
        ai_suitability = st.slider("Suitability (AI)", 1, 10, 7, key="ai_suit")
        ai_professionalism = st.slider("Professionalism (AI)", 1, 10, 8, key="ai_prof")
        ai_clarity = st.slider("Clarity (AI)", 1, 10, 7, key="ai_clar")
        ai_hireability = st.slider("Hireability (AI)", 1, 10, 6, key="ai_hire")
    
    with col2:
        st.subheader("Human Resume Rating")
        human_suitability = st.slider("Suitability (Human)", 1, 10, 7, key="human_suit")
        human_professionalism = st.slider("Professionalism (Human)", 1, 10, 7, key="human_prof")
        human_clarity = st.slider("Clarity (Human)", 1, 10, 8, key="human_clar")
        human_hireability = st.slider("Hireability (Human)", 1, 10, 8, key="human_hire")
    
    if st.button("📊 Submit Ratings", key="submit_ratings"):
        st.success("✅ Thank you for your feedback! Your ratings have been recorded.")
        
        # Show comparison
        comparison_data = {
            'Metric': ['Suitability', 'Professionalism', 'Clarity', 'Hireability'],
            'AI Resume': [ai_suitability, ai_professionalism, ai_clarity, ai_hireability],
            'Human Resume': [human_suitability, human_professionalism, human_clarity, human_hireability]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig = px.bar(comparison_df, x='Metric', y=['AI Resume', 'Human Resume'],
                     barmode='group', title="Your Rating Comparison")
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family="Inter"
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔍 Survey Interface":
    st.markdown('<h2 class="section-header">🔍 Recruiter Survey Interface</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="survey-card">
        <h3>📋 Participant Information</h3>
        <p>Please provide some background information to help us analyze the results.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        recruiter_industry = st.selectbox("Your Industry", 
                                        ['Technology', 'Finance', 'Healthcare', 'Marketing', 'Education', 'Consulting', 'Other'])
        recruiter_experience = st.slider("Years of Recruiting Experience", 1, 30, 5)
        company_size = st.selectbox("Company Size", 
                                  ['Startup (1-50)', 'Small (51-200)', 'Medium (201-1000)', 'Large (1000+)'])
    
    with col2:
        hiring_frequency = st.selectbox("How often do you review resumes?", 
                                      ['Daily', 'Weekly', 'Monthly', 'Quarterly'])
        ai_familiarity = st.slider("Familiarity with AI tools (1-10)", 1, 10, 5)
        preferred_format = st.selectbox("Preferred Resume Format", 
                                      ['Traditional', 'Modern', 'Creative', 'No Preference'])
    
    st.markdown("""
    <div class="survey-card">
        <h3>📄 Resume Review Process</h3>
        <p>You will be shown a series of resumes. Please rate each one without knowing whether it was AI-generated or human-written.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample resume for rating
    sample_resume = st.selectbox("Select Resume to Review", 
                               ['Resume A', 'Resume B', 'Resume C', 'Resume D', 'Resume E'])
    
    if sample_resume:
        st.markdown(f"""
        <div class="resume-preview">
            <h4>Resume {sample_resume[-1]}</h4>
            <hr>
            <h5>Alex Johnson</h5>
            <p><strong>Marketing Manager</strong></p>
            <p>📧 alex.johnson@email.com | 📱 (555) 456-7890</p>
            
            <h6>Professional Summary</h6>
            <p>Dynamic marketing professional with 6+ years of experience in digital marketing, brand management, and campaign optimization. Proven track record of increasing brand awareness and driving revenue growth through innovative marketing strategies.</p>
            
            <h6>Key Achievements</h6>
            <ul>
                <li>Increased social media engagement by 150% in 12 months</li>
                <li>Led successful product launch generating $2M in first quarter</li>
                <li>Managed marketing budget of $500K+ with 20% cost reduction</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Rating form
        col1, col2 = st.columns(2)
        
        with col1:
            suitability = st.slider(f"Suitability for Marketing Role", 1, 10, 7, key=f"suit_{sample_resume}")
            professionalism = st.slider(f"Professionalism", 1, 10, 8, key=f"prof_{sample_resume}")
        
        with col2:
            clarity = st.slider(f"Clarity and Organization", 1, 10, 7, key=f"clar_{sample_resume}")
            hireability = st.slider(f"Overall Hireability", 1, 10, 7, key=f"hire_{sample_resume}")
        
        additional_comments = st.text_area("Additional Comments (Optional)", 
                                         placeholder="Any specific observations about this resume...")
        
        if st.button(f"Submit Rating for {sample_resume}", key=f"submit_{sample_resume}"):
            st.success(f"✅ Rating submitted for {sample_resume}!")
            
            # Store the rating (in a real app, this would go to a database)
            rating_data = {
                'resume_id': sample_resume,
                'suitability': suitability,
                'professionalism': professionalism,
                'clarity': clarity,
                'hireability': hireability,
                'comments': additional_comments,
                'recruiter_industry': recruiter_industry,
                'recruiter_experience': recruiter_experience
            }
            
            st.json(rating_data)

elif page == "📊 Statistical Results":
    st.markdown('<h2 class="section-header">📊 Statistical Analysis Results</h2>', unsafe_allow_html=True)
    
    ai_scores = df[df['resume_type'] == 'AI Generated']['hireability_score']
    human_scores = df[df['resume_type'] == 'Human Written']['hireability_score']
    
    if HAS_SCIPY:
        # T-test
        t_stat, p_value = stats.ttest_ind(ai_scores, human_scores)
    else:
        t_stat, p_value = np.nan, np.nan
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        t_text = f"{t_stat:.3f}" if HAS_SCIPY else "Install SciPy to compute"
        st.markdown(f"""
        <div class="metric-card">
            <p class="stat-number">{t_text}</p>
            <p class="stat-label">T-Statistic</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        p_text = f"{p_value:.4f}" if HAS_SCIPY else "Install SciPy to compute"
        st.markdown(f"""
        <div class="metric-card">
            <p class="stat-number">{p_text}</p>
            <p class="stat-label">P-Value</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if HAS_SCIPY:
            significance = "Significant" if p_value < 0.05 else "Not Significant"
        else:
            significance = "SciPy not installed"
        st.markdown(f"""
        <div class="metric-card">
            <p class="stat-number" style="font-size: 1.5rem;">{significance}</p>
            <p class="stat-label">Statistical Significance</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(ai_scores) - 1) * ai_scores.var() + (len(human_scores) - 1) * human_scores.var()) / (len(ai_scores) + len(human_scores) - 2))
    cohens_d = (human_scores.mean() - ai_scores.mean()) / pooled_std
    
    st.markdown('<h3 class="section-header">📈 Effect Size Analysis</h3>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="survey-card">
        <h4>Cohen's d: {cohens_d:.3f}</h4>
        <p><strong>Interpretation:</strong> 
        {"Small effect" if abs(cohens_d) < 0.5 else "Medium effect" if abs(cohens_d) < 0.8 else "Large effect"}
        </p>
        <p>This indicates the magnitude of difference between AI and human resume ratings.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Box plot comparison
    fig_box = go.Figure()
    
    fig_box.add_trace(go.Box(
        y=ai_scores,
        name='AI Generated',
        boxpoints='outliers',
        marker_color='#ff6b6b'
    ))
    
    fig_box.add_trace(go.Box(
        y=human_scores,
        name='Human Written',
        boxpoints='outliers',
        marker_color='#4ecdc4'
    ))
    
    fig_box.update_layout(
        title="Score Distribution Comparison (Box Plot)",
        yaxis_title="Hireability Score",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family="Inter"
    )
    
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Industry-specific analysis
    st.markdown('<h3 class="section-header">🏢 Industry-Specific Bias Analysis</h3>', unsafe_allow_html=True)
    
    industry_stats = []
    for industry in df['industry'].unique():
        industry_data = df[df['industry'] == industry]
        ai_industry = industry_data[industry_data['resume_type'] == 'AI Generated']['hireability_score']
        human_industry = industry_data[industry_data['resume_type'] == 'Human Written']['hireability_score']
        
        if len(ai_industry) > 5 and len(human_industry) > 5:
            bias_score = (human_industry.mean() - ai_industry.mean()) / max(human_industry.mean(), 1e-6) * 100
            if HAS_SCIPY:
                t_stat_ind, p_val_ind = stats.ttest_ind(ai_industry, human_industry)
                p_value_ind = float(p_val_ind)
                significant = p_value_ind < 0.05
            else:
                p_value_ind = np.nan
                significant = False
            
            industry_stats.append({
                'Industry': industry,
                'AI Mean': ai_industry.mean(),
                'Human Mean': human_industry.mean(),
                'Bias %': bias_score,
                'P-Value': p_value_ind,
                'Significant': significant
            })
    
    industry_df = pd.DataFrame(industry_stats)
    
    if not industry_df.empty:
        fig_industry = px.bar(industry_df, x='Industry', y='Bias %',
                             color='Significant',
                             color_discrete_map={True: '#ff6b6b', False: '#95a5a6'},
                             title="Bias Percentage by Industry")
        
        fig_industry.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family="Inter"
        )
        
        st.plotly_chart(fig_industry, use_container_width=True)
        
        st.dataframe(industry_df.round(3), use_container_width=True)

elif page == "📄 Resume Upload":
    st.markdown('<h2 class="section-header">📄 AI Resume Detection Tool</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="survey-card">
        <h3>🔍 Upload Resume for AI Analysis</h3>
        <p>Upload a resume and our AI will analyze whether it was likely generated by AI or written by a human.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "📎 Choose a resume file (PDF, DOCX, TXT)",
        type=['pdf', 'docx', 'txt'],
        help="Upload a resume file to analyze whether it's AI-generated or human-written"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Text input as alternative
    st.markdown("**Or paste resume text directly:**")
    resume_text = st.text_area(
        "Resume Content",
        placeholder="Paste the resume content here...",
        height=200
    )
    
    # Analysis button
    if st.button("🔍 Analyze Resume", key="analyze_resume"):
        text_to_analyze = ""
        
        # Process uploaded file
        if uploaded_file is not None:
            # Priority: filename rule overrides text analysis
            file_label = classify_resume_by_filename(getattr(uploaded_file, "name", ""))
            if file_label == "AI":
                # Show as human-style card (no red alert) and skip further analysis
                st.markdown(f"""
                <div class="human-result">
                    <h3>🤖 AI Generated</h3>
                    <p><strong>100.0% probability</strong></p>
g                </div>
                """, unsafe_allow_html=True)
                st.stop()
            
            if uploaded_file.type == "text/plain":
                text_to_analyze = str(uploaded_file.read(), "utf-8", errors="ignore")
            elif uploaded_file.type == "application/pdf":
                with st.spinner("📄 Extracting text from PDF..."):
                    file_bytes = uploaded_file.read()
                    text_to_analyze = extract_text_from_pdf_with_fallbacks(file_bytes)
                    if not text_to_analyze.strip():
                        st.warning("⚠️ Could not extract text from this PDF. Please try another file or paste the text.")
            elif uploaded_file.type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",):
                with st.spinner("📄 Extracting text from DOCX..."):
                    file_bytes = uploaded_file.read()
                    text_to_analyze = extract_text_from_docx_bytes(file_bytes)
                    if not text_to_analyze.strip():
                        st.warning("⚠️ Could not extract text from this DOCX. Please try another file or paste the text.")
            else:
                st.warning("📝 Unsupported file type. Please upload .pdf, .docx, or .txt, or paste text.")
                text_to_analyze = ""
        else:
            text_to_analyze = resume_text
        
        if text_to_analyze.strip():
            # Show loading animation
            with st.spinner("🤖 Analyzing resume patterns..."):
                time.sleep(2)  # Simulate processing time
                
                # Analyze the resume
                detector_result = detect_ai_generated_with_model(text_to_analyze)
                if detector_result is not None:
                    ai_probability = detector_result.get("ai_prob", 0.0) * 100.0
                else:
                    ai_probability = analyze_resume_ai_probability(text_to_analyze)
                confidence = abs(ai_probability - 50) * 2  # Convert to confidence percentage
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    if ai_probability > 60:
                        st.markdown(f"""
                        <div class="ai-result">
                            <h3>🤖 AI Generated</h3>
                            <p><strong>{ai_probability:.1f}% probability</strong></p>
                            <p>This resume shows patterns typical of AI-generated content.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif ai_probability < 40:
                        st.markdown(f"""
                        <div class="human-result">
                            <h3>👤 Human Written</h3>
                            <p><strong>{100-ai_probability:.1f}% probability</strong></p>
                            <p>This resume shows patterns typical of human-written content.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="survey-card">
                            <h3>🤔 Uncertain</h3>
                            <p><strong>Mixed patterns detected</strong></p>
                            <p>This resume contains both AI and human writing characteristics.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Confidence meter
                    st.markdown("**Analysis Confidence**")
                    st.markdown(f"""
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence}%; background: linear-gradient(90deg, #ff6b6b, #4ecdc4);"></div>
                    </div>
                    <p style="text-align: center;">{confidence:.1f}% Confident</p>
                    """, unsafe_allow_html=True)
                    
                    # Detailed analysis
                    st.markdown("**Key Indicators:**")
                    
                    # Check for AI phrases
                    ai_phrases = ['results-driven', 'detail-oriented', 'team player', 'proven track record']
                    found_phrases = [phrase for phrase in ai_phrases if phrase.lower() in text_to_analyze.lower()]
                    
                    if found_phrases:
                        st.write("🔍 AI-typical phrases found:")
                        for phrase in found_phrases[:3]:  # Show max 3
                            st.write(f"  • {phrase}")
                    
                    # Sentence analysis
                    sentences = re.split(r'[.!?]+', text_to_analyze)
                    sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
                    
                    if sentence_lengths:
                        avg_length = np.mean(sentence_lengths)
                        st.write(f"📏 Average sentence length: {avg_length:.1f} words")
                        
                        if avg_length > 15:
                            st.write("  • Long sentences (AI tendency)")
                        elif avg_length < 8:
                            st.write("  • Short sentences (Human tendency)")
                
                # Detailed breakdown chart
                st.markdown('<h3 class="section-header">📊 Analysis Breakdown</h3>', unsafe_allow_html=True)
                
                # Create radar chart for different aspects
                categories = ['Formal Language', 'Consistency', 'Buzzwords', 'Structure', 'Originality']
                
                # Generate scores for each category (simplified for demo)
                ai_scores = [
                    min(100, ai_probability + np.random.normal(0, 10)),
                    min(100, ai_probability + np.random.normal(5, 8)),
                    min(100, ai_probability + np.random.normal(-5, 12)),
                    min(100, ai_probability + np.random.normal(3, 7)),
                    min(100, 100 - ai_probability + np.random.normal(0, 10))
                ]
                ai_scores = [max(0, score) for score in ai_scores]
                
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=ai_scores,
                    theta=categories,
                    fill='toself',
                    name='AI Likelihood',
                    line_color='#ff6b6b'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=True,
                    title="Resume Analysis Breakdown",
                    font_family="Inter"
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
                
        else:
            st.error("⚠️ Please upload a file or paste resume text to analyze.")

elif page == "📡 Real Time Analysis":
    st.markdown('<h2 class="section-header">📡 Real Time Statistical Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="survey-card">
        <h3>⚡ Live Data Stream</h3>
        <p>Watch real-time updates of resume review statistics and bias metrics as new data comes in.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("🔄 Auto-refresh (every 3 seconds)", value=True)
    
    if auto_refresh:
        # Initialize session state for storing real-time data
        if 'realtime_data' not in st.session_state:
            st.session_state.realtime_data = []
            st.session_state.last_update = datetime.now()
        
        # Create placeholder for real-time updates
        placeholder = st.empty()
        
        # Generate new data point
        current_time = datetime.now()
        if (current_time - st.session_state.last_update).seconds >= 3:
            new_data = generate_realtime_data()
            st.session_state.realtime_data.append(new_data)
            st.session_state.last_update = current_time
            
            # Keep only last 20 data points
            if len(st.session_state.realtime_data) > 20:
                st.session_state.realtime_data = st.session_state.realtime_data[-20:]
        
        with placeholder.container():
            # Real-time metrics
            col1, col2, col3, col4 = st.columns(4)
            
            if st.session_state.realtime_data:
                latest_data = st.session_state.realtime_data[-1]
                
                with col1:
                    st.markdown(f"""
                    <div class="realtime-metric">
                        <h3>{latest_data['new_reviews']}</h3>
                        <p>New Reviews</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="realtime-metric">
                        <h3>{latest_data['ai_score']:.1f}</h3>
                        <p>AI Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="realtime-metric">
                        <h3>{latest_data['human_score']:.1f}</h3>
                        <p>Human Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="realtime-metric">
                        <h3>{latest_data['bias_percentage']:.1f}%</h3>
                        <p>Bias Level</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Real-time charts
            if len(st.session_state.realtime_data) > 1:
                realtime_df = pd.DataFrame(st.session_state.realtime_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Live score comparison
                    fig_live = go.Figure()
                    
                    fig_live.add_trace(go.Scatter(
                        x=realtime_df['timestamp'],
                        y=realtime_df['ai_score'],
                        mode='lines+markers',
                        name='AI Score',
                        line=dict(color='#ff6b6b', width=3),
                        marker=dict(size=8)
                    ))
                    
                    fig_live.add_trace(go.Scatter(
                        x=realtime_df['timestamp'],
                        y=realtime_df['human_score'],
                        mode='lines+markers',
                        name='Human Score',
                        line=dict(color='#4ecdc4', width=3),
                        marker=dict(size=8)
                    ))
                    
                    fig_live.update_layout(
                        title="Live Score Trends",
                        xaxis_title="Time",
                        yaxis_title="Score",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_family="Inter",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_live, use_container_width=True)
                
                with col2:
                    # Live bias percentage
                    fig_bias = go.Figure()
                    
                    fig_bias.add_trace(go.Scatter(
                        x=realtime_df['timestamp'],
                        y=realtime_df['bias_percentage'],
                        mode='lines+markers',
                        name='Bias %',
                        line=dict(color='#9b59b6', width=3),
                        marker=dict(size=8),
                        fill='tonexty'
                    ))
                    
                    fig_bias.update_layout(
                        title="Live Bias Percentage",
                        xaxis_title="Time",
                        yaxis_title="Bias %",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_family="Inter"
                    )
                    
                    st.plotly_chart(fig_bias, use_container_width=True)
                
                # Volume chart
                fig_volume = go.Figure()
                
                fig_volume.add_trace(go.Bar(
                    x=realtime_df['timestamp'],
                    y=realtime_df['new_reviews'],
                    name='New Reviews',
                    marker_color='#f39c12'
                ))
                
                fig_volume.update_layout(
                    title="Real-time Review Volume",
                    xaxis_title="Time",
                    yaxis_title="Number of Reviews",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_family="Inter"
                )
                
                st.plotly_chart(fig_volume, use_container_width=True)
        
        # Auto-refresh every 3 seconds
        time.sleep(3)
        st.rerun()
    
    else:
        st.info("🔄 Enable auto-refresh to see live data updates")
        
        # Show static sample of what real-time data would look like
        sample_times = [datetime.now() - timedelta(minutes=x) for x in range(10, 0, -1)]
        sample_data = []
        
        for t in sample_times:
            sample_data.append({
                'timestamp': t,
                'new_reviews': np.random.poisson(5),
                'ai_score': np.random.normal(6.8, 1.2),
                'human_score': np.random.normal(7.3, 1.1),
                'bias_percentage': np.random.normal(15, 5)
            })
        
        sample_df = pd.DataFrame(sample_data)
        
        # Sample charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_sample = px.line(sample_df, x='timestamp', y=['ai_score', 'human_score'],
                               title="Sample: Score Trends Over Time")
            fig_sample.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_family="Inter"
            )
            st.plotly_chart(fig_sample, use_container_width=True)
        
        with col2:
            fig_sample2 = px.bar(sample_df, x='timestamp', y='new_reviews',
                               title="Sample: Review Volume")
            fig_sample2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_family="Inter"
            )
            st.plotly_chart(fig_sample2, use_container_width=True)

# Footer
st.markdown("""
---
<div style="text-align: center; color: #64748b; padding: 2rem;">
    <p>🎓 AI vs Human Resume Bias Study | Built with Streamlit & Plotly</p>
    <p>📊 Interactive Data Visualization | 🔬 Statistical Analysis</p>
</div>
""", unsafe_allow_html=True)