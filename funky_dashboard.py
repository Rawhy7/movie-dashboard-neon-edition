import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
import os
import requests
from streamlit_lottie import st_lottie
import plotly.graph_objects as go
import time

# --- 1. PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Studio Executive Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings('ignore')

# --- ANIMATION ASSETS LOADER ---
@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Animations (Cinema, Loading, Success)
lottie_cinema = load_lottieurl("https://lottie.host/4a6639d6-5389-4977-9626-474012028889/t2g7z3PzGf.json")
lottie_loading = load_lottieurl("https://lottie.host/95179043-4103-455e-827d-9226de3e9086/Vb0Z7u9bF5.json")

# --- CUSTOM CSS (NOW WITH KEYFRAMES) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Animation Keyframes */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translate3d(0, 40px, 0); }
        to { opacity: 1; transform: translate3d(0, 0, 0); }
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(46, 204, 113, 0); }
        100% { box-shadow: 0 0 0 0 rgba(46, 204, 113, 0); }
    }

    /* Apply Animation to Cards */
    .css-card {
        background-color: #1e2329;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        border: 1px solid #2d333b;
        animation: fadeInUp 0.8s ease-out both; /* Trigger animation */
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #2d333b;
    }
    
    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(75, 108, 183, 0.4);
    }
    
    /* Sliders */
    .stSlider > div > div > div > div {
        background-color: #4b6cb7;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #4b6cb7;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. DATA LOADING & MODEL TRAINING (Cached) ---
@st.cache_resource
def load_and_train_models():
    # PATHS TO CHECK
    paths_to_check = [
        r"D:\Boudy\الجامعة\DS 2\Intro DS\Datasets\Dataset Authentic Movies.xlsx",
        "Dataset Authentic Movies.xlsx",
        "Dataset Authentic Movies.xlsx - Final_Clean_10000_Movies.csv"
    ]
    
    df = None
    loaded_path = ""

    for path in paths_to_check:
        if os.path.exists(path):
            try:
                if path.endswith('.csv'):
                    df = pd.read_csv(path)
                else:
                    df = pd.read_excel(path)
                loaded_path = path
                break
            except Exception as e:
                continue
    
    if df is None:
        st.error("❌ CRITICAL ERROR: Dataset not found!")
        st.info("💡 TIP: Copy 'Dataset Authentic Movies.xlsx' into the same folder as this script.")
        return None, None, None

    # Prepare Data
    required_cols = ['BudgetUSD', 'IMDbRating', 'RottenTomatoesScore', 'SuccessStatus']
    if 'Global_BoxOfficeUSD' not in df.columns and 'BoxOfficeUSD' in df.columns:
         df['Global_BoxOfficeUSD'] = df['BoxOfficeUSD']
    
    if 'Global_BoxOfficeUSD' in df.columns:
        df['ROI'] = (df['Global_BoxOfficeUSD'] - df['BudgetUSD']) / df['BudgetUSD']
        if df['SuccessStatus'].dtype == 'O':
            df['SuccessStatus'] = df['SuccessStatus'].map({'Success': 1, 'Failure': 0})
        
        df_clean = df[['BudgetUSD', 'Global_BoxOfficeUSD', 'IMDbRating', 'RottenTomatoesScore', 'SuccessStatus', 'ROI']].dropna()
        
        # Train Financial Model
        X_fin = df_clean[['BudgetUSD']]
        y_fin = df_clean['Global_BoxOfficeUSD']
        fin_model = RandomForestRegressor(n_estimators=100, random_state=42)
        fin_model.fit(X_fin, y_fin)
        
        # Train Success Probability Model
        X_main = df_clean[['BudgetUSD', 'IMDbRating', 'RottenTomatoesScore']]
        y_main = df_clean['SuccessStatus']
        main_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        main_model.fit(X_main, y_main)
        
        return fin_model, main_model, df_clean
    else:
        st.error(f"Dataset loaded but missing 'Global_BoxOfficeUSD'.")
        return None, None, None

fin_model, main_model, df = load_and_train_models()

# --- 3. UI LAYOUT ---

# Title Section with subtle fade in
st.markdown("""
    <div style="text-align: center; margin-bottom: 20px; animation: fadeInUp 1s ease-out;">
        <h1 style="font-size: 3.5rem; background: -webkit-linear-gradient(eee, #333); -webkit-background-clip: text; color: white; text-shadow: 0 0 20px rgba(75, 108, 183, 0.5);">
            🎬 STUDIO EXECUTIVE <span style="color: #4b6cb7;">DASHBOARD</span>
        </h1>
        <p style="font-size: 1.2rem; color: #8b949e;">AI-Powered Film Investment & Risk Analysis System</p>
    </div>
""", unsafe_allow_html=True)

if fin_model is not None and main_model is not None:
    
    # --- SIDEBAR INPUTS ---
    with st.sidebar:
        # ANIMATED ICON IN SIDEBAR
        if lottie_cinema:
            st_lottie(lottie_cinema, height=150, key="cinema")
            
        st.header("🎛️ Project Parameters")
        st.markdown("Adjust the core parameters.")
        
        st.divider()
        
        budget_m = st.slider("💰 Production Budget ($ Millions)", 1, 350, 80, 1)
        st.divider()
        imdb_score = st.slider("⭐ Projected IMDb Rating", 1.0, 10.0, 7.2, 0.1)
        rt_score = st.slider("🍅 Rotten Tomatoes Score (%)", 0, 100, 75, 1)
        
        st.markdown("---")
        analyze_btn = st.button("🚀 RUN ANALYSIS ENGINE", type="primary")

    # --- MAIN ANALYSIS LOGIC ---
    if analyze_btn:
        
        # ANIMATED LOADING STATE
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if lottie_loading:
                    st_lottie(lottie_loading, height=200, key="loader")
                else:
                    st.spinner('Running AI Simulations...')
            
            # Artificial sleep to let the animation play (optional)
            time.sleep(1.5)

        # Clear the loader by emptying the container (Streamlit reruns the script, 
        # but to make it feel seamless we proceed with calculation)
        
        # Calculations
        real_budget = budget_m * 1_000_000
        pred_revenue = fin_model.predict([[real_budget]])[0]
        pred_roi = (pred_revenue - real_budget) / real_budget
        pred_profit = pred_revenue - real_budget
        prob = main_model.predict_proba([[real_budget, imdb_score, rt_score]])[0][1] * 100
        
        rt_scaled = rt_score / 10
        composite = (imdb_score * 0.6) + (rt_scaled * 0.4)
        gap = abs(imdb_score - rt_scaled)

        # Verdict Logic
        if prob >= 85 and pred_roi > 2.0:
            verdict, verdict_color = "💎 BLOCKBUSTER POTENTIAL", "#2ecc71"
            rec_text = "GREEN LIGHT - Exceptional promise across all metrics."
            st.balloons() # Native Streamlit Animation
        elif prob >= 60 and pred_roi > 0.5:
            verdict, verdict_color = "✅ STRONG CONTENDER", "#3498db"
            rec_text = "RECOMMEND - Solid fundamentals with good potential."
        elif prob >= 40 or pred_roi > 0:
            verdict, verdict_color = "⚖️ MODERATE RISK", "#f1c40f"
            rec_text = "CAUTION - Requires strong marketing execution."
        else:
            verdict, verdict_color = "🛑 HIGH RISK", "#e74c3c"
            rec_text = "RECONSIDER - Significant challenges predicted."

        # Icons
        if pred_roi > 3.0: fin_icon = "💸"
        elif pred_roi > 1.0: fin_icon = "✅"
        elif pred_roi > 0: fin_icon = "⚖️"
        else: fin_icon = "📉"

        if composite >= 8.0: crit_icon = "🏆"
        elif composite >= 6.5: crit_icon = "✅"
        elif gap > 2.0: crit_icon = "⚡"
        else: crit_icon = "📉"

        # --- RESULTS UI ---
        
        # 1. VERDICT BANNER (Animated via CSS class 'css-card')
        st.markdown(f"""
            <div style="background: {verdict_color}20; border: 2px solid {verdict_color}; border-radius: 15px; padding: 30px; text-align: center; margin-bottom: 30px; animation: fadeInUp 0.5s ease-out;">
                <h2 style="color: {verdict_color}; font-size: 3rem; margin: 0; text-transform: uppercase; letter-spacing: 2px;">{verdict}</h2>
                <p style="color: #ddd; font-size: 1.1rem; margin-top: 10px;">{rec_text}</p>
            </div>
        """, unsafe_allow_html=True)

        # 2. GAUGE CHART & METRICS
        col1, col2, col3 = st.columns(3)

        # Financial Column
        with col1:
            st.markdown(f"""
                <div class="css-card" style="border-top: 5px solid #2ecc71; animation-delay: 0.1s;">
                    <h3 style="text-align: center; color: #2ecc71; margin-bottom: 20px;">{fin_icon} FINANCIAL</h3>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <span style="color: #8b949e;">Revenue</span>
                        <span style="font-weight: bold; font-size: 1.2rem;">${pred_revenue/1e6:.1f}M</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <span style="color: #8b949e;">Profit</span>
                        <span style="font-weight: bold; font-size: 1.2rem; color: {'#2ecc71' if pred_profit > 0 else '#e74c3c'}">${pred_profit/1e6:.1f}M</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #8b949e;">ROI</span>
                        <span style="font-weight: bold; font-size: 1.2rem; color: {'#2ecc71' if pred_roi > 0 else '#e74c3c'}">{pred_roi*100:.1f}%</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # PROBABILITY GAUGE (ANIMATED CHART)
        with col2:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Success Probability", 'font': {'size': 18, 'color': "white"}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': verdict_color},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "#333",
                    'steps': [
                        {'range': [0, 40], 'color': "#555"},
                        {'range': [40, 85], 'color': "#777"}],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': prob}}))
            
            fig.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font = {'color': "white", 'family': "Helvetica Neue"}, height=250, margin=dict(l=20, r=20, t=50, b=20))
            
            # Wrap Gauge in a card
            st.markdown('<div class="css-card" style="border-top: 5px solid #3498db; animation-delay: 0.2s; height: 100%;">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Risk Column
        with col3:
            break_even = (real_budget * 2.5) / 1e6
            st.markdown(f"""
                <div class="css-card" style="border-top: 5px solid #9b59b6; animation-delay: 0.3s;">
                    <h3 style="text-align: center; color: #9b59b6; margin-bottom: 20px;">📊 RISK PROFILE</h3>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <span style="color: #8b949e;">Budget</span>
                        <span style="font-weight: bold; font-size: 1.2rem;">${budget_m}M</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <span style="color: #8b949e;">Break-Even</span>
                        <span style="font-weight: bold; font-size: 1.2rem;">~${break_even:.1f}M</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #8b949e;">Critic Gap</span>
                        <span style="font-weight: bold; font-size: 1.2rem; color: {'#e74c3c' if gap > 2 else '#2ecc71'}">{gap:.1f}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # 3. INSIGHTS SECTION
        st.markdown("### 📋 AI Strategic Insights")
        insight_col1, insight_col2 = st.columns(2)
        with insight_col1:
            st.info(f"""
            **💰 Financial Outlook:** {'Based on historical data, projects at this budget level typically see strong returns.' if pred_roi > 1.0 else 'The financial model indicates challenges in achieving profitability.'}
            """)
        with insight_col2:
            st.warning(f"""
            **📣 Critical Reception:** {'Both critics and audiences are expected to respond positively.' if gap < 1.5 and composite > 7 else 'Expect divergent opinions between critics and general audiences.'}
            """)

else:
    st.write("---")

# & C:/Users/abdel/anaconda3/python.exe -m streamlit run "D:\Boudy\الجامعة\DS 2\Algorithimic Foundation\Presentation\funky dashboard\funky_dashboard.py"
