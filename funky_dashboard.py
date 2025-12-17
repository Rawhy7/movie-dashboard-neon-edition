import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
import warnings
import os

# --- 1. PAGE CONFIG & "FUNKY" CSS ---
st.set_page_config(page_title="Future Studio Dashboard", page_icon="🍿", layout="wide")
warnings.filterwarnings('ignore')

# Function to load Lottie Animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

# Load Animations (Cinema & Money themes)
lottie_movie = load_lottieurl("https://lottie.host/8b8e4426-3f0e-4402-b06f-714041d8c0b5/1aS0w5S5gW.json")
lottie_money = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_sf6lGk.json")

# --- CUSTOM CSS FOR 3D & NEON EFFECTS ---
st.markdown("""
    <style>
    /* 1. Background - Cyber Grid */
    .stApp {
        background-color: #050505;
        background-image: linear-gradient(rgba(0, 255, 255, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 30px 30px;
    }

    /* 2. 3D Text Effect */
    h1 {
        color: #fff;
        text-transform: uppercase;
        text-shadow: 4px 4px 0px #d600ff; /* Neon Pink Shadow */
        font-family: 'Courier New', monospace;
        font-weight: 900;
    }
    
    /* 3. Funky Card Container with Hover Animation */
    .funky-card {
        background: rgba(25, 25, 25, 0.9);
        border: 2px solid #00ffff; /* Cyan Border */
        border-radius: 15px;
        padding: 20px;
        box-shadow: 10px 10px 0px #00ffff; /* Hard Shadow for 3D effect */
        transition: transform 0.2s;
        margin-bottom: 25px;
    }
    .funky-card:hover {
        transform: translate(-5px, -5px);
        box-shadow: 15px 15px 0px #d600ff; /* Switch to Pink on Hover */
    }

    /* 4. Custom Buttons */
    div.stButton > button {
        background: linear-gradient(45deg, #ff00cc, #3333ff);
        color: white;
        border: none;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0px 5px 15px rgba(255, 0, 204, 0.4);
        transition: 0.3s;
        border-radius: 50px;
        text-transform: uppercase;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0px 10px 25px rgba(51, 51, 255, 0.6);
    }
    
    /* 5. Metrics Styling */
    label { color: #00ffff !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- 2. ROBUST DATA LOADING ---
@st.cache_resource
def load_data():
    paths = [
        r"D:\Boudy\الجامعة\DS 2\Intro DS\Datasets\Dataset Authentic Movies.xlsx",
        "Dataset Authentic Movies.xlsx",
        "Dataset Authentic Movies.xlsx - Final_Clean_10000_Movies.csv"
    ]
    df = None
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_excel(p) if p.endswith('.xlsx') else pd.read_csv(p)
                break
            except: continue
    
    if df is None: return None, None, None

    # Logic
    if 'Global_BoxOfficeUSD' not in df.columns and 'BoxOfficeUSD' in df.columns:
         df['Global_BoxOfficeUSD'] = df['BoxOfficeUSD']
    
    if 'Global_BoxOfficeUSD' in df.columns:
        df['ROI'] = (df['Global_BoxOfficeUSD'] - df['BudgetUSD']) / df['BudgetUSD']
        df['SuccessStatus'] = df['SuccessStatus'].apply(lambda x: 1 if x in ['Success', 1] else 0)
        df_clean = df[['BudgetUSD', 'Global_BoxOfficeUSD', 'IMDbRating', 'RottenTomatoesScore', 'SuccessStatus', 'ROI']].dropna()
        
        # Train Models
        rf_reg = RandomForestRegressor(n_estimators=50, random_state=42).fit(df_clean[['BudgetUSD']], df_clean['Global_BoxOfficeUSD'])
        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42).fit(df_clean[['BudgetUSD', 'IMDbRating', 'RottenTomatoesScore']], df_clean['SuccessStatus'])
        
        return rf_reg, rf_clf, df_clean
    return None, None, None

fin_model, main_model, df = load_data()

# --- 3. HEADER WITH ANIMATION ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown("<h1>🎬 CYBER-STUDIO <br>EXECUTIVE DASHBOARD</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #00ffff; font-family: monospace;'>// AI-POWERED RISK ANALYSIS SYSTEM v2.0</p>", unsafe_allow_html=True)
with col_head2:
    if lottie_movie: st_lottie(lottie_movie, height=150)

if df is not None:
    # --- 4. INPUTS & 3D VISUALIZATION ---
    st.markdown("---")
    
    col_input, col_viz = st.columns([1, 2])
    
    with col_input:
        st.markdown('<div class="funky-card">', unsafe_allow_html=True)
        st.markdown("### 🎛️ CONTROL PANEL")
        budget = st.slider("💰 Budget ($M)", 1, 350, 80)
        imdb = st.slider("⭐ IMDb Rating", 1.0, 10.0, 7.2)
        rt = st.slider("🍅 Rotten Tomatoes", 0, 100, 75)
        run_btn = st.button("🔮 PREDICT FUTURE")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_viz:
        # 3D SCATTER PLOT
        st.markdown("### 🧊 MARKET METAVERSE (3D DATA)")
        # Create a 3D scatter plot of the actual data
        fig_3d = px.scatter_3d(
            df.sample(1000), # Sample to keep it fast
            x='BudgetUSD', 
            y='Global_BoxOfficeUSD', 
            z='IMDbRating',
            color='SuccessStatus',
            color_continuous_scale='hsv',
            opacity=0.7,
            template="plotly_dark",
            height=400,
            title="Budget vs Revenue vs Quality"
        )
        fig_3d.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_3d, use_container_width=True)

    # --- 5. RESULTS ---
    if run_btn:
        with st.spinner('Accessing the Neural Mainframe...'):
            # Predict
            real_budget = budget * 1e6
            pred_rev = fin_model.predict([[real_budget]])[0]
            pred_roi = (pred_rev - real_budget) / real_budget
            prob = main_model.predict_proba([[real_budget, imdb, rt]])[0][1] * 100
            
            # Determine Vibe
            if prob > 80: 
                vibe = "🚀 MOONSHOT"
                color = "#00ff00" # Lime
                st.balloons()
            elif prob > 50: 
                vibe = "⚠️ GRIND"
                color = "#ffff00" # Yellow
            else: 
                vibe = "💀 FLOP"
                color = "#ff0000" # Red
                st.snow()

            # --- DISPLAY RESULTS IN FUNKY CARDS ---
            st.markdown(f"<h2 style='text-align: center; color: {color}; text-shadow: 0 0 10px {color};'>VERDICT: {vibe} ({prob:.1f}%)</h2>", unsafe_allow_html=True)
            
            res_c1, res_c2, res_c3 = st.columns(3)
            
            with res_c1:
                st.markdown(f"""
                <div class="funky-card" style="border-color: #d600ff; box-shadow: 5px 5px 0 #d600ff;">
                    <h3 style="color:#d600ff">💰 REVENUE</h3>
                    <h1 style="color:white">${pred_rev/1e6:.1f}M</h1>
                    <p>Predicted Global Box Office</p>
                </div>
                """, unsafe_allow_html=True)
                
            with res_c2:
                st.markdown(f"""
                <div class="funky-card" style="border-color: #00ffff; box-shadow: 5px 5px 0 #00ffff;">
                    <h3 style="color:#00ffff">📈 ROI</h3>
                    <h1 style="color:white">{pred_roi*100:.1f}%</h1>
                    <p>Return on Investment</p>
                </div>
                """, unsafe_allow_html=True)
                
            with res_c3:
                st.markdown(f"""
                <div class="funky-card" style="border-color: #ffff00; box-shadow: 5px 5px 0 #ffff00;">
                    <h3 style="color:#ffff00">⚖️ PROFIT</h3>
                    <h1 style="color:white">${(pred_rev - real_budget)/1e6:.1f}M</h1>
                    <p>Net Profit Prediction</p>
                </div>
                """, unsafe_allow_html=True)

            # Interactive Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob,
                title = {'text': "SUCCESS PROBABILITY"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': color},
                    'bgcolor': "black",
                    'borderwidth': 2,
                    'bordercolor': "white",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(255, 0, 0, 0.3)'},
                        {'range': [50, 80], 'color': 'rgba(255, 255, 0, 0.3)'},
                        {'range': [80, 100], 'color': 'rgba(0, 255, 0, 0.3)'}
                    ]}
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)

else:
    st.error("❌ Data not found. Please upload 'Dataset Authentic Movies.xlsx'")

# & C:/Users/abdel/anaconda3/python.exe -m streamlit run "d:/Boudy/الجامعة/DS 2/Algorithimic Foundation/Presentation/funky_dashboard.py"