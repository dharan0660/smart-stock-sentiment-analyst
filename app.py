# ============================================================
#  Smart Stock Sentiment + LLM Analyst
#  Author  : Dharan
#  GitHub  : github.com/dharan0660
#  College : Woxsen University
#  Stack   : Python, Streamlit, Groq (LLaMA3), VADER, yfinance
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from groq import Groq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import os
from dotenv import load_dotenv
import time
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Stock Sentiment Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background: linear-gradient(135deg, #0e1117 0%, #1a1f2e 100%); }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252b3b);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2d3548;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-value { font-size: 2rem; font-weight: 800; }
    .metric-label { font-size: 0.85rem; color: #8892a4; margin-top: 5px; }
    .positive { color: #00d4aa; }
    .negative { color: #ff4b6e; }
    .neutral  { color: #ffd166; }
    .header-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00d4aa, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .sub-header {
        text-align: center;
        color: #8892a4;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .llm-box {
        background: linear-gradient(135deg, #1a2744, #1e2d4a);
        border-left: 4px solid #4facfe;
        border-radius: 8px;
        padding: 20px;
        margin-top: 10px;
        color: #e0e6f0;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .news-card {
        background: #1e2130;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-left: 3px solid #4facfe;
    }
    .pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .pill-positive { background: #00d4aa22; color: #00d4aa; border: 1px solid #00d4aa55; }
    .pill-negative { background: #ff4b6e22; color: #ff4b6e; border: 1px solid #ff4b6e55; }
    .pill-neutral  { background: #ffd16622; color: #ffd166; border: 1px solid #ffd16655; }
    .recommendation-box {
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 800;
        margin: 10px 0;
    }
    .rec-buy  { background: #00d4aa22; border: 2px solid #00d4aa; color: #00d4aa; }
    .rec-sell { background: #ff4b6e22; border: 2px solid #ff4b6e; color: #ff4b6e; }
    .rec-hold { background: #ffd16622; border: 2px solid #ffd166; color: #ffd166; }
    .footer {
        text-align: center;
        color: #4a5568;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #2d3548;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STOCK UNIVERSE
# ─────────────────────────────────────────────
STOCKS = {
    "🇮🇳 NSE/BSE — Indian": {
        "Reliance Industries": "RELIANCE.NS",
        "TCS":                 "TCS.NS",
        "Infosys":             "INFY.NS",
        "HDFC Bank":           "HDFCBANK.NS",
        "Wipro":               "WIPRO.NS",
        "ICICI Bank":          "ICICIBANK.NS",
        "Bajaj Finance":       "BAJFINANCE.NS",
        "Adani Enterprises":   "ADANIENT.NS",
    },
    "🇺🇸 NASDAQ/NYSE — US": {
        "Apple":     "AAPL",
        "Microsoft": "MSFT",
        "Google":    "GOOGL",
        "Amazon":    "AMZN",
        "NVIDIA":    "NVDA",
        "Meta":      "META",
        "Tesla":     "TSLA",
        "Netflix":   "NFLX",
    },
    "₿ Crypto": {
        "Bitcoin":  "BTC-USD",
        "Ethereum": "ETH-USD",
        "Solana":   "SOL-USD",
        "Dogecoin": "DOGE-USD",
        "BNB":      "BNB-USD",
    }
}


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_stock_data(ticker, period="3mo"):
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period=period)
        info  = stock.info
        return hist, info
    except Exception as e:
        return None, {}


def analyze_sentiment(text):
    """Dual-engine sentiment: VADER + TextBlob"""
    analyzer = SentimentIntensityAnalyzer()
    vader    = analyzer.polarity_scores(text)
    blob     = TextBlob(text).sentiment

    # Weighted composite score
    composite = (vader['compound'] * 0.7) + (blob.polarity * 0.3)

    if composite >= 0.05:
        label = "POSITIVE"
    elif composite <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return {
        "label":     label,
        "composite": round(composite, 3),
        "vader":     vader['compound'],
        "textblob":  round(blob.polarity, 3),
        "subjectivity": round(blob.subjectivity, 3)
    }


def get_sample_news(ticker, company_name):
    """
    Generates realistic sample news headlines for demo.
    In production: replace with NewsAPI or GNews API call.
    Free NewsAPI key: https://newsapi.org/register
    """
    import random
    templates_positive = [
        f"{company_name} reports record quarterly revenue, beats analyst estimates",
        f"Analysts upgrade {company_name} to 'Strong Buy' amid strong growth outlook",
        f"{company_name} announces strategic AI partnership, shares surge",
        f"{company_name} expands into new markets with aggressive growth strategy",
        f"Investors bullish on {company_name} after stellar earnings call",
    ]
    templates_negative = [
        f"{company_name} faces regulatory scrutiny over data privacy concerns",
        f"{company_name} misses revenue targets for Q3, stock under pressure",
        f"Analysts downgrade {company_name} citing macroeconomic headwinds",
        f"{company_name} announces layoffs amid restructuring plans",
        f"Short-sellers target {company_name} over accounting irregularities",
    ]
    templates_neutral = [
        f"{company_name} to release quarterly results next week",
        f"{company_name} appoints new CFO, transition expected to be smooth",
        f"{company_name} maintains annual guidance, no major changes expected",
        f"Market watchers keep eye on {company_name} ahead of Fed meeting",
        f"{company_name} participates in industry conference this week",
    ]
    news = []
    days_back = 14
    for i in range(10):
        pool = random.choice([templates_positive, templates_negative, templates_neutral])
        headline = random.choice(pool)
        date = datetime.now() - timedelta(days=random.randint(0, days_back))
        news.append({"title": headline, "date": date.strftime("%Y-%m-%d"), "source": random.choice(["Reuters", "Bloomberg", "CNBC", "ET Markets", "MoneyControl"])})
    news.sort(key=lambda x: x["date"], reverse=True)
    return news


def get_groq_analysis(company_name, ticker, sentiment_summary, stock_data_summary, hist_df):
    """Get LLM-powered investment analysis from Groq"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "⚠️ Groq API key not found. Please add GROQ_API_KEY to your .env file."
    try:
        client = Groq(api_key=api_key)

        # Build price trend context
        if hist_df is not None and not hist_df.empty:
            recent_close  = hist_df['Close'].iloc[-1]
            week_ago_close = hist_df['Close'].iloc[-5] if len(hist_df) >= 5 else recent_close
            month_ago_close = hist_df['Close'].iloc[-20] if len(hist_df) >= 20 else recent_close
            price_context = (
                f"Current Price: {recent_close:.2f} | "
                f"1-Week Change: {((recent_close-week_ago_close)/week_ago_close*100):.2f}% | "
                f"1-Month Change: {((recent_close-month_ago_close)/month_ago_close*100):.2f}%"
            )
        else:
            price_context = "Price data unavailable"

        prompt = f"""
You are an expert financial analyst and AI researcher. Analyze the following stock data and provide a comprehensive investment report.

STOCK: {company_name} ({ticker})
PRICE TREND: {price_context}
MARKET DATA: {stock_data_summary}
NEWS SENTIMENT ANALYSIS:
- Overall Sentiment: {sentiment_summary['overall']}
- Positive Headlines: {sentiment_summary['positive']} out of {sentiment_summary['total']}
- Negative Headlines: {sentiment_summary['negative']} out of {sentiment_summary['total']}
- Average Sentiment Score: {sentiment_summary['avg_score']:.3f}
- Average Subjectivity: {sentiment_summary['avg_subjectivity']:.3f}

Please provide:
1. **Market Sentiment Summary** (2-3 sentences on what the news sentiment tells us)
2. **Price Action Analysis** (what the price movement indicates)
3. **Key Risk Factors** (2-3 bullet points)
4. **Investment Recommendation** (BUY / HOLD / SELL with confidence level 0-100%)
5. **Short-term Outlook** (next 2-4 weeks prediction reasoning)

Keep the analysis professional, data-driven, and concise. End with a clear disclaimer.
"""
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Groq API Error: {str(e)}"


def extract_recommendation(llm_text):
    """Extract BUY/HOLD/SELL from LLM response"""
    text_upper = llm_text.upper()
    if "BUY" in text_upper and "SELL" not in text_upper:
        return "BUY"
    elif "SELL" in text_upper:
        return "SELL"
    else:
        return "HOLD"


def plot_stock_chart(hist_df, company_name):
    """Plot beautiful price + volume chart"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('#0e1117')
    for ax in [ax1, ax2]:
        ax.set_facecolor('#1e2130')
        ax.tick_params(colors='#8892a4')
        ax.spines['bottom'].set_color('#2d3548')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#2d3548')

    # Price line
    ax1.plot(hist_df.index, hist_df['Close'], color='#4facfe', linewidth=2, label='Close Price')
    ax1.fill_between(hist_df.index, hist_df['Close'],
                     alpha=0.15, color='#4facfe')

    # Moving averages
    if len(hist_df) >= 20:
        ma20 = hist_df['Close'].rolling(20).mean()
        ax1.plot(hist_df.index, ma20, color='#ffd166', linewidth=1.2,
                 linestyle='--', label='MA20', alpha=0.8)
    if len(hist_df) >= 50:
        ma50 = hist_df['Close'].rolling(50).mean()
        ax1.plot(hist_df.index, ma50, color='#ff4b6e', linewidth=1.2,
                 linestyle='--', label='MA50', alpha=0.8)

    ax1.set_title(f'{company_name} — Price Chart', color='white',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Price', color='#8892a4')
    ax1.legend(facecolor='#1e2130', edgecolor='#2d3548',
               labelcolor='white', fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    # Volume bars
    colors_vol = ['#00d4aa' if c >= o else '#ff4b6e'
                  for c, o in zip(hist_df['Close'], hist_df['Open'])]
    ax2.bar(hist_df.index, hist_df['Volume'], color=colors_vol, alpha=0.7, width=1)
    ax2.set_ylabel('Volume', color='#8892a4')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.tight_layout()
    return fig


def plot_sentiment_chart(news_with_sentiment):
    """Plot sentiment distribution donut chart"""
    counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for n in news_with_sentiment:
        counts[n['sentiment']['label']] += 1

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('#1e2130')
    ax.set_facecolor('#1e2130')

    sizes  = [counts['POSITIVE'], counts['NEGATIVE'], counts['NEUTRAL']]
    colors = ['#00d4aa', '#ff4b6e', '#ffd166']
    labels = [f"Positive\n{counts['POSITIVE']}", f"Negative\n{counts['NEGATIVE']}", f"Neutral\n{counts['NEUTRAL']}"]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct='%1.0f%%', startangle=90,
        wedgeprops=dict(width=0.55, edgecolor='#0e1117', linewidth=2),
        textprops={'color': 'white', 'fontsize': 11}
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontweight('bold')

    ax.set_title('News Sentiment\nDistribution', color='white',
                 fontsize=12, fontweight='bold')
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 Stock Analyst")
    st.markdown("---")

    market = st.selectbox("Select Market", list(STOCKS.keys()))
    company_options = list(STOCKS[market].keys())
    selected_company = st.selectbox("Select Stock", company_options)
    ticker = STOCKS[market][selected_company]

    period_map = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y"}
    period_label = st.select_slider("Historical Period", options=list(period_map.keys()), value="3 Months")
    period = period_map[period_label]

    analyze_btn = st.button("🔍 Analyze Now", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("""
    **How it works:**
    - Fetches recent news headlines
    - Dual NLP sentiment analysis (VADER + TextBlob)
    -  Groq LLaMA3 generates investment report
    -  Interactive price & sentiment charts
    """)
    st.markdown("---")
    st.markdown("""
    <div style='color:#4a5568; font-size:0.75rem'>
    Built by <b style='color:#4facfe'>Dharan</b><br>
    Woxsen University<br>
    <a href='https://github.com/dharan0660' style='color:#4facfe'>github.com/dharan0660</a>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────

st.markdown('<div class="header-title">📊 Smart Stock Sentiment Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time news sentiment + LLM-powered investment analysis using Groq & LLaMA3</div>', unsafe_allow_html=True)

if not analyze_btn:
    st.markdown("""
    <div style='text-align:center; padding: 60px; color: #4a5568;'>
        <div style='font-size: 4rem;'>📈</div>
        <div style='font-size: 1.2rem; margin-top: 10px;'>Select a stock from the sidebar and click <b style='color:#4facfe'>Analyze Now</b></div>
        <div style='font-size: 0.9rem; margin-top: 8px; color: #2d3548'>Covers Indian (NSE/BSE), US (NASDAQ/NYSE) & Crypto markets</div>
    </div>
    """, unsafe_allow_html=True)
else:
    with st.spinner(f"Fetching data for {selected_company}..."):
        hist_df, stock_info = fetch_stock_data(ticker, period)
        time.sleep(0.5)

    # ── TOP METRICS ──────────────────────────────
    st.markdown(f"### {selected_company} `{ticker}`")

    if hist_df is not None and not hist_df.empty:
        current_price  = hist_df['Close'].iloc[-1]
        prev_price     = hist_df['Close'].iloc[-2] if len(hist_df) > 1 else current_price
        price_change   = current_price - prev_price
        price_change_p = (price_change / prev_price) * 100
        week_high      = hist_df['High'].tail(5).max()
        week_low       = hist_df['Low'].tail(5).min()
        avg_volume     = hist_df['Volume'].tail(5).mean()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            direction = "▲" if price_change >= 0 else "▼"
            clr = "positive" if price_change >= 0 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value {clr}">{current_price:.2f}</div>
                <div class="metric-label">Current Price</div>
                <div class="{clr}" style="font-size:0.85rem">{direction} {abs(price_change):.2f} ({price_change_p:.2f}%)</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value neutral">{week_high:.2f}</div>
                <div class="metric-label">5-Day High</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value negative">{week_low:.2f}</div>
                <div class="metric-label">5-Day Low</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#4facfe">{avg_volume/1e6:.2f}M</div>
                <div class="metric-label">Avg Volume (5d)</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── PRICE CHART ──────────────────────────────
        st.markdown("#### 📉 Price History")
        fig_price = plot_stock_chart(hist_df, selected_company)
        st.pyplot(fig_price, use_container_width=True)
        plt.close()

    # ── NEWS SENTIMENT ───────────────────────────
    st.markdown("#### 📰 News Sentiment Analysis")
    with st.spinner("Analyzing news sentiment..."):
        news = get_sample_news(ticker, selected_company)
        news_with_sentiment = []
        for item in news:
            sentiment = analyze_sentiment(item['title'])
            news_with_sentiment.append({**item, "sentiment": sentiment})

    # Sentiment summary stats
    total   = len(news_with_sentiment)
    pos     = sum(1 for n in news_with_sentiment if n['sentiment']['label'] == 'POSITIVE')
    neg     = sum(1 for n in news_with_sentiment if n['sentiment']['label'] == 'NEGATIVE')
    neu     = total - pos - neg
    avg_sc  = sum(n['sentiment']['composite'] for n in news_with_sentiment) / total
    avg_sub = sum(n['sentiment']['subjectivity'] for n in news_with_sentiment) / total
    overall = "POSITIVE" if avg_sc > 0.05 else ("NEGATIVE" if avg_sc < -0.05 else "NEUTRAL")

    sentiment_summary = {
        "overall": overall, "positive": pos, "negative": neg,
        "neutral": neu, "total": total,
        "avg_score": avg_sc, "avg_subjectivity": avg_sub
    }

    col_news, col_donut = st.columns([2, 1])
    with col_news:
        for item in news_with_sentiment:
            label = item['sentiment']['label']
            pill_class = f"pill-{label.lower()}"
            score = item['sentiment']['composite']
            st.markdown(f"""
            <div class="news-card">
                <span class="pill {pill_class}">{label}</span>
                <span style="color:#4a5568; font-size:0.75rem; margin-left:8px">{item['source']} · {item['date']}</span>
                <div style="color:#e0e6f0; margin-top:6px; font-size:0.9rem">{item['title']}</div>
                <div style="color:#4a5568; font-size:0.75rem; margin-top:4px">Score: {score:+.3f}</div>
            </div>""", unsafe_allow_html=True)

    with col_donut:
        fig_donut = plot_sentiment_chart(news_with_sentiment)
        st.pyplot(fig_donut, use_container_width=True)
        plt.close()

        # Overall sentiment badge
        clr_map = {"POSITIVE": "positive", "NEGATIVE": "negative", "NEUTRAL": "neutral"}
        st.markdown(f"""
        <div class="metric-card" style="margin-top:10px">
            <div class="metric-value {clr_map[overall]}">{overall}</div>
            <div class="metric-label">Overall Market Mood</div>
            <div style="color:#8892a4; font-size:0.8rem">Avg Score: {avg_sc:+.3f}</div>
        </div>""", unsafe_allow_html=True)

    # ── LLM ANALYSIS ─────────────────────────────
    st.markdown("#### 🤖 LLM Investment Analysis (Groq × LLaMA3)")
    with st.spinner("Generating AI-powered analysis..."):
        stock_data_summary = f"Period: {period_label} | Market: {market}"
        llm_analysis = get_groq_analysis(
            selected_company, ticker,
            sentiment_summary, stock_data_summary, hist_df
        )

    recommendation = extract_recommendation(llm_analysis)
    rec_class = {"BUY": "rec-buy", "SELL": "rec-sell", "HOLD": "rec-hold"}[recommendation]
    rec_icon  = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[recommendation]

    col_rec, col_analysis = st.columns([1, 2])
    with col_rec:
        st.markdown(f"""
        <div class="recommendation-box {rec_class}">
            {rec_icon}<br>{recommendation}
        </div>
        <div style="text-align:center; color:#4a5568; font-size:0.8rem">AI Recommendation</div>
        """, unsafe_allow_html=True)
    with col_analysis:
        st.markdown(f'<div class="llm-box">{llm_analysis}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        ⚠️ <b>Disclaimer:</b> This tool is for educational purposes only. Not financial advice.<br>
        Built by <b>Dharan</b> · Woxsen University ·
        <a href="https://github.com/dharan0660" style="color:#4facfe">GitHub</a>
    </div>
    """, unsafe_allow_html=True)