# 📊 Smart Stock Sentiment Analyst

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-LLaMA3-orange?style=for-the-badge)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> A real-time stock market intelligence tool combining **traditional NLP sentiment analysis** with **LLM-powered investment reasoning** using Groq's LLaMA3 model.

---

## 🚀 Live Demo

> Coming soon — deploying on Streamlit Cloud

---

## ✨ Features

- 📰 **Real-time News Sentiment** — Dual-engine analysis using VADER + TextBlob
- 📈 **Interactive Price Charts** — Historical price, moving averages (MA20, MA50), volume
- 🤖 **LLM Investment Report** — Groq × LLaMA3 generates professional buy/hold/sell analysis
- 🌏 **Multi-Market Coverage** — Indian (NSE/BSE), US (NASDAQ/NYSE), and Crypto
- 🎨 **Beautiful Dark UI** — Professional Streamlit interface with custom CSS

---

## 🧠 How It Works

```
News Headlines
      ↓
VADER + TextBlob (Dual NLP Sentiment)
      ↓
Composite Sentiment Score
      ↓
Groq LLaMA3 API (Investment Reasoning)
      ↓
BUY / HOLD / SELL Recommendation + Report
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Frontend | Streamlit |
| NLP | VADER Sentiment, TextBlob, NLTK |
| LLM | Groq API (LLaMA3-8B) |
| Market Data | yfinance |
| Visualization | Matplotlib, Seaborn |
| Environment | python-dotenv |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/dharan0660/smart-stock-sentiment.git
cd smart-stock-sentiment
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
```bash
# Copy the example env file
cp .env.example .env

# Open .env and paste your Groq API key
# Get your FREE key at: https://console.groq.com
```

Edit `.env`:
```
GROQ_API_KEY=your_actual_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` 🎉

---

## 📸 Screenshots

> Add screenshots of your running app here after deployment

---

## 📁 Project Structure

```
smart-stock-sentiment/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

---

## 🔮 Future Improvements

- [ ] Integrate live NewsAPI for real headlines
- [ ] Add portfolio tracker with multi-stock comparison
- [ ] Add technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Export PDF reports
- [ ] Email alerts for sentiment shifts

---

## ⚠️ Disclaimer

This tool is built for **educational and research purposes only**. Nothing here constitutes financial advice. Always consult a certified financial advisor before making investment decisions.

---

## 👨‍💻 Author

**Dharan**
- 🎓 Woxsen University
- 💻 GitHub: [@dharan0660](https://github.com/dharan0660)

---

## 📄 License

This project is licensed under the MIT License.