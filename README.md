# 💸 SpendSense — AI Financial Coach & Budget Planner for College Students

> **A functional, grounded AI budgeting assistant built for college students with tight budgets and fluctuating expenses.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Model: Open--Weight](https://img.shields.io/badge/Model-Llama--3.1--8B%20%7C%20Mistral--7B-7C3AED.svg)](https://groq.com)

---

## 📌 Problem & Purpose

College students face erratic expenses: textbooks one week, split ride-shares, campus food, boba runs, and rent. Most financial tools either:
1. Demand intrusive bank credentials (Plaid/Yodlee) that students are hesitant to link.
2. Require tedious manual accounting with complex spreadsheets.
3. Use ungrounded LLMs that hallucinate math and fabricate totals.

**SpendSense** solves this with an intuitive, student-friendly application:
- **Zero Bank Credentials Required:** Simply upload any bank/credit card CSV export or use the 1-click college sample data.
- **Deterministic Financial Engine:** **Python executes 100% of all calculations.** The AI model is never asked to invent numbers.
- **Open-Weight AI Agent (≤ 10B parameters):** Powered by hosted open-weight models such as **Llama-3.1-8B-Instant** or **Llama-3.2-3B-Instruct** via Groq or Hugging Face.
- **Actionable Student Advice:** Understands college spending patterns (e.g., student discounts, textbook rentals, meal prepping, subscription sharing).
- **"What If?" Interactive Simulator:** Shows the exact, real-time math of cutting back on specific categories (e.g., "What if I cut dining out by 30%?").

---

## 🚀 Key Features

1. **Flexible Expense Ingestion**:
   - **1-Click College Sample Dataset:** 25 realistic college student expenses across groceries, dining, boba, tuition supplies, and rent.
   - **Robust CSV Upload:** Flexibly maps column variations (`date`, `amount`, `cost`, `category`, `merchant`, etc.) and strips messy currency symbols (`$`, `,`).
   - **Manual Transaction Entry:** Instant form to log cash, split bills, or spot purchases.
2. **Pure Python Financial Calculations**:
   - Total spending and transaction count.
   - Category breakdowns with percentages and average spend per purchase.
   - Daily average burn rate and projected 30-day month-end spending.
   - Budget remaining and over-budget warnings.
   - Savings goal deficit/surplus tracking.
   - Identification of top spending drivers and largest single expenses.
3. **Hosted Open-Weight AI Advisor (≤ 10B Parameters)**:
   - Evaluates spending patterns against student-specific financial realities.
   - Grounded strictly in pre-computed Python facts to eliminate hallucinations.
   - Interactive Q&A chat: Ask *"Can I afford a $50 dinner?"* or *"Where am I leaking money?"*
   - **Zero-Setup Demo Fallback:** Operates out of the box with an intelligent deterministic fallback engine if no API key is supplied!
4. **Interactive "What If?" Scenario Simulator**:
   - Adjust spending limits or percentage cuts on any category.
   - Instant calculation of dollar savings, new total spend, new remaining budget, and savings goal progress.
   - Visual comparison chart showing "Before" vs "After".
5. **Visual Student Dashboard**:
   - Category distribution donut chart (Plotly).
   - Cumulative spend vs budget cap timeline.
   - Transaction explorer with category filtering, merchant search, and row deletion.
   - Cleaned CSV export button.

---

## 🛠️ Tech Stack

- **Language:** Python 3.9+
- **Frontend / UI:** Streamlit
- **Data Processing:** Pandas
- **Visualizations:** Plotly Express & Plotly Graph Objects
- **AI Models (≤ 10B params):**
  - Meta Llama 3.1 8B Instant (`llama-3.1-8b-instant`) via Groq
  - Meta Llama 3.2 3B Instruct (`meta-llama/Llama-3.2-3B-Instruct`) via Hugging Face Inference API
- **Environment Management:** python-dotenv
- **Testing:** Pytest

---

## 📐 Architecture & Grounding Philosophy

```
  ┌─────────────────────────────────────────────────────────────┐
  │                    User Input & Data                        │
  │   - CSV Upload / Manual Transaction / Demo Preset (25 txns) │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │         Python Deterministic Engine (utils.py)             │
  │   - Data Cleaning & Normalization                           │
  │   - Mathematical Calculations (Total, Avg, Pace, Budget)   │
  │   - What-If Scenario Simulations                            │
  └──────────────┬──────────────────────────────┬───────────────┘
                 │                              │
                 │ Computed Facts               │ Pure Numbers
                 ▼                              ▼
  ┌──────────────────────────────┐ ┌────────────────────────────┐
  │  Grounded AI Prompt Builder  │ │  Interactive Visualizations│
  │  - Injects STRICT facts only │ │  - Plotly Donut & Timeline │
  │  - Hosted Open-Weight LLM    │ │  - KPI Metrics Cards       │
  │    (Llama-3.1-8B on Groq)    │ │  - What-If Comparisons     │
  └──────────────┬───────────────┘ └────────────┬───────────────┘
                 │                              │
                 └───────────────┬──────────────┘
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 Streamlit Web Application                   │
  └─────────────────────────────────────────────────────────────┘
```

> **Guaranteed Mathematical Truth:** LLMs are known to hallucinate arithmetic. In SpendSense, the model is strictly passed pre-calculated metrics in its system context and instructed to provide qualitative insights and advice based on those facts.

---

## ⚡ Quickstart Guide

### 1. Clone & Set Up Environment

```bash
# Navigate to the project directory
cd SpendSense

# Create a virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional for Live LLM)

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` to include your free Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
DEFAULT_MONTHLY_BUDGET=1200.00
DEFAULT_SAVINGS_GOAL=150.00
```

> **Note:** A Groq API key is free and takes 30 seconds to obtain at [console.groq.com](https://console.groq.com). If you run without a key, SpendSense automatically runs in **Demo Mode**, using its deterministic rule-based advice engine with 100% of UI features intact.

### 3. Run the Application

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the test suite to verify calculations, CSV parsing edge cases, and simulation math:

```bash
pytest tests/ -v
```

---

## 📂 Project Structure

```
SpendSense/
├── app.py                   # Streamlit web application & UI tabs
├── utils.py                 # Deterministic financial calculations & AI engine
├── sample_expenses.csv      # 25 realistic college student demo expenses
├── requirements.txt         # Project dependencies
├── .env.example             # Example environment configuration
├── .gitignore               # Git ignore rules
├── LICENSE                  # MIT License
├── README.md                # Documentation & architecture guide
└── tests/
    └── test_calculations.py # Automated test suite for calculations & parsing
```

---

## ⚖️ License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
