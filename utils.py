"""
SpendSense Financial Analysis & AI Engine
Provides deterministic calculation functions, CSV parsing & validation,
what-if scenario simulation, and grounded AI insights using open-weight models.
"""

import os
import json
import re
from datetime import datetime, date
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import requests

# Predefined College Student Expense Categories
COLLEGE_CATEGORIES = [
    "Groceries",
    "Dining & Fast Food",
    "Coffee & Boba",
    "Housing & Utilities",
    "Education & Supplies",
    "Subscriptions",
    "Transportation",
    "Entertainment & Social",
    "Personal & Health",
    "Other"
]

def normalize_column_name(col: str) -> str:
    """Normalize column header to lowercase alphanumeric without punctuation."""
    return re.sub(r'[^a-z0-9]', '', str(col).strip().lower())

def validate_and_process_csv(file_or_path: Any) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Validates and standardizes an uploaded CSV file or path into a clean DataFrame.
    Expected output columns: ['date', 'amount', 'category', 'description'].
    Returns: (cleaned_df, warnings_list)
    """
    warnings = []
    df = None
    try:
        df = pd.read_csv(file_or_path)
    except Exception as initial_err:
        try:
            if hasattr(file_or_path, 'seek'):
                file_or_path.seek(0)
            df = pd.read_csv(file_or_path, on_bad_lines='skip')
            warnings.append("Note: Some malformed lines in the CSV were skipped automatically.")
        except Exception as retry_err:
            return None, [f"Failed to read CSV file: {str(initial_err)}"]

    if df.empty:
        return None, ["The uploaded CSV file is completely empty."]

    # Map column headers flexibly
    col_map = {}
    normalized_cols = {col: normalize_column_name(col) for col in df.columns}

    # Date column mapping candidates
    date_candidates = ["date", "transactiondate", "transdate", "timestamp", "time", "day", "posteddate"]
    amount_candidates = ["amount", "cost", "price", "total", "spent", "value", "debit", "transactionamount"]
    category_candidates = ["category", "cat", "type", "tag", "group", "classification", "department"]
    desc_candidates = ["description", "desc", "merchant", "vendor", "name", "item", "details", "title", "payee", "memo"]

    for orig_col, norm in normalized_cols.items():
        if norm in date_candidates and 'date' not in col_map:
            col_map['date'] = orig_col
        elif norm in amount_candidates and 'amount' not in col_map:
            col_map['amount'] = orig_col
        elif norm in category_candidates and 'category' not in col_map:
            col_map['category'] = orig_col
        elif norm in desc_candidates and 'description' not in col_map:
            col_map['description'] = orig_col

    # Check required columns
    missing_required = []
    if 'date' not in col_map:
        missing_required.append("date (e.g. 'date', 'transaction_date')")
    if 'amount' not in col_map:
        missing_required.append("amount (e.g. 'amount', 'cost', 'price')")

    if missing_required:
        return None, [f"Missing required column(s): {', '.join(missing_required)}. Available columns: {list(df.columns)}"]

    # Build standardized DataFrame
    cleaned = pd.DataFrame()
    cleaned['date'] = df[col_map['date']]
    cleaned['amount'] = df[col_map['amount']]

    if 'category' in col_map:
        cleaned['category'] = df[col_map['category']]
    else:
        cleaned['category'] = "Other"
        warnings.append("No 'category' column detected; assigned default category 'Other'.")

    if 'description' in col_map:
        cleaned['description'] = df[col_map['description']]
    else:
        cleaned['description'] = "General Expense"
        warnings.append("No 'description' column detected; assigned default description 'General Expense'.")

    initial_row_count = len(cleaned)

    # Clean amount: strip currency symbols, commas, whitespace, convert to float
    def parse_amount(val):
        if pd.isna(val):
            return None
        if isinstance(val, (int, float)):
            return abs(float(val))
        val_str = str(val).strip()
        # Remove $, commas, quotes
        val_clean = re.sub(r'[^\d.-]', '', val_str)
        try:
            parsed = float(val_clean)
            return abs(parsed)
        except (ValueError, TypeError):
            return None

    cleaned['amount'] = cleaned['amount'].apply(parse_amount)
    invalid_amounts = cleaned['amount'].isna() | (cleaned['amount'] == 0)
    if invalid_amounts.sum() > 0:
        warnings.append(f"Dropped {invalid_amounts.sum()} row(s) with missing, zero, or unparseable amounts.")
        cleaned = cleaned[~invalid_amounts].copy()

    # Clean date: parse to datetime, drop unparseable
    cleaned['parsed_date'] = pd.to_datetime(cleaned['date'], errors='coerce')
    invalid_dates = cleaned['parsed_date'].isna()
    if invalid_dates.sum() > 0:
        warnings.append(f"Dropped {invalid_dates.sum()} row(s) with invalid date formats.")
        cleaned = cleaned[~invalid_dates].copy()

    if cleaned.empty:
        return None, ["All rows were dropped due to invalid date or amount values."]

    # Standardize types and strings
    cleaned['date'] = cleaned['parsed_date'].dt.strftime('%Y-%m-%d')
    cleaned.drop(columns=['parsed_date'], inplace=True)

    cleaned['category'] = cleaned['category'].fillna("Other").astype(str).str.strip()
    cleaned['category'] = cleaned['category'].apply(lambda x: x if x else "Other")

    cleaned['description'] = cleaned['description'].fillna("Expense").astype(str).str.strip()
    cleaned['description'] = cleaned['description'].apply(lambda x: x if x else "Expense")

    # Round amount to 2 decimal places
    cleaned['amount'] = cleaned['amount'].round(2)

    # Sort chronologically
    cleaned.sort_values(by=['date', 'amount'], ascending=[True, False], inplace=True)
    cleaned.reset_index(drop=True, inplace=True)

    rows_kept = len(cleaned)
    if rows_kept < initial_row_count:
        warnings.append(f"Kept {rows_kept} of {initial_row_count} total rows.")

    return cleaned, warnings


def add_manual_transaction(df: pd.DataFrame, date_val: str, amount_val: float, category_val: str, desc_val: str) -> pd.DataFrame:
    """Appends a new transaction to the DataFrame and sorts it."""
    new_row = {
        'date': str(date_val),
        'amount': round(abs(float(amount_val)), 2),
        'category': str(category_val).strip() if str(category_val).strip() else "Other",
        'description': str(desc_val).strip() if str(desc_val).strip() else "Manual Expense"
    }
    updated_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    updated_df.sort_values(by=['date', 'amount'], ascending=[True, False], inplace=True)
    updated_df.reset_index(drop=True, inplace=True)
    return updated_df


def delete_transaction(df: pd.DataFrame, index_to_delete: int) -> pd.DataFrame:
    """Deletes a transaction row by index."""
    if 0 <= index_to_delete < len(df):
        updated_df = df.drop(index_to_delete).reset_index(drop=True)
        return updated_df
    return df


def calculate_financial_metrics(
    df: pd.DataFrame,
    monthly_budget: float = 1200.0,
    savings_goal: float = 150.0
) -> Dict[str, Any]:
    """
    Performs purely deterministic Python financial calculations.
    Returns comprehensive metrics for dashboard display, What-If simulation,
    and grounding AI prompts.
    """
    if df is None or df.empty:
        return {
            "total_spending": 0.0,
            "transaction_count": 0,
            "category_breakdown": [],
            "category_totals": {},
            "date_range": {"start": None, "end": None, "days_span": 0},
            "daily_average": 0.0,
            "projected_monthly_spend": 0.0,
            "monthly_budget": float(monthly_budget),
            "remaining_budget": float(monthly_budget),
            "budget_utilization_pct": 0.0,
            "budget_status": "No Expenses",
            "savings_goal": float(savings_goal),
            "current_savings_potential": float(monthly_budget),
            "savings_goal_met": monthly_budget >= savings_goal,
            "savings_deficit_or_surplus": float(monthly_budget - savings_goal),
            "top_categories": [],
            "top_transactions": []
        }

    total_spending = round(float(df['amount'].sum()), 2)
    transaction_count = int(len(df))

    # Date range & daily average calculation
    dates = pd.to_datetime(df['date'])
    min_date = dates.min()
    max_date = dates.max()
    days_span = max((max_date - min_date).days + 1, 1)

    daily_average = round(total_spending / days_span, 2)
    # 30-day projected spending based on current daily pace
    projected_monthly_spend = round(daily_average * 30, 2)

    # Category breakdown
    cat_group = df.groupby('category')['amount'].agg(['sum', 'count']).reset_index()
    cat_group['percentage'] = ((cat_group['sum'] / total_spending) * 100).round(1)
    cat_group['avg_spend'] = (cat_group['sum'] / cat_group['count']).round(2)
    cat_group.sort_values(by='sum', ascending=False, inplace=True)

    category_breakdown = []
    category_totals = {}
    for _, row in cat_group.iterrows():
        cat_name = str(row['category'])
        cat_total = round(float(row['sum']), 2)
        category_totals[cat_name] = cat_total
        category_breakdown.append({
            "category": cat_name,
            "total": cat_total,
            "count": int(row['count']),
            "percentage": float(row['percentage']),
            "avg_spend": float(row['avg_spend'])
        })

    # Budget Tracking
    monthly_budget = float(monthly_budget)
    savings_goal = float(savings_goal)
    remaining_budget = round(monthly_budget - total_spending, 2)
    budget_utilization_pct = round((total_spending / monthly_budget) * 100, 1) if monthly_budget > 0 else 0.0

    if remaining_budget < 0:
        budget_status = "Over Budget"
    elif budget_utilization_pct >= 85.0:
        budget_status = "Caution (Near Limit)"
    else:
        budget_status = "Under Budget"

    # Savings Goal Tracking
    # Current net savings potential is what's left of the budget after actual spending
    current_savings_potential = round(remaining_budget, 2)
    savings_deficit_or_surplus = round(current_savings_potential - savings_goal, 2)
    savings_goal_met = current_savings_potential >= savings_goal

    # Top categories & top individual transactions
    top_categories = category_breakdown[:3]
    top_transactions = (
        df.sort_values(by='amount', ascending=False)
        .head(3)[['date', 'amount', 'category', 'description']]
        .to_dict(orient='records')
    )

    return {
        "total_spending": total_spending,
        "transaction_count": transaction_count,
        "category_breakdown": category_breakdown,
        "category_totals": category_totals,
        "date_range": {
            "start": min_date.strftime('%Y-%m-%d'),
            "end": max_date.strftime('%Y-%m-%d'),
            "days_span": int(days_span)
        },
        "daily_average": daily_average,
        "projected_monthly_spend": projected_monthly_spend,
        "monthly_budget": monthly_budget,
        "remaining_budget": remaining_budget,
        "budget_utilization_pct": budget_utilization_pct,
        "budget_status": budget_status,
        "savings_goal": savings_goal,
        "current_savings_potential": current_savings_potential,
        "savings_goal_met": savings_goal_met,
        "savings_deficit_or_surplus": savings_deficit_or_surplus,
        "top_categories": top_categories,
        "top_transactions": top_transactions
    }


def simulate_what_if(
    metrics: Dict[str, Any],
    target_category: str,
    cut_percentage: float = 0.0,
    target_cap: Optional[float] = None
) -> Dict[str, Any]:
    """
    Performs a deterministic 'What If?' scenario simulation.
    Calculates the exact effect of adjusting a category's spending on:
    - Dollar savings
    - New total spend
    - New remaining budget
    - Progress toward savings goal
    """
    category_totals = metrics.get("category_totals", {})
    baseline_cat_spend = float(category_totals.get(target_category, 0.0))
    baseline_total = float(metrics.get("total_spending", 0.0))
    monthly_budget = float(metrics.get("monthly_budget", 1200.0))
    savings_goal = float(metrics.get("savings_goal", 150.0))
    baseline_remaining = float(metrics.get("remaining_budget", monthly_budget - baseline_total))

    if baseline_cat_spend <= 0:
        return {
            "target_category": target_category,
            "baseline_cat_spend": 0.0,
            "new_cat_spend": 0.0,
            "dollar_savings": 0.0,
            "new_total_spend": baseline_total,
            "baseline_remaining": baseline_remaining,
            "new_remaining": baseline_remaining,
            "new_budget_utilization": metrics.get("budget_utilization_pct", 0.0),
            "savings_goal_achieved": baseline_remaining >= savings_goal,
            "new_savings_gap": max(0.0, savings_goal - baseline_remaining),
            "summary_text": f"No spending recorded for '{target_category}' to simulate cuts on."
        }

    # Calculate new category spend
    if target_cap is not None and target_cap >= 0:
        new_cat_spend = min(baseline_cat_spend, float(target_cap))
        dollar_savings = baseline_cat_spend - new_cat_spend
        adjustment_desc = f"capping '{target_category}' at ${target_cap:,.2f}"
    else:
        cut_fraction = min(max(float(cut_percentage) / 100.0, 0.0), 1.0)
        dollar_savings = baseline_cat_spend * cut_fraction
        new_cat_spend = baseline_cat_spend - dollar_savings
        adjustment_desc = f"cutting '{target_category}' by {cut_percentage:.0f}%"

    dollar_savings = round(dollar_savings, 2)
    new_cat_spend = round(baseline_cat_spend - dollar_savings, 2)
    new_total_spend = round(baseline_total - dollar_savings, 2)
    new_remaining = round(monthly_budget - new_total_spend, 2)
    new_budget_utilization = round((new_total_spend / monthly_budget) * 100, 1) if monthly_budget > 0 else 0.0

    savings_goal_achieved = new_remaining >= savings_goal
    new_savings_gap = round(max(0.0, savings_goal - new_remaining), 2)
    pct_goal_covered = round((new_remaining / savings_goal) * 100, 1) if savings_goal > 0 else 100.0

    summary_text = (
        f"By {adjustment_desc}, you save ${dollar_savings:,.2f}. "
        f"Your total spending drops from ${baseline_total:,.2f} to ${new_total_spend:,.2f}, "
        f"leaving you with ${new_remaining:,.2f} remaining budget (covering {pct_goal_covered}% of your ${savings_goal:,.2f} savings goal)."
    )

    return {
        "target_category": target_category,
        "baseline_cat_spend": baseline_cat_spend,
        "new_cat_spend": new_cat_spend,
        "dollar_savings": dollar_savings,
        "new_total_spend": new_total_spend,
        "baseline_remaining": baseline_remaining,
        "new_remaining": new_remaining,
        "new_budget_utilization": new_budget_utilization,
        "savings_goal_achieved": savings_goal_achieved,
        "new_savings_gap": new_savings_gap,
        "pct_goal_covered": pct_goal_covered,
        "summary_text": summary_text
    }


def build_ai_grounding_context(metrics: Dict[str, Any]) -> str:
    """
    Constructs a grounded, factual financial summary for the open-weight LLM prompt.
    Enforces strict truthfulness by providing pre-calculated figures.
    """
    lines = [
        "### CALCULATED FINANCIAL SUMMARY (STRICT GROUND TRUTH - DO NOT INVENT NUMBERS)",
        f"- Monthly Budget Set: ${metrics.get('monthly_budget', 0.0):,.2f}",
        f"- Monthly Savings Goal: ${metrics.get('savings_goal', 0.0):,.2f}",
        f"- Total Actual Spending Recorded: ${metrics.get('total_spending', 0.0):,.2f}",
        f"- Number of Transactions: {metrics.get('transaction_count', 0)}",
        f"- Remaining Budget: ${metrics.get('remaining_budget', 0.0):,.2f}",
        f"- Budget Status: {metrics.get('budget_status', 'N/A')} ({metrics.get('budget_utilization_pct', 0.0)}% utilized)",
        f"- Date Span of Records: {metrics.get('date_range', {}).get('days_span', 0)} days ({metrics.get('date_range', {}).get('start')} to {metrics.get('date_range', {}).get('end')})",
        f"- Daily Average Spend: ${metrics.get('daily_average', 0.0):,.2f}/day",
        f"- 30-Day Projected Spend at Current Pace: ${metrics.get('projected_monthly_spend', 0.0):,.2f}",
        f"- Current Potential Savings: ${metrics.get('current_savings_potential', 0.0):,.2f} " +
        (f"(Surplus of ${metrics.get('savings_deficit_or_surplus', 0.0):,.2f} above goal)" if metrics.get('savings_goal_met') else f"(Deficit of ${abs(metrics.get('savings_deficit_or_surplus', 0.0)):,.2f} below goal)")
    ]

    lines.append("\n### Category Breakdown (Actual Calculated Amounts):")
    for cat in metrics.get('category_breakdown', []):
        lines.append(f"  * {cat['category']}: ${cat['total']:,.2f} ({cat['percentage']}% of spend, {cat['count']} transactions, avg ${cat['avg_spend']:,.2f}/txn)")

    lines.append("\n### Top 3 Largest Single Expenses:")
    for txn in metrics.get('top_transactions', []):
        lines.append(f"  * {txn.get('date')}: ${txn.get('amount'):,.2f} - {txn.get('category')} ({txn.get('description')})")

    return "\n".join(lines)


def generate_rule_based_insights(metrics: Dict[str, Any]) -> str:
    """
    Deterministic student financial insights generator.
    Serves as an instant, zero-latency fallback when an external API key is not configured,
    ensuring the hackathon MVP works 100% out of the box with zero setup.
    """
    total = metrics.get('total_spending', 0.0)
    budget = metrics.get('monthly_budget', 1200.0)
    remaining = metrics.get('remaining_budget', 0.0)
    savings_goal = metrics.get('savings_goal', 150.0)
    daily_avg = metrics.get('daily_average', 0.0)
    projected = metrics.get('projected_monthly_spend', 0.0)
    top_cats = metrics.get('top_categories', [])
    top_txns = metrics.get('top_transactions', [])

    lines = []
    lines.append("### 🎓 SpendSense Financial Diagnosis & Action Plan")
    lines.append("")

    # Status overview
    if remaining >= savings_goal:
        lines.append(f"✅ **Great Job! You're on Track:** You have **${remaining:,.2f}** remaining in your budget, which easily satisfies your **${savings_goal:,.2f}** savings goal with a **${(remaining - savings_goal):,.2f}** cushion.")
    elif remaining > 0:
        lines.append(f"⚠️ **Budget Tight:** You have **${remaining:,.2f}** left of your **${budget:,.2f}** budget. While you are not in the red, you are currently **${(savings_goal - remaining):,.2f}** short of your **${savings_goal:,.2f}** savings goal.")
    else:
        lines.append(f"🚨 **Budget Alert:** You have overspent by **${abs(remaining):,.2f}** against your **${budget:,.2f}** budget. Immediate adjustments are recommended to prevent debt accumulation.")

    lines.append(f"- **Current Daily Burn Rate:** `${daily_avg:,.2f}/day`. At this rate, your estimated 30-day spending will reach **${projected:,.2f}**.")

    # Category analysis
    lines.append("")
    lines.append("#### 🔍 Where Your Money Is Going:")
    if top_cats:
        for idx, cat in enumerate(top_cats, 1):
            lines.append(f"{idx}. **{cat['category']}**: `${cat['total']:,.2f}` ({cat['percentage']}% of total expenses across {cat['count']} purchases).")

    # College-specific tactical tips based on detected top categories
    lines.append("")
    lines.append("#### 💡 Tactical Student Recommendations:")
    cat_names = [c['category'].lower() for c in top_cats]

    if any("coffee" in c or "boba" in c for c in cat_names):
        lines.append("- **Coffee & Boba Hack:** You have regular beverage spending. Investing in an insulated tumbler and making iced coffee or cold brew at home 3 days a week can easily save you $40–$60 every month.")

    if any("dining" in c or "food" in c for c in cat_names):
        lines.append("- **Campus Dining & Takeout:** Late-night takeout and quick bowls add up rapidly. Try batch-cooking simple staples (burrito bowls, pasta, curry) on Sunday afternoons to replace 2 takeout meals weekly—saving approximately $25–$35/week.")

    if any("groceries" in c for c in cat_names):
        lines.append("- **Grocery Optimization:** Use student discount days (many local grocers offer 5-10% off with university ID on Tuesdays). Stick to store brands and buy frozen fruits/veggies to prevent food waste in dorm mini-fridges.")

    if any("education" in c or "supplies" in c for c in cat_names):
        lines.append("- **Course Materials:** Always check library course reserves, textbook rental portals (Chegg, CampusBookRentals), or student buy/sell groups before paying full retail at the campus bookstore.")

    if any("subscriptions" in c for c in cat_names):
        lines.append("- **Subscription Audit:** Ensure you are enrolled in student discounted rates (Spotify/Hulu Student at $5.99/mo, Amazon Prime Student 6-month trial). Cancel services you haven't opened in the past 14 days.")

    # High-impact transaction callout
    if top_txns:
        biggest = top_txns[0]
        lines.append("")
        lines.append(f"📌 **Largest Expense Callout:** Your single highest expense was **${biggest['amount']:,.2f}** on {biggest['date']} for *{biggest['description']}* ({biggest['category']}). Plan ahead for similar one-time spikes.")

    lines.append("")
    lines.append("*(Note: Calculations performed deterministically by Python. Add your Groq API key in the sidebar to engage with the live Llama 3.1 8B model!)*")

    return "\n".join(lines)


def call_groq_api(
    prompt: str,
    system_prompt: str,
    api_key: str,
    model: str = "llama-3.1-8b-instant"
) -> str:
    """
    Calls Groq API to query open-weight model (Llama-3.1-8b-instant or Llama-3.2-3b-preview).
    Both models are <= 10B parameters and provide rapid inference.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4,
        "max_tokens": 1000
    }
    response = requests.post(url, headers=headers, json=payload, timeout=20)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def call_hf_api(
    prompt: str,
    system_prompt: str,
    api_key: str,
    model: str = "meta-llama/Llama-3.2-3B-Instruct"
) -> str:
    """
    Calls Hugging Face Inference API for open-weight models (Llama-3.2-3B or Mistral-7B).
    """
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json"
    }
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 800,
            "temperature": 0.4,
            "return_full_text": False
        }
    }
    response = requests.post(url, headers=headers, json=payload, timeout=25)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
        return data[0]["generated_text"]
    elif isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    return str(data)


SYSTEM_FINANCIAL_COACH_PROMPT = """You are SpendSense, an encouraging, practical, and highly empathetic AI financial advisor specifically designed for college students.

CRITICAL INSTRUCTIONS:
1. Grounding in Facts: You MUST use ONLY the exact calculated numbers provided in the Ground Truth Context. NEVER invent, hallucinate, or alter financial numbers, totals, or percentages.
2. Distinguish Estimates: Explicitly differentiate historical actual spending from projections, targets, or hypothetical estimates.
3. College-Specific Actionability: College students have constrained income and unique pressures. Provide high-impact, realistic advice (e.g., student discounts, textbook rentals, sharing rides, dining hall hacks, canceling unused subscriptions) rather than generic advice like "stop eating out completely".
4. Tone: Encouraging, supportive, transparent, and non-judgmental. Keep responses clear and well-structured with Markdown headings and bullet points.
"""

def generate_ai_financial_insights(
    metrics: Dict[str, Any],
    groq_api_key: Optional[str] = None,
    hf_api_key: Optional[str] = None,
    groq_model: str = "llama-3.1-8b-instant",
    hf_model: str = "meta-llama/Llama-3.2-3B-Instruct"
) -> Tuple[str, str]:
    """
    Generates spending insights.
    Returns: (insights_text, engine_used_description)
    """
    grounding_context = build_ai_grounding_context(metrics)
    prompt = (
        f"{grounding_context}\n\n"
        "TASK:\n"
        "1. Diagnose the student's financial health this month based on their budget, spending pace, and savings goal.\n"
        "2. Break down their primary spending drivers and identify potential leakages.\n"
        "3. Provide 3-4 concrete, student-tailored actions to protect their remaining budget and reach their savings goal.\n"
        "4. Highlight any large or unusual transactions that require future planning."
    )

    # 1. Try Groq if key provided
    if groq_api_key and groq_api_key.strip():
        try:
            result = call_groq_api(
                prompt=prompt,
                system_prompt=SYSTEM_FINANCIAL_COACH_PROMPT,
                api_key=groq_api_key.strip(),
                model=groq_model
            )
            return result, f"Groq ({groq_model} - <=10B open-weight)"
        except Exception as e:
            fallback = generate_rule_based_insights(metrics)
            return f"*(Note: Groq API call encountered an error: {str(e)}. Displaying grounded deterministic insights:)*\n\n{fallback}", "Deterministic Fallback Engine"

    # 2. Try Hugging Face if key provided
    if hf_api_key and hf_api_key.strip():
        try:
            result = call_hf_api(
                prompt=prompt,
                system_prompt=SYSTEM_FINANCIAL_COACH_PROMPT,
                api_key=hf_api_key.strip(),
                model=hf_model
            )
            return result, f"Hugging Face ({hf_model} - <=10B open-weight)"
        except Exception as e:
            fallback = generate_rule_based_insights(metrics)
            return f"*(Note: Hugging Face API call encountered an error: {str(e)}. Displaying grounded deterministic insights:)*\n\n{fallback}", "Deterministic Fallback Engine"

    # 3. Default to deterministic high-quality rule-based engine
    return generate_rule_based_insights(metrics), "Deterministic Student Insights Engine (Zero Setup / Demo Mode)"


def ask_ai_financial_question(
    user_question: str,
    metrics: Dict[str, Any],
    chat_history: Optional[List[Dict[str, str]]] = None,
    groq_api_key: Optional[str] = None,
    hf_api_key: Optional[str] = None,
    groq_model: str = "llama-3.1-8b-instant"
) -> Tuple[str, str]:
    """
    Answers user queries grounded in their transaction metrics.
    """
    grounding_context = build_ai_grounding_context(metrics)
    prompt = (
        f"{grounding_context}\n\n"
        f"STUDENT QUESTION: {user_question}\n\n"
        "INSTRUCTIONS:\n"
        "- Answer the student's question accurately using ONLY the numbers provided above.\n"
        "- If asked 'can I afford X', compare the cost against their remaining budget and savings goal.\n"
        "- Give a clear, direct conclusion first, then explain the math and reasoning."
    )

    if groq_api_key and groq_api_key.strip():
        try:
            ans = call_groq_api(
                prompt=prompt,
                system_prompt=SYSTEM_FINANCIAL_COACH_PROMPT,
                api_key=groq_api_key.strip(),
                model=groq_model
            )
            return ans, f"Groq ({groq_model})"
        except Exception as e:
            pass

    # Deterministic intelligent fallback Q&A
    q_lower = user_question.lower()
    rem = metrics.get('remaining_budget', 0.0)
    tot = metrics.get('total_spending', 0.0)
    budget = metrics.get('monthly_budget', 1200.0)
    top_cats = metrics.get('top_categories', [])
    top_txns = metrics.get('top_transactions', [])
    goal = metrics.get('savings_goal', 150.0)

    if "afford" in q_lower:
        # Try to find a dollar amount in the question
        found_amounts = re.findall(r'\$?(\d+(?:\.\d{1,2})?)', q_lower)
        item_cost = float(found_amounts[0]) if found_amounts else None
        if item_cost is not None:
            if rem >= item_cost + goal:
                return (
                    f"**Yes, you can afford this!**\n\n"
                    f"- Item cost: **${item_cost:,.2f}**\n"
                    f"- Current remaining budget: **${rem:,.2f}**\n"
                    f"- After this purchase, you will still have **${(rem - item_cost):,.2f}**, which still preserves your **${goal:,.2f}** savings goal with **${(rem - item_cost - goal):,.2f}** to spare.",
                    "Deterministic Advisor"
                )
            elif rem >= item_cost:
                return (
                    f"**Technically yes, but with caution.**\n\n"
                    f"- Item cost: **${item_cost:,.2f}**\n"
                    f"- Current remaining budget: **${rem:,.2f}**\n"
                    f"- You have enough budget, but spending ${item_cost:,.2f} leaves you with **${(rem - item_cost):,.2f}**, which will dip into your **${goal:,.2f}** savings goal by **${(goal - (rem - item_cost)):,.2f}**.",
                    "Deterministic Advisor"
                )
            else:
                return (
                    f"**No, this will exceed your budget.**\n\n"
                    f"- Item cost: **${item_cost:,.2f}**\n"
                    f"- Current remaining budget: **${rem:,.2f}**\n"
                    f"- Buying this would put you **${(item_cost - rem):,.2f}** over your monthly budget of ${budget:,.2f}.",
                    "Deterministic Advisor"
                )
        else:
            return (
                f"You currently have **${rem:,.2f}** remaining in your monthly budget of ${budget:,.2f}. "
                f"To keep your **${goal:,.2f}** savings goal intact, keep any discretionary spending below **${max(0.0, rem - goal):,.2f}**.",
                "Deterministic Advisor"
            )

    elif "leak" in q_lower or "waste" in q_lower or "where" in q_lower:
        if top_cats:
            top_1 = top_cats[0]
            top_2 = top_cats[1] if len(top_cats) > 1 else None
            msg = f"Your biggest spending category is **{top_1['category']}** at **${top_1['total']:,.2f}** ({top_1['percentage']}% of your total expenses across {top_1['count']} transactions)."
            if top_2:
                msg += f" Your second largest is **{top_2['category']}** at **${top_2['total']:,.2f}** ({top_2['percentage']}%)."
            msg += "\n\nTry simulating a 15–25% reduction in these categories in the 'What If?' tab to see how much you could save!"
            return msg, "Deterministic Advisor"

    elif "save" in q_lower or "goal" in q_lower:
        surplus_or_deficit = rem - goal
        if surplus_or_deficit >= 0:
            return (
                f"You are currently on track to reach your **${goal:,.2f}** savings goal! "
                f"You have **${rem:,.2f}** remaining in your budget, creating a safety margin of **${surplus_or_deficit:,.2f}**.",
                "Deterministic Advisor"
            )
        else:
            return (
                f"You are currently **${abs(surplus_or_deficit):,.2f}** short of your **${goal:,.2f}** savings goal. "
                f"Your remaining budget is **${rem:,.2f}**. Use the 'What If?' simulator to find small adjustments in food or subscriptions to bridge this gap.",
                "Deterministic Advisor"
            )

    return (
        f"Based on your transactions: you have spent **${tot:,.2f}** of your **${budget:,.2f}** budget, "
        f"leaving **${rem:,.2f}** ({metrics.get('budget_status')}). Your daily burn rate is **${metrics.get('daily_average'):,.2f}/day**. "
        f"Tip: Enter an API key in the sidebar to ask open-ended questions to Llama-3.1-8B!",
        "Deterministic Advisor"
    )
