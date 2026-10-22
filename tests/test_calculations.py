"""
Unit tests for SpendSense deterministic financial engine, CSV parsing,
what-if simulations, and grounded AI advisor Q&A.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
from utils import (
    validate_and_process_csv,
    add_manual_transaction,
    delete_transaction,
    calculate_financial_metrics,
    simulate_what_if,
    build_ai_grounding_context,
    generate_rule_based_insights,
    ask_ai_financial_question
)

@pytest.fixture
def sample_csv_path():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_expenses.csv")
    assert os.path.exists(path), "sample_expenses.csv must exist"
    return path

def test_load_sample_expenses(sample_csv_path):
    df, warnings = validate_and_process_csv(sample_csv_path)
    assert df is not None
    assert len(df) == 25
    assert set(df.columns) == {"date", "amount", "category", "description"}
    assert (df['amount'] > 0).all()

def test_csv_validation_with_missing_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("amount,category\n12.50,Food\n")
    df, warnings = validate_and_process_csv(str(bad_csv))
    assert df is None
    assert any("date" in w for w in warnings)

def test_csv_validation_with_messy_amounts_and_dates(tmp_path):
    messy_csv = tmp_path / "messy.csv"
    messy_csv.write_text(
        "Date,Amount,Category,Description\n"
        "2026-10-01,\"$1,200.50\",Rent,Apartment rent\n"
        "2026-10-02,  -15.75  ,Food,Chipotle\n"
        "not-a-date,10.00,Food,Coffee\n"
        "2026-10-04,0.00,Misc,Free sample\n"
        "2026-10-05,25.00,,Bus pass\n"
    )
    df, warnings = validate_and_process_csv(str(messy_csv))
    assert df is not None
    assert len(df) == 3
    assert df.loc[df['description'] == 'Apartment rent', 'amount'].iloc[0] == 1200.50
    assert df.loc[df['description'] == 'Chipotle', 'amount'].iloc[0] == 15.75
    assert df.loc[df['description'] == 'Bus pass', 'category'].iloc[0] == "Other"

def test_csv_alternate_column_headers(tmp_path):
    alt_csv = tmp_path / "bank_export.csv"
    alt_csv.write_text(
        "Transaction Date,Cost,Classification,Vendor\n"
        "2026-10-01,15.50,Dining,Subway\n"
        "2026-10-02,45.00,Books,Campus Bookstore\n"
    )
    df, warnings = validate_and_process_csv(str(alt_csv))
    assert df is not None
    assert len(df) == 2
    assert set(df.columns) == {"date", "amount", "category", "description"}
    assert df.iloc[0]['amount'] == 15.50
    assert df.iloc[0]['description'] == "Subway"

def test_manual_transaction_operations():
    df = pd.DataFrame([
        {"date": "2026-10-01", "amount": 50.0, "category": "Food", "description": "Groceries"}
    ])
    df = add_manual_transaction(df, "2026-10-02", 12.50, "Coffee & Boba", "Latte")
    assert len(df) == 2
    assert df.iloc[1]['amount'] == 12.50
    assert df.iloc[1]['category'] == "Coffee & Boba"

    df = delete_transaction(df, 0)
    assert len(df) == 1
    assert df.iloc[0]['description'] == "Latte"

def test_financial_metrics_calculation(sample_csv_path):
    df, _ = validate_and_process_csv(sample_csv_path)
    metrics = calculate_financial_metrics(df, monthly_budget=1200.0, savings_goal=150.0)

    assert metrics['transaction_count'] == 25
    expected_sum = round(float(df['amount'].sum()), 2)
    assert metrics['total_spending'] == expected_sum
    assert metrics['monthly_budget'] == 1200.0
    assert metrics['remaining_budget'] == round(1200.0 - expected_sum, 2)
    assert metrics['daily_average'] > 0
    assert len(metrics['category_breakdown']) > 0
    assert len(metrics['top_categories']) <= 3
    assert len(metrics['top_transactions']) <= 3

    total_pct = sum(c['percentage'] for c in metrics['category_breakdown'])
    assert 99.0 <= total_pct <= 101.0

def test_empty_dataframe_metrics():
    empty_df = pd.DataFrame(columns=['date', 'amount', 'category', 'description'])
    metrics = calculate_financial_metrics(empty_df, monthly_budget=1000.0, savings_goal=100.0)

    assert metrics['total_spending'] == 0.0
    assert metrics['remaining_budget'] == 1000.0
    assert metrics['transaction_count'] == 0
    assert metrics['daily_average'] == 0.0
    assert metrics['budget_status'] == "No Expenses"

def test_simulate_what_if_percentage_cut(sample_csv_path):
    df, _ = validate_and_process_csv(sample_csv_path)
    metrics = calculate_financial_metrics(df, monthly_budget=1200.0, savings_goal=150.0)

    dining_spend = metrics['category_totals'].get('Dining & Fast Food', 0.0)
    assert dining_spend > 0

    res = simulate_what_if(
        metrics=metrics,
        target_category='Dining & Fast Food',
        cut_percentage=50.0
    )

    expected_savings = round(dining_spend * 0.5, 2)
    assert res['dollar_savings'] == expected_savings
    assert res['new_cat_spend'] == round(dining_spend - expected_savings, 2)
    assert res['new_total_spend'] == round(metrics['total_spending'] - expected_savings, 2)
    assert res['new_remaining'] == round(metrics['remaining_budget'] + expected_savings, 2)

def test_simulate_what_if_fixed_cap(sample_csv_path):
    df, _ = validate_and_process_csv(sample_csv_path)
    metrics = calculate_financial_metrics(df, monthly_budget=1200.0, savings_goal=150.0)

    groceries_spend = metrics['category_totals'].get('Groceries', 0.0)
    assert groceries_spend > 100.0

    target_cap = 100.0
    res = simulate_what_if(
        metrics=metrics,
        target_category='Groceries',
        target_cap=target_cap
    )

    expected_savings = round(groceries_spend - target_cap, 2)
    assert res['dollar_savings'] == expected_savings
    assert res['new_cat_spend'] == target_cap
    assert res['new_total_spend'] == round(metrics['total_spending'] - expected_savings, 2)

def test_ai_grounding_context_contains_pure_facts(sample_csv_path):
    df, _ = validate_and_process_csv(sample_csv_path)
    metrics = calculate_financial_metrics(df, monthly_budget=1200.0, savings_goal=150.0)
    context = build_ai_grounding_context(metrics)

    assert f"${metrics['total_spending']:,.2f}" in context
    assert f"${metrics['remaining_budget']:,.2f}" in context
    assert "CALCULATED FINANCIAL SUMMARY" in context

def test_rule_based_insights_generation(sample_csv_path):
    df, _ = validate_and_process_csv(sample_csv_path)
    metrics = calculate_financial_metrics(df, monthly_budget=1200.0, savings_goal=150.0)
    insights = generate_rule_based_insights(metrics)

    assert "SpendSense Financial Diagnosis" in insights
    assert "Tactical Student Recommendations" in insights

def test_ask_ai_financial_questions_grounded(sample_csv_path):
    df, _ = validate_and_process_csv(sample_csv_path)
    metrics = calculate_financial_metrics(df, monthly_budget=1200.0, savings_goal=150.0)

    # Test afford question within budget
    ans, engine = ask_ai_financial_question("Can I afford a $30 dinner?", metrics)
    assert "afford" in ans.lower() or "$30" in ans

    # Test afford question exceeding budget
    ans_expensive, _ = ask_ai_financial_question("Can I afford a $2000 laptop right now?", metrics)
    assert "exceed" in ans_expensive.lower() or "no" in ans_expensive.lower()

    # Test spending leakage question
    ans_leak, _ = ask_ai_financial_question("Where am I leaking the most money?", metrics)
    top_cat = metrics['top_categories'][0]['category']
    assert top_cat in ans_leak

    # Test savings goal question
    ans_goal, _ = ask_ai_financial_question("How is my savings goal looking?", metrics)
    assert "savings" in ans_goal.lower() or "goal" in ans_goal.lower()
