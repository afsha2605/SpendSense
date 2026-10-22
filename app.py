"""
SpendSense - AI Financial Coach & Budget Planner for College Students
A complete hackathon MVP combining Python calculations with hosted open-weight LLMs.
"""

import os
import io
from datetime import datetime, date
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Import domain calculations and AI engine
from utils import (
    COLLEGE_CATEGORIES,
    validate_and_process_csv,
    add_manual_transaction,
    delete_transaction,
    calculate_financial_metrics,
    simulate_what_if,
    generate_ai_financial_insights,
    ask_ai_financial_question,
    generate_rule_based_insights
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SpendSense | AI Budget Coach for Students",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished Hackathon Presentation
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-green { background-color: #D1FAE5; color: #065F46; }
    .badge-yellow { background-color: #FEF3C7; color: #92400E; }
    .badge-red { background-color: #FEE2E2; color: #991B1B; }
    .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'df' not in st.session_state:
    sample_path = os.path.join(os.path.dirname(__file__), "sample_expenses.csv")
    if os.path.exists(sample_path):
        sample_df, _ = validate_and_process_csv(sample_path)
        st.session_state.df = sample_df
    else:
        st.session_state.df = pd.DataFrame(columns=['date', 'amount', 'category', 'description'])

if 'monthly_budget' not in st.session_state:
    env_budget = os.getenv("DEFAULT_MONTHLY_BUDGET")
    st.session_state.monthly_budget = float(env_budget) if env_budget else 1200.0

if 'savings_goal' not in st.session_state:
    env_goal = os.getenv("DEFAULT_SAVINGS_GOAL")
    st.session_state.savings_goal = float(env_goal) if env_goal else 150.0

if 'ai_insights_cache' not in st.session_state:
    st.session_state.ai_insights_cache = None

if 'ai_engine_used' not in st.session_state:
    st.session_state.ai_engine_used = None

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("## 💸 SpendSense")
    st.markdown("*AI Budget Coach for College Students*")
    st.divider()

    # Budget & Goal Inputs
    st.markdown("### 🎯 Monthly Targets")
    monthly_budget = st.number_input(
        "Monthly Budget ($)",
        min_value=50.0,
        max_value=10000.0,
        value=float(st.session_state.monthly_budget),
        step=50.0,
        help="Your total expected money available to spend this month."
    )
    st.session_state.monthly_budget = monthly_budget

    savings_goal = st.number_input(
        "Savings Target ($)",
        min_value=0.0,
        max_value=5000.0,
        value=float(st.session_state.savings_goal),
        step=25.0,
        help="How much you want to tuck away in an emergency fund or savings."
    )
    st.session_state.savings_goal = savings_goal

    st.divider()

    # Data Ingestion Source
    st.markdown("### 📥 Expense Input")
    input_method = st.radio(
        "Choose Data Source:",
        ["Sample Student Data", "Upload CSV File", "Manual Entry"],
        index=0
    )

    if input_method == "Sample Student Data":
        if st.button("🔄 Reload College Sample Dataset", use_container_width=True):
            sample_path = os.path.join(os.path.dirname(__file__), "sample_expenses.csv")
            if os.path.exists(sample_path):
                sample_df, warnings = validate_and_process_csv(sample_path)
                st.session_state.df = sample_df
                st.session_state.ai_insights_cache = None
                st.success("Loaded 25 realistic college transactions!")
                st.rerun()

    elif input_method == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "Upload Bank/Expense CSV",
            type=["csv"],
            help="Must include date and amount. Category and description are optional."
        )
        if uploaded_file is not None:
            if st.button("Process & Load CSV", type="primary", use_container_width=True):
                cleaned_df, warnings = validate_and_process_csv(uploaded_file)
                if cleaned_df is not None:
                    st.session_state.df = cleaned_df
                    st.session_state.ai_insights_cache = None
                    st.success(f"Successfully loaded {len(cleaned_df)} transactions!")
                    for w in warnings:
                        st.info(w)
                    st.rerun()
                else:
                    for err in warnings:
                        st.error(err)

    elif input_method == "Manual Entry":
        with st.form("manual_tx_form", clear_on_submit=True):
            m_date = st.date_input("Transaction Date", value=date.today())
            m_amount = st.number_input("Amount ($)", min_value=0.01, step=1.00, format="%.2f")
            m_category = st.selectbox("Category", COLLEGE_CATEGORIES)
            m_desc = st.text_input("Description / Merchant", placeholder="e.g. Campus bookstore, Boba, Ramen")
            submit_tx = st.form_submit_button("➕ Add Transaction", use_container_width=True)

            if submit_tx:
                st.session_state.df = add_manual_transaction(
                    st.session_state.df,
                    date_val=str(m_date),
                    amount_val=m_amount,
                    category_val=m_category,
                    desc_val=m_desc
                )
                st.session_state.ai_insights_cache = None
                st.success(f"Added ${m_amount:.2f} for {m_desc or m_category}!")
                st.rerun()

    st.divider()

    # AI Configuration (Hosted Open-Weight Models)
    st.markdown("### 🤖 Open-Weight AI Model")
    with st.expander("AI API Settings", expanded=False):
        api_provider = st.selectbox(
            "API Provider:",
            ["Groq (Llama-3.1-8B)", "Hugging Face (Llama-3.2-3B)", "Demo Mode (Offline / Rule-Based)"],
            index=0
        )

        groq_key_default = os.getenv("GROQ_API_KEY", "")
        hf_key_default = os.getenv("HF_API_KEY", "")

        groq_api_key = st.text_input(
            "Groq API Key",
            value=groq_key_default,
            type="password",
            help="Free instant key from https://console.groq.com"
        )
        groq_model = st.selectbox(
            "Groq Model (Open <=10B):",
            ["llama-3.1-8b-instant", "llama-3.2-3b-preview", "llama-3.2-1b-preview"],
            index=0
        )

        hf_api_key = st.text_input(
            "Hugging Face Token",
            value=hf_key_default,
            type="password",
            help="Optional free token from https://huggingface.co/settings/tokens"
        )

    st.caption("🔒 **Security & Privacy:** SpendSense does not link to real bank logins. API keys are loaded securely from `.env` or input above.")


# ==============================================================================
# MAIN PAGE CONTENT
# ==============================================================================
st.markdown('<div class="main-header">🎓 SpendSense</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Budgeting & Expense Intelligence for College Students</div>', unsafe_allow_html=True)

df = st.session_state.df

# Guard against empty dataset
if df is None or df.empty:
    st.warning("⚠️ No transactions loaded yet. Choose 'Sample Student Data' from the sidebar to instantly populate 25 demo expenses, or upload a CSV!")
    if st.button("🚀 Load College Sample Data Now", type="primary"):
        sample_path = os.path.join(os.path.dirname(__file__), "sample_expenses.csv")
        if os.path.exists(sample_path):
            sample_df, _ = validate_and_process_csv(sample_path)
            st.session_state.df = sample_df
            st.rerun()
    st.stop()

# ------------------------------------------------------------------------------
# DETERMINISTIC FINANCIAL CALCULATIONS (Python-Only)
# ------------------------------------------------------------------------------
metrics = calculate_financial_metrics(
    df=df,
    monthly_budget=st.session_state.monthly_budget,
    savings_goal=st.session_state.savings_goal
)

# ------------------------------------------------------------------------------
# TOP METRICS BANNER (4 KPI Cards)
# ------------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Spending",
        value=f"${metrics['total_spending']:,.2f}",
        delta=f"{metrics['transaction_count']} txns",
        delta_color="off"
    )

with col2:
    rem = metrics['remaining_budget']
    budget_pct = metrics['budget_utilization_pct']
    st.metric(
        label=f"Remaining Budget (of ${metrics['monthly_budget']:,.0f})",
        value=f"${rem:,.2f}",
        delta=f"{100.0 - budget_pct:.1f}% left" if rem >= 0 else f"-${abs(rem):,.2f} over",
        delta_color="normal" if rem >= 0 else "inverse"
    )

with col3:
    daily_avg = metrics['daily_average']
    proj = metrics['projected_monthly_spend']
    st.metric(
        label="Daily Burn Rate",
        value=f"${daily_avg:,.2f}/day",
        delta=f"Proj. ${proj:,.0f}/mo",
        delta_color="off"
    )

with col4:
    goal = metrics['savings_goal']
    cur_sav = metrics['current_savings_potential']
    goal_met = metrics['savings_goal_met']
    st.metric(
        label=f"Savings Target (Goal: ${goal:,.0f})",
        value=f"${cur_sav:,.2f}",
        delta="Target On Track" if goal_met else f"-${abs(metrics['savings_deficit_or_surplus']):,.2f} gap",
        delta_color="normal" if goal_met else "inverse"
    )

# Visual status badge
status_color = "badge-green" if metrics['budget_status'] == "Under Budget" else ("badge-yellow" if "Caution" in metrics['budget_status'] else "badge-red")
st.markdown(
    f"<span class='badge-pill {status_color}'>● Status: {metrics['budget_status']}</span> "
    f"&nbsp; <span class='badge-pill badge-blue'>📅 Data Span: {metrics['date_range']['start']} to {metrics['date_range']['end']} ({metrics['date_range']['days_span']} days)</span>",
    unsafe_allow_html=True
)
st.write("")

# ------------------------------------------------------------------------------
# MAIN APPLICATION TABS
# ------------------------------------------------------------------------------
tab_dash, tab_whatif, tab_ai, tab_data = st.tabs([
    "📊 Financial Dashboard",
    "🔮 'What If?' Simulator",
    "🤖 SpendSense AI Advisor",
    "📝 Transactions & Management"
])

# ==============================================================================
# TAB 1: DASHBOARD & CHARTS
# ==============================================================================
with tab_dash:
    st.markdown("### 📈 Spending Insights & Trends")
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown("#### 🍩 Category Breakdown")
        cat_df = pd.DataFrame(metrics['category_breakdown'])
        if not cat_df.empty:
            fig_pie = px.pie(
                cat_df,
                names="category",
                values="total",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Prism,
                hover_data=["percentage", "count", "avg_spend"]
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No category data available.")

    with c_right:
        st.markdown("#### 📉 Cumulative Spend vs Budget")
        # Compute cumulative spending series
        daily_df = df.groupby('date')['amount'].sum().reset_index()
        daily_df['cumulative'] = daily_df['amount'].cumsum()

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=daily_df['date'],
            y=daily_df['cumulative'],
            mode='lines+markers',
            name='Actual Spending',
            line=dict(color='#3B82F6', width=3),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))
        # Add budget reference line
        fig_line.add_trace(go.Scatter(
            x=daily_df['date'],
            y=[st.session_state.monthly_budget] * len(daily_df),
            mode='lines',
            name='Monthly Budget Cap',
            line=dict(color='#EF4444', width=2, dash='dash')
        ))
        fig_line.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=350,
            xaxis_title="Date",
            yaxis_title="Total Dollars ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_line, use_container_width=True)

    st.divider()

    # Category Breakdown Table & Top Outliers
    st.markdown("#### 🏆 Category Rankings & Outliers")
    col_rank, col_top = st.columns([1.2, 0.8])

    with col_rank:
        if not cat_df.empty:
            cat_display = cat_df.copy()
            cat_display['total'] = cat_display['total'].apply(lambda x: f"${x:,.2f}")
            cat_display['avg_spend'] = cat_display['avg_spend'].apply(lambda x: f"${x:,.2f}")
            cat_display['percentage'] = cat_display['percentage'].apply(lambda x: f"{x:.1f}%")
            cat_display.columns = ["Category", "Total Spent", "Transactions", "% of Total", "Avg / Txn"]
            st.dataframe(cat_display, hide_index=True, use_container_width=True)

    with col_top:
        st.markdown("**🔥 Top 3 Single Largest Expenses:**")
        for i, txn in enumerate(metrics['top_transactions'], 1):
            st.markdown(
                f"""
                <div style="background-color:#F1F5F9; padding:8px 12px; border-radius:8px; margin-bottom:8px;">
                    <b>{i}. {txn['description']}</b> (${txn['amount']:,.2f})<br>
                    <span style="font-size:0.85rem; color:#64748B;">{txn['category']} • {txn['date']}</span>
                </div>
                """,
                unsafe_allow_html=True
            )


# ==============================================================================
# TAB 2: "WHAT IF?" SCENARIO SIMULATOR
# ==============================================================================
with tab_whatif:
    st.markdown("### 🔮 'What If?' Scenario Engine")
    st.markdown(
        "See the calculated, real-time effect of reducing spending in any category on your **remaining budget** and **savings goal**."
    )

    categories_present = list(metrics['category_totals'].keys())
    if not categories_present:
        st.info("No expense categories found to simulate.")
    else:
        sim_col1, sim_col2 = st.columns([1, 1.2])

        with sim_col1:
            st.markdown("#### 1. Adjust Spending Limit")
            target_cat = st.selectbox("Select Category to Modify:", categories_present, index=0)
            current_cat_spend = metrics['category_totals'].get(target_cat, 0.0)

            st.write(f"Current actual spending in **{target_cat}**: **${current_cat_spend:,.2f}**")

            sim_mode = st.radio(
                "Adjustment Style:",
                ["Cut by Percentage (%)", "Set Maximum Spending Cap ($)"],
                horizontal=True
            )

            if sim_mode == "Cut by Percentage (%)":
                cut_pct = st.slider(
                    f"Reduce {target_cat} by:",
                    min_value=5,
                    max_value=100,
                    value=25,
                    step=5,
                    format="%d%%"
                )
                target_cap = None
            else:
                cut_pct = 0.0
                target_cap = st.number_input(
                    f"Max Cap for {target_cat} ($):",
                    min_value=0.0,
                    max_value=float(current_cat_spend),
                    value=float(round(current_cat_spend * 0.7, 2)),
                    step=5.0
                )

            # Perform deterministic calculation
            sim_result = simulate_what_if(
                metrics=metrics,
                target_category=target_cat,
                cut_percentage=cut_pct,
                target_cap=target_cap
            )

        with sim_col2:
            st.markdown("#### 2. Calculated Financial Impact")
            st.success(sim_result['summary_text'])

            # Impact Metric Cards
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric(
                    label="Immediate Savings",
                    value=f"+${sim_result['dollar_savings']:,.2f}",
                    delta="Kept in pocket",
                    delta_color="normal"
                )
            with sc2:
                st.metric(
                    label="New Remaining Budget",
                    value=f"${sim_result['new_remaining']:,.2f}",
                    delta=f"+${sim_result['dollar_savings']:,.2f}",
                    delta_color="normal"
                )
            with sc3:
                st.metric(
                    label="Savings Goal Coverage",
                    value=f"{sim_result['pct_goal_covered']:.1f}%",
                    delta="Goal achieved!" if sim_result['savings_goal_achieved'] else f"${sim_result['new_savings_gap']:,.2f} gap",
                    delta_color="normal" if sim_result['savings_goal_achieved'] else "inverse"
                )

            # Comparison Visual Chart
            comp_fig = go.Figure(data=[
                go.Bar(
                    name="Current Spending",
                    x=[target_cat, "Total Spending", "Remaining Budget"],
                    y=[sim_result['baseline_cat_spend'], metrics['total_spending'], sim_result['baseline_remaining']],
                    marker_color='#94A3B8'
                ),
                go.Bar(
                    name="After What-If Simulation",
                    x=[target_cat, "Total Spending", "Remaining Budget"],
                    y=[sim_result['new_cat_spend'], sim_result['new_total_spend'], sim_result['new_remaining']],
                    marker_color='#10B981'
                )
            ])
            comp_fig.update_layout(
                barmode='group',
                height=260,
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(comp_fig, use_container_width=True)


# ==============================================================================
# TAB 3: AI FINANCIAL ADVISOR & CHAT
# ==============================================================================
with tab_ai:
    st.markdown("### 🤖 SpendSense AI Advisor")
    st.markdown(
        "Powered by hosted open-weight models (<=10B parameters, e.g. **Llama-3.1-8B**). "
        "All calculations are grounded in pure Python truth."
    )

    ai_col1, ai_col2 = st.columns([1, 1])

    with ai_col1:
        st.markdown("#### 📋 Comprehensive Financial Diagnosis")
        if st.button("✨ Generate / Refresh AI Action Plan", type="primary", use_container_width=True):
            with st.spinner("Analyzing spending patterns and generating student recommendations..."):
                active_groq_key = groq_api_key if api_provider.startswith("Groq") else None
                active_hf_key = hf_api_key if api_provider.startswith("Hugging Face") else None

                insights_text, engine_used = generate_ai_financial_insights(
                    metrics=metrics,
                    groq_api_key=active_groq_key,
                    hf_api_key=active_hf_key,
                    groq_model=groq_model
                )
                st.session_state.ai_insights_cache = insights_text
                st.session_state.ai_engine_used = engine_used

        if st.session_state.ai_insights_cache:
            st.caption(f"⚡ Engine: **{st.session_state.ai_engine_used}**")
            st.markdown(st.session_state.ai_insights_cache)
        else:
            st.info("Click the button above to generate a customized diagnosis of your college spending.")

    with ai_col2:
        st.markdown("#### 💬 Ask SpendSense AI Anything")
        st.caption("Ask questions about your finances, affordability, or savings tactics.")

        # Quick Prompt Chips
        st.write("**Quick Student Questions:**")
        qp_col1, qp_col2 = st.columns(2)
        quick_prompt = None
        with qp_col1:
            if st.button("🍕 Can I afford a $45 dinner?", use_container_width=True):
                quick_prompt = "Can I afford to spend $45 on dinner with friends tonight?"
            if st.button("🎯 How do I hit my savings goal?", use_container_width=True):
                quick_prompt = "How can I realistically reach my savings goal this month?"
        with qp_col2:
            if st.button("🔍 Where am I leaking money?", use_container_width=True):
                quick_prompt = "Where am I leaking the most money and how can I cut back?"
            if st.button("☕ Audit my coffee/boba spend", use_container_width=True):
                quick_prompt = "How much am I spending on coffee and boba, and what's a good alternative?"

        # Chat input
        user_input = st.chat_input("Ask about your budget, e.g., 'Can I afford a $120 concert ticket?'")
        final_query = user_input or quick_prompt

        # Display Chat History
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if final_query:
            # Append User message
            st.session_state.chat_messages.append({"role": "user", "content": final_query})
            with st.chat_message("user"):
                st.markdown(final_query)

            # Generate AI Response
            with st.chat_message("assistant"):
                with st.spinner("SpendSense is consulting your transactions..."):
                    active_groq_key = groq_api_key if api_provider.startswith("Groq") else None
                    active_hf_key = hf_api_key if api_provider.startswith("Hugging Face") else None

                    answer, engine = ask_ai_financial_question(
                        user_question=final_query,
                        metrics=metrics,
                        chat_history=st.session_state.chat_messages,
                        groq_api_key=active_groq_key,
                        hf_api_key=active_hf_key,
                        groq_model=groq_model
                    )
                    st.markdown(answer)
                    st.caption(f"Grounding source: Python Metrics Engine | LLM: {engine}")

            st.session_state.chat_messages.append({"role": "assistant", "content": answer})


# ==============================================================================
# TAB 4: TRANSACTIONS & MANAGEMENT
# ==============================================================================
with tab_data:
    st.markdown("### 📝 Transaction Explorer")

    # Filter bar
    f1, f2, f3 = st.columns([1, 1, 1])
    with f1:
        cat_filter = st.multiselect("Filter by Category:", options=COLLEGE_CATEGORIES + ["Other"])
    with f2:
        search_query = st.text_input("Search Description / Merchant:", "")
    with f3:
        sort_choice = st.selectbox("Sort By:", ["Date (Newest First)", "Date (Oldest First)", "Amount (Highest)", "Amount (Lowest)"])

    # Apply filters
    filtered_df = df.copy()
    if cat_filter:
        filtered_df = filtered_df[filtered_df['category'].isin(cat_filter)]
    if search_query:
        filtered_df = filtered_df[filtered_df['description'].str.contains(search_query, case=False, na=False)]

    if sort_choice == "Date (Newest First)":
        filtered_df = filtered_df.sort_values(by="date", ascending=False)
    elif sort_choice == "Date (Oldest First)":
        filtered_df = filtered_df.sort_values(by="date", ascending=True)
    elif sort_choice == "Amount (Highest)":
        filtered_df = filtered_df.sort_values(by="amount", ascending=False)
    elif sort_choice == "Amount (Lowest)":
        filtered_df = filtered_df.sort_values(by="amount", ascending=True)

    # Display count and table
    st.write(f"Showing **{len(filtered_df)}** of **{len(df)}** transactions")

    # Format table for display
    display_table = filtered_df.copy()
    display_table['amount'] = display_table['amount'].apply(lambda x: f"${x:,.2f}")
    st.dataframe(display_table, use_container_width=True, hide_index=False)

    # Download button
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned Transactions CSV",
        data=csv_bytes,
        file_name="spendsense_expenses.csv",
        mime="text/csv"
    )

    # Delete row utility
    with st.expander("🗑️ Delete a Transaction"):
        del_idx = st.number_input(
            "Enter Row Index to Delete:",
            min_value=0,
            max_value=max(0, len(df)-1),
            value=0,
            step=1
        )
        if st.button("Confirm Delete Row", type="secondary"):
            st.session_state.df = delete_transaction(st.session_state.df, int(del_idx))
            st.session_state.ai_insights_cache = None
            st.success(f"Deleted transaction at index {del_idx}!")
            st.rerun()

# ------------------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------------------
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #94A3B8; font-size: 0.85rem;">
        <b>SpendSense</b> — Built for College Students • Deterministic Financial Math • Hosted Open-Weight Models (≤10B)<br>
        <i>Disclaimer: Educational budgeting tool. Not professional financial advice. Never input confidential banking credentials.</i>
    </div>
    """,
    unsafe_allow_html=True
)
