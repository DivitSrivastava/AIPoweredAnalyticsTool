# -*- coding: utf-8 -*-

# --- 1. CORE DATA & UI IMPORTS ---
import os
import gradio as gr
import pandas as pd
import numpy as np
import dill as pickle
import datetime as dt
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from ucimlrepo import fetch_ucirepo
import re

# --- 2. LLM AGENT LIBRARIES (Compatibility Mode) ---
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool

# Using the classic compatibility layer for stable Agent logic
try:
    from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
except ImportError:
    # Extreme fallback for local testing if classic isn't installed
    from langchain.agents import create_tool_calling_agent, AgentExecutor

# --- 3. VERSION CHECK ---
try:
    import langchain_classic
    print(f"LANGCHAIN CLASSIC LOADED: {getattr(langchain_classic, '__version__', 'Yes')}")
except Exception:
    print("Version check skipped.")

# --- CONFIGURATION (CRITICAL EDITS FOR HUGGING FACE SPACE) ---
# SECURE API KEY: This app now reads API KEY
# from the environment (Hugging Face Space Secrets).

# Set the model directory to the root of the space (where all model files are located).
MODELS_DIR = './' # All files are in the main

# --- MODEL PATHS ---
SALES_MODEL_PATH = os.path.join(MODELS_DIR, 'sales_trend_prophet_model.pkl')
TOP_PRODUCTS_PATH = os.path.join(MODELS_DIR, 'top_products.pkl')
BG_NBD_MODEL_PATH = os.path.join(MODELS_DIR, 'bg_nbd_model.pkl')
GAMMA_GAMMA_MODEL_PATH = os.path.join(MODELS_DIR, 'gamma_gamma_model.pkl')

# --- Helper Function to Load Models ---
def load_model(path):
    try:
        # Use 'dill' to load models saved with dill
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"Loaded model from: {path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {path}. Please ensure all .pkl files are in the root directory.")
        return None
    except Exception as e:
        print(f"Error loading model from {path}: {e}")
        return None

# --- Global Model Loading ---
print("Attempting to load models...")
sales_model = load_model(SALES_MODEL_PATH)
top_products = load_model(TOP_PRODUCTS_PATH)
bg_nbd_model = load_model(BG_NBD_MODEL_PATH)
gamma_gamma_model = load_model(GAMMA_GAMMA_MODEL_PATH)

demand_models = {}
if top_products:
    for p_code in top_products:
        # Model path relative to the root directory
        model_path = os.path.join(MODELS_DIR, f'demand_prophet_model_{p_code}.pkl')
        model = load_model(model_path)
        if model:
            demand_models[p_code] = model
        else:
            print(f"Could not load demand model for product {p_code}.")
else:
    print("Could not load top products list.")
print("Model loading complete.")

# --- GLOBAL DATA CACHE (Prepare data once for the AI) ---
print("Pre-fetching and cleaning UCI dataset for the AI Assistant...")
try:
    online_retail = fetch_ucirepo(id=352)
    raw_df = online_retail.data.original
    
    # Pre-clean the data so the tools don't have to do it every time
    raw_df['InvoiceDate'] = pd.to_datetime(raw_df['InvoiceDate'])
    raw_df['Sales'] = raw_df['Quantity'] * raw_df['UnitPrice']
    raw_df = raw_df[~raw_df['InvoiceNo'].astype(str).str.contains('C', na=False)]
    raw_df.dropna(subset=['CustomerID'], inplace=True)
    raw_df['CustomerID'] = raw_df['CustomerID'].astype(int)
    raw_df = raw_df[(raw_df['Sales'] > 0) & (raw_df['Quantity'] > 0)]
    
    # The 'summary_df' is what the CLV models actually need
    analysis_end_date = raw_df['InvoiceDate'].max() + dt.timedelta(days=1)
    CACHED_SUMMARY_DF = summary_data_from_transaction_data(
        raw_df, 
        customer_id_col='CustomerID', 
        datetime_col='InvoiceDate', 
        monetary_value_col='Sales', 
        observation_period_end=analysis_end_date
    )
    print(f"Cache Ready! Processed {len(CACHED_SUMMARY_DF)} customer records.")
except Exception as e:
    CACHED_SUMMARY_DF = None
    print(f"WARNING: Data cache failed: {e}")

# --- Plotting Style ---
sns.set_theme(style="whitegrid", palette="pastel")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# --------------------------------------------------------
# --- CORE PREDICTION FUNCTIONS (FOR GRADIO UI TABS) ---
# --------------------------------------------------------

def predict_overall_sales(periods, output_choice):
    if sales_model is None:
        gr.Warning("Overall sales trend model not loaded.")
        return (gr.DataFrame(value=pd.DataFrame(), visible=False), gr.Number(value=0.0, visible=False), None, None)

    future = sales_model.make_future_dataframe(periods=periods)
    forecast = sales_model.predict(future)

    output_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).copy()
    output_df['ds'] = output_df['ds'].dt.strftime('%Y-%m-%d')
    output_df['yhat'] = output_df['yhat'].round(2)
    output_df['yhat_lower'] = output_df['yhat_lower'].round(2)
    output_df['yhat_upper'] = output_df['yhat_upper'].round(2)

    total_sales_sum = output_df['yhat'].sum().round(2)

    output_df.rename(columns={
        'ds': 'Date',
        'yhat': 'Predicted Daily Sales (Sterling)',
        'yhat_lower': 'Lower Confidence Bound (Sterling)',
        'yhat_upper': 'Upper Confidence Bound (Sterling)'
    }, inplace=True)

    fig1 = sales_model.plot(forecast)
    plt.title('Overall Daily Sales Forecast')
    plt.xlabel('Datestamp')
    plt.ylabel('Total Sales (Sterling)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    fig2 = sales_model.plot_components(forecast)
    plt.suptitle('Overall Sales Forecast Components', y=1.02)
    for ax in fig2.axes:
        ax.set_xlabel('Datestamp')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    return (gr.DataFrame(value=output_df, visible=output_choice == "Detailed Table"),
            gr.Number(value=total_sales_sum, visible=output_choice == "Total Sum for Period"),
            fig1, fig2)


def predict_product_demand(product_code, periods, output_choice):
    if product_code not in demand_models:
        gr.Warning("Invalid or missing product model.")
        return (gr.DataFrame(value=pd.DataFrame(), visible=False), gr.Number(value=0.0, visible=False), None, None)

    model = demand_models[product_code]
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    output_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).copy()
    output_df['ds'] = output_df['ds'].dt.strftime('%Y-%m-%d')
    output_df['yhat'] = output_df['yhat'].round(2)
    output_df['yhat_lower'] = output_df['yhat_lower'].round(2)
    output_df['yhat_upper'] = output_df['yhat_upper'].round(2)

    total_quantity_sum = output_df['yhat'].sum().round(2)

    output_df.rename(columns={
        'ds': 'Date',
        'yhat': 'Predicted Daily Quantity',
        'yhat_lower': 'Lower Confidence Bound (Quantity)',
        'yhat_upper': 'Upper Confidence Bound (Quantity)'
    }, inplace=True)

    fig1 = model.plot(forecast)
    plt.title(f'Demand Forecast for Product {product_code}')
    plt.xlabel('Datestamp')
    plt.ylabel('Quantity Sold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    fig2 = model.plot_components(forecast)
    plt.suptitle(f'Forecast Components for Product {product_code}', y=1.02)
    for ax in fig2.axes:
        ax.set_xlabel('Datestamp')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    return (gr.DataFrame(value=output_df, visible=output_choice == "Detailed Table"),
            gr.Number(value=total_quantity_sum, visible=output_choice == "Total Sum for Period"),
            fig1, fig2)


def predict_clv(periods_to_predict, customers_to_show, customer_id=None):
    # 1. Validation Check
    if bg_nbd_model is None or gamma_gamma_model is None or CACHED_SUMMARY_DF is None:
        gr.Warning("CLV models or data cache not loaded.")
        return pd.DataFrame({'CustomerID': [], 'Predicted CLV (£)': []})

    try:
        # 2. Use the Cached Data (No more fetch_ucirepo here!)
        summary_df = CACHED_SUMMARY_DF.copy()

        # 3. Predict future purchases
        summary_df['predicted_purchases'] = bg_nbd_model.predict(
            periods_to_predict, summary_df['frequency'], summary_df['recency'], summary_df['T']
        )

        # 4. Predict monetary value (for customers with frequency > 0)
        summary_df['predicted_monetary'] = 0.0
        eligible = summary_df['frequency'] > 0
        if eligible.any():
            summary_df.loc[eligible, 'predicted_monetary'] = gamma_gamma_model.conditional_expected_average_profit(
                summary_df.loc[eligible, 'frequency'], summary_df.loc[eligible, 'monetary_value']
            )

        # 5. Calculate final CLV
        summary_df['CLV'] = (summary_df['predicted_purchases'] * summary_df['predicted_monetary']).round(2)
        result_df = summary_df[['CLV']].copy()
        result_df.index.name = 'CustomerID'

        # 6. Logic for Single Customer ID vs. Top List
        if customer_id and customer_id > 0:
            customer_id = int(customer_id)
            if customer_id in result_df.index:
                single_clv = result_df.loc[[customer_id]].reset_index()
                single_clv.rename(columns={'CLV': 'Predicted CLV (£)'}, inplace=True)
                return single_clv[['CustomerID', 'Predicted CLV (£)']]
            else:
                gr.Warning(f"Customer ID {customer_id} not found.")
                return pd.DataFrame({'CustomerID': [], 'Predicted CLV (£)': []})
        else:
            # Handle top customers display
            overall_clv_df = result_df.sort_values(by='CLV', ascending=False).reset_index()
            num_results = min(int(customers_to_show), len(overall_clv_df))
            overall_clv_df = overall_clv_df.head(num_results)
            overall_clv_df.rename(columns={'CLV': 'Predicted CLV (£)'}, inplace=True)
            return overall_clv_df[['CustomerID', 'Predicted CLV (£)']]

    except Exception as e:
        gr.Error(f"Error during CLV prediction: {e}")
        return pd.DataFrame({'CustomerID': [], 'Predicted CLV (£)': []})

# -----------------------------------------------------
# --- LLM TOOL FUNCTIONS ---
# -----------------------------------------------------

@tool
def llm_forecast_overall_sales(periods_days: int) -> str:
    """Forecasts total store sales trend in Sterling for next N days using Prophet."""
    if sales_model is None:
        return "Overall sales trend model not loaded. Cannot perform forecast."
    try:
        future = sales_model.make_future_dataframe(periods=periods_days)
        forecast = sales_model.predict(future)
        total_sales_sum = forecast['yhat'].tail(periods_days).sum().round(2)
        return f"Total Overall Sales Forecasted for the next {periods_days} days is: £{total_sales_sum:,.2f}."
    except Exception as e:
        return f"An error occurred during overall sales forecasting: {e}"

@tool
def llm_forecast_product_demand(product_code: str, periods_days: int) -> str:
    """Forecasts units for product IDs (e.g., 23166, 22197, 84077) for next N days."""
    if product_code not in demand_models:
        return f"Demand model for product '{product_code}' not loaded or product ID is invalid. Available IDs: {list(demand_models.keys())}"
    try:
        model = demand_models[product_code]
        future = model.make_future_dataframe(periods=periods_days)
        forecast = model.predict(future)
        total_quantity_sum = forecast['yhat'].tail(periods_days).sum().round(2)
        return f"Total forecasted quantity for product '{product_code}' over {periods_days} days is: {total_quantity_sum:,.0f} units."
    except Exception as e:
        return f"An error occurred during product demand forecasting: {e}"

@tool
def llm_predict_clv(customer_id: int, periods_months: int) -> str:
    """Predicts Customer Lifetime Value (CLV) in £ for a specific ID over N months."""
    if bg_nbd_model is None or gamma_gamma_model is None or CACHED_SUMMARY_DF is None:
        return "CLV engine or data cache is unavailable."

    try:
        if customer_id not in CACHED_SUMMARY_DF.index:
            return f"Customer ID {customer_id} not found in our records."

        customer_data = CACHED_SUMMARY_DF.loc[customer_id]

        # Prediction logic using the cache
        predicted_purchases = bg_nbd_model.predict(
            periods_months, customer_data['frequency'], customer_data['recency'], customer_data['T']
        )
        
        predicted_monetary = gamma_gamma_model.conditional_expected_average_profit(
            customer_data['frequency'], customer_data['monetary_value']
        ) if customer_data['frequency'] > 0 else 0.0

        predicted_clv = predicted_purchases * predicted_monetary

        return f"The predicted CLV for Customer ID {customer_id} over {periods_months} months is: £{predicted_clv:,.2f}."

    except Exception as e:
        return f"Error during CLV calculation: {e}"

@tool
def llm_get_top_clv_customers(periods_months: int, customers_to_show: int = 5) -> str:
    """Returns top N customer IDs with highest predicted CLV for a period."""
    if bg_nbd_model is None or gamma_gamma_model is None or CACHED_SUMMARY_DF is None:
        return "CLV engine or data cache is unavailable."

    try:
        # We work on a copy of the cache to avoid modifying the original data
        temp_df = CACHED_SUMMARY_DF.copy()
        
        temp_df['predicted_purchases'] = bg_nbd_model.predict(
            periods_months, temp_df['frequency'], temp_df['recency'], temp_df['T']
        )
        
        # Calculate expected profit for eligible customers
        eligible = temp_df['frequency'] > 0
        temp_df.loc[eligible, 'expected_profit'] = gamma_gamma_model.conditional_expected_average_profit(
            temp_df.loc[eligible, 'frequency'], temp_df.loc[eligible, 'monetary_value']
        )
        temp_df['expected_profit'].fillna(0, inplace=True)

        temp_df['CLV'] = temp_df['predicted_purchases'] * temp_df['expected_profit']
        top_customers = temp_df.sort_values(by='CLV', ascending=False).head(customers_to_show)
        
        results = [f"ID {cid}: £{clv:,.2f}" for cid, clv in top_customers['CLV'].items()]
        return f"Top {customers_to_show} customers for the next {periods_months} months:\n" + "\n".join(results)

    except Exception as e:
        return f"Error fetching top customers: {e}"

# --------------------------------------------------------
# --- LLM AGENT SETUP (Groq Integration) ---
# --------------------------------------------------------

# Check for Groq API key
groq_key = os.environ.get("GROQ_API_KEY")

if groq_key:
    # Initialize Groq LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", # High-performance, tool-optimized model
        groq_api_key=groq_key,
        temperature=0,      # Absolute zero for consistent logic & tool calling
        max_retries=2,      # Groq is fast, so few retries are needed
        timeout=60
    )
else:
    llm = None
    print("FATAL: GROQ_API_KEY not set. Chatbot tab will not function.")

tools = [
    llm_predict_clv,
    llm_get_top_clv_customers, 
    llm_forecast_overall_sales,
    llm_forecast_product_demand
]

# Strict Content for the Llama 3.3 model
system_prompt_content = """Role: Strategic E-commerce Analytics Engine.

CRITICAL PROTOCOLS:
1. TOOL SELECTION: If the user asks for sales, demand, or CLV, call ONE relevant tool immediately.
2. DURATION PARSING: Convert "a week" to 7, "month" to 30, "year" to 365, etc.
3. FINALITY: Once you have the data from a tool, provide the final answer and STOP. Do not call more tools.
4. NO CONVERSATION: Provide ONLY the data.

STRICT FORMATTING:
- For Product IDs: Use only the numbers provided.
- For Durations: Use only integers.

If the query is out of scope, respond: "Out of scope. I support: Total Sales, Product Demand, and Customer CLV."
"""

if llm:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_content),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Executor with supported parameters
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,               # Kept True to see the "Thought" process in logs
        max_iterations=10, 
        handle_parsing_errors=True, # Sends logic errors back to Llama to fix
        early_stopping_method="force" # Does not support "generate" so swapped to the supported "force"
    )

    def respond_to_chat(message, chat_history):
        """Invokes the Groq Agent. Optimized for speed and no-payment tier."""
        user_input = message["text"] if isinstance(message, dict) else message

        try:
            # Empty chat_history to stay within free tier token limits
            response = agent_executor.invoke({
                "input": user_input, 
                "chat_history": []
            }) 
            
            return response.get('output', "I'm sorry, I couldn't generate a response.")

        except Exception as e:
            err_str = str(e)
            
            # Handling Rate Limits (Because of Free Tiers)
            if "429" in err_str or "rate_limit" in err_str:
                seconds_match = re.search(r'retry in (\d+\.?\d*)s', err_str)
                wait_time = seconds_match.group(1) if seconds_match else "a few"
                return f"Rate limit reached. Please try again in {wait_time} seconds."
            
            return f"An error occurred: {e}"
else:
    def respond_to_chat(message, chat_history):
        return "The AI agent is not configured. Please set the GROQ_API_KEY environment variable."

# -----------------------------------------------------
# --- GRADIO UI LAYOUT ---
# -----------------------------------------------------

custom_theme = gr.themes.Soft()

with gr.Blocks(theme=custom_theme, title="AI Powered Analytics Tool") as demo:
    # --- 1. HEADER & DYNAMIC STATUS ---
    gr.Markdown("# 📈 AI Powered Analytics Tool - By Divit Srivastava")
    
    # This component is updated automatically via the demo.load function at the bottom
    status_display = gr.Markdown("⏳ **Status:** Initializing Analytics Engine...")
    
    gr.Markdown("---")
    
    # --- 2. ABOUT THE APP ---
    gr.Markdown("""
    ## About This App
    This **AI-Driven Analytics Forecaster** is a comprehensive web application designed to provide predictive insights across three core business areas: **Sales**, **Product Demand**, and **Customer Lifetime Value (CLV)**.

    It offers two distinct modes of interaction:
    1.  **Dedicated Tabs** for in-depth analysis, visualizations, and detailed tables.
    2.  **AI Assistant Chatbot** 🤖 for fast, conversational answers to forecasting questions, powered by a LangChain agent.

    The application leverages historical transaction data from the **UCI Machine Learning Repository**, applying sophisticated **time-series forecasting** (**Prophet**) and **probabilistic customer behavior models** (**Lifetimes**).

    Developed using **Python 🐍**, the app integrates **Gradio 🎨** for its interactive interface and a powerful **LangChain 🦜** agent powered by **Llama 3.3 🦙 on Groq Cloud ⚡**. **This powerful fusion of predictive models and conversational AI transforms complex data into precise, actionable intelligence.**
    """)
    
    # --- 3. HOW TO USE ---
    with gr.Accordion("📖 How to Use", open=False):
        gr.Markdown("""
        1.  **Select Your Tab:** Choose between **'Overall Sales Forecast'** to predict store-wide trends, **'Product Demand Forecast'** for specific items, the **'Customer Lifetime Value (CLV) Predictor'** for customer insights, or the **'AI Assistant Chatbot'** for conversational queries.
        2.  **Input Future Days/Months:** Enter the number of future days for sales/demand forecasts, or the number of months for CLV prediction.
        3.  **Choose a Product (for Demand Forecast):** If on the 'Product Demand Forecast' tab, use the dropdown to select a specific top-selling product.
        4.  **Specify CLV Options (for CLV Predictor):** On the CLV tab, you can choose how many top CLV customers to view, or enter a specific Customer ID to see their individual prediction.
        5.  **Click to Forecast/Predict:** Hit the 'Forecast Sales', 'Forecast Demand', or 'Predict CLV' button which will generate and display the prediction.
        """)

    # --- 4. TABS SECTION ---
    with gr.Tabs():
        # TAB 1: Overall Sales Forecast 
        with gr.Tab("Overall Sales Forecast"):
            gr.Markdown("### 🔮 Forecast Overall Sales")
            overall_days = gr.Number(
                label="Enter number of future days to forecast (e.g., 30):",
                value=30,
                minimum=7,
                maximum=365,
                step=7
            )
            overall_output_choice = gr.Radio(
                ["Detailed Table", "Total Sum for Period"],
                label="Choose Output Format",
                value="Detailed Table",
                interactive=True
            )
            overall_btn = gr.Button("Forecast Sales")
            gr.Markdown("""
            **Note on Negative Values:** Negative sales values in the forecast, indicate a **prediction of returned orders** for that period.
            """)
            overall_table = gr.DataFrame(label="Forecast Data Table", visible=True)
            overall_total_output = gr.Number(label="Total Forecasted Sales (Sterling) for Period", precision=2, visible=False)
            overall_plot = gr.Plot(label="Overall Sales Forecast Plot")
            gr.Markdown("This graph illustrates the overall daily sales forecast, showing the predicted trend and confidence intervals.")
            overall_components = gr.Plot(label="Overall Sales Forecast Components")
            gr.Markdown("This graph breaks down the overall sales forecast into its underlying components: trend, weekly seasonality, and yearly seasonality.")
            overall_btn.click(
                fn=predict_overall_sales,
                inputs=[overall_days, overall_output_choice],
                outputs=[overall_table, overall_total_output, overall_plot, overall_components]
            )

        # TAB 2: Product Demand Forecast 
        with gr.Tab("Product Demand Forecast"):
            gr.Markdown("### 📦 Forecast Product Demand")
            product_selector = gr.Dropdown(
                label="Select the Product ID:",
                choices=list(demand_models.keys()) if top_products else ["No products loaded"],
                value=list(demand_models.keys())[0] if top_products else None,
                interactive=True
            )
            product_days = gr.Number(
                label="Enter number of future days to forecast:",
                value=30,
                minimum=7,
                maximum=365,
                step=7
            )
            product_output_choice = gr.Radio(
                ["Detailed Table", "Total Sum for Period"],
                label="Choose Output Format",
                value="Detailed Table",
                interactive=True
            )
            product_btn = gr.Button("Forecast Demand")
            gr.Markdown("""
            **Note on Negative Values:** Negative quantities in the forecast, indicate a **prediction of returned orders** for that specific product during that period.
            """)
            product_table = gr.DataFrame(label="Forecast Data Table", visible=True)
            product_total_output = gr.Number(label="Total Forecasted Quantity for Period", precision=2, visible=False)
            product_plot = gr.Plot(label="Demand Forecast Plot")
            gr.Markdown("This graph illustrates the daily demand forecast for the selected product, showing the predicted trend and confidence intervals.")
            product_components = gr.Plot(label="Demand Forecast Components")
            gr.Markdown("This graph breaks down the product demand forecast into its underlying components: trend, weekly seasonality, and yearly seasonality.")
            product_btn.click(
                fn=predict_product_demand,
                inputs=[product_selector, product_days, product_output_choice],
                outputs=[product_table, product_total_output, product_plot, product_components]
            )

        # TAB 3: Customer Lifetime Value 
        with gr.Tab("Customer Lifetime Value (CLV) Predictor"):
            gr.Markdown("### 💰 Predict Customer Lifetime Value")
            clv_periods = gr.Slider(
                minimum=1, maximum=24, value=12, step=1,
                label="Number of Future Months to Predict CLV For:"
            )
            clv_results_count = gr.Slider(
                minimum=1, maximum=100, value=20, step=1,
                label="Number of Top Customers to Display (Max 100):"
            )
            clv_customer_id = gr.Number(
                label="Enter a specific Customer ID (optional, leave blank for overall top customers):",
                interactive=True
            )
            gr.Markdown("Example Customer IDs: `17850`, `13047`, `12583`")
            clv_btn = gr.Button("Predict CLV")
            gr.Markdown("""
            **Note on Negative CLV:** A negative Customer Lifetime Value indicates that, based on historical patterns, a customer is predicted to generate a net loss for the business during the specified prediction period (e.g., due to returns outweighing purchases).
            """)
            clv_output_df = gr.DataFrame(
                label="Predicted CLV",
                visible=True,
                headers=["CustomerID", "Predicted CLV (£)"]
            )
            clv_btn.click(
                fn=predict_clv,
                inputs=[clv_periods, clv_results_count, clv_customer_id],
                outputs=[clv_output_df]
            )

        # TAB 4: AI Assistant Chatbot 
        with gr.Tab("AI Assistant Chatbot"):
            gr.Markdown("### 🤖 Ask the AI Assistant")
            gr.Markdown("""
            *Powered by Llama 3.3 (Groq).* Supports: **Total Sales**, **Product Demand**, and **Customer CLV**.
            """)
            
            gr.ChatInterface(
                fn=respond_to_chat,
                chatbot=gr.Chatbot(height=500, type="messages"),
                fill_height=False,
                cache_examples=False,
                type="messages",
                examples=[
                    {"text": "Predict the CLV for customer 17850 over 12 months."},
                    {"text": "What is the total sales forcast for the next month?"},
                    {"text": "Which customers will have the highest CLV for the next 3 months?"},
                    {"text": "Demand forcast for product 23166 for 14 days."}
                ]
            )

    # --- Footer ---
    gr.Markdown("""
    ---
    *Developed with **Python** 🐍, **Pandas** 🐼, **NumPy** 🔢, **Matplotlib** 📊, **Seaborn** 🌊, **Gradio** 🎨 (for UI), **Prophet** 🔮 (for Sales & Demand), **Lifetimes** 💰 (for CLV), **ucimlrepo** 🏗️ (for data fetching), **dill** 🥒 (for model serialization), and **LangChain** 🦜 and **Llama 3.3 on Groq Cloud** ⚡ (for the AI Assistant).*
    """)

    # --- 6. DYNAMIC STATUS LOGIC ---
    # This function checks if the cache is ready and updates the component
    def update_status():
        if CACHED_SUMMARY_DF is not None:
            return "✅ **Status:** Engine Ready (Dataset Cached)"
        else:
            return "⚠️ **Status:** Data Cache Offline. Predictive features may be limited."

    # demo.load triggers the function as soon as the app finishes loading
    demo.load(fn=update_status, outputs=status_display)

# --- Launch App ---
if __name__ == "__main__":
    demo.launch()