import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. CONFIGURATION & STYLES
# ==========================================
st.set_page_config(
    page_title="Corporate Analytics (Hybrid)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stDataFrame td {
        white-space: pre-wrap !important;
        vertical-align: top !important;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. API CONFIGURATION & MAPPINGS
# ==========================================
API_TOKEN = "cb96240069dfaf99fee34e7bfb1c8b"  # Ideally, move this to st.secrets
BASE_URL = "https://api.chat2desk.com/v1"
HEADERS = {"Authorization": API_TOKEN}
TIME_OFFSET = 3  # UTC+3

# --- API Operator Mapping (Clean Depts) ---
DEPARTMENT_MAPPING = {
    # Updates based on your request
    "Никита Приходько": "Concierge", 
    "Алина Федулова": "Trainer", 
    
    # Existing List
    "Илья Аврамов": "Appointment",
    "Виктория Суворова": "Appointment",
    "Кирилл Минаев": "Appointment",
    "Мария Попова": "No Dept",
    "Станислав Басов": "Claims",
    "Милена Говорова": "No Dept",
    "Надежда Смирнова": "Support",
    "Ирина Вережан": "Claims",
    "Наталья Половникова": "Claims",
    "Администратор": "No Dept",
    "Владимир Асатрян": "No Dept",
    "Екатерина Ермакова": "No Dept",
    "Константин Гетман": "SMM",
    "Екатерина Анисимова": "No Dept",
    "Оля Трущелева": "No Dept",
    "Алина Новикова": "SMM",
    "Иван Савицкий": "SMM",
    "Анастасия Ванян": "SALE",
    "Павел Новиков": "SMM",
    "Александра Шаповал": "SMM",
    "Георгий Астапов": "Deep_support",
    "Елена Панова": "Deep_support",
    "Татьяна Сошникова": "SMM",
    "Виктория Вороняк": "SMM",
    "Анна Чернышова": "SMM",
    "Алина Ребрина": "Claims",
    "Алена Воронина": "Claims",
    "Ксения Бухонина": "Support",
    "Елизавета Давыденко": "Support",
    "Екатерина Кондратьева": "Support",
    "Ксения Гаврилова": "Claims",
    "Снежана Ефимова": "Support",
    "Анастасия Карпеева": "Claims",
    "Кристина Любина": "Support",
    "Наталья Серебрякова": "Support",
    "Константин Клишин": "Claims",
    "Наталья Баландина": "Claims",
    "Даниил Гусев": "Appointment",
    "Анна Власенкова": "SMM",
    "Регина Арендт": "Support",
    "Екатерина Щукина": "Support",
    "Ксения Кривко": "Claims",
    "Вероника Софронова": "SMM",
    "Юрий Кобелев": "Claims",
    "Арина Прохорова": "SMM"
}

# --- Google Sheet Dept Mapping (Merging Micro-depts) ---
SHEET_DEPT_MAPPING = {
    'Cleaner_Payments': 'Support',
    'Penalty': 'Support',
    'Operations': 'Support',
    'Storage': 'Support',
    'Сопровождение': 'Support',
    'Concierge': 'Concierge',
    'SMM': 'SMM',
    'Deep_support': 'Deep_support',
    'Appointment': 'Appointment',
    'Claims': 'Claims'
}

OPERATORS_MAP_CACHE = {310507: "Bot AI", 0: "System"}

# --- Helpers ---
def normalize_text(text):
    if not text: return ""
    return str(text).lower().strip().replace("ё", "е")

def find_department_smart(api_name_full):
    clean_api = normalize_text(api_name_full)
    for name, dept in DEPARTMENT_MAPPING.items():
        if normalize_text(name) == clean_api:
            return dept
    for name_key, dept in DEPARTMENT_MAPPING.items():
        parts = normalize_text(name_key).split()
        if not parts: continue
        if all(part in clean_api for part in parts):
            return dept
    return "Other / Not Found"

# ==========================================
# 3. AUTHENTICATION
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### Login")
            with st.form("credentials"):
                password = st.text_input("Enter Password", type="password")
                submit_button = st.form_submit_button("Sign In")
                
                if submit_button:
                    try:
                        secret_password = st.secrets["PASSWORD"]
                    except (FileNotFoundError, KeyError):
                        # Fallback for testing if secrets not set up
                        secret_password = "admin" 
                    
                    if password == secret_password:
                        st.session_state["password_correct"] = True
                        st.rerun()
                    else:
                        st.error("Invalid password")
        return False
    return True

if not check_password():
    st.stop()

# ==========================================
# 4. DATA LOADING (HYBRID)
# ==========================================

# --- 4.1 Load Google Sheet Data ---
@st.cache_data(ttl=600)
def load_sheet_data():
    sheet_id = "123VexBVR3y9o6f6pnJKJAWV47PBpT0uhnCL9JSGwIBo"
    gid = "465082032"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    
    try:
        df = pd.read_csv(url)
        df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Дата'])
        
        for col in ['Отдел', 'Статус', 'Тип обращения', 'Причина перевода']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Apply Mapping to Sheet Data immediately
        df['Original_Dept'] = df['Отдел']
        df['Отдел'] = df['Отдел'].map(SHEET_DEPT_MAPPING).fillna(df['Отдел'])
        
        df['Час'] = df['Дата'].dt.hour
        return df
    except Exception as e:
        st.error(f"Error loading Google Sheet: {e}")
        return pd.DataFrame()

# --- 4.2 Load API Data (The heavy lifting) ---
def fetch_api_stats_for_date(date_str):
    """Fetches list of dialogs for a specific date"""
    active_requests = []
    limit = 200
    offset = 0
    
    # Lazy load operator names if cache is empty
    if len(OPERATORS_MAP_CACHE) < 5:
        try:
            r = requests.get(f"{BASE_URL}/operators", headers=HEADERS, params={"limit": 500})
            for op in r.json().get('data', []):
                name = f"{op.get('first_name', '')} {op.get('last_name', '')}".strip()
                if not name: name = op.get('email', str(op['id']))
                OPERATORS_MAP_CACHE[op['id']] = name
        except: pass

    while True:
        try:
            params = {"report": "request_stats", "date": date_str, "limit": limit, "offset": offset}
            r = requests.get(f"{BASE_URL}/statistics", headers=HEADERS, params=params)
            data = r.json().get('data', [])
            if not data: break
            for row in data:
                active_requests.append({
                    'req_id': row['request_id'],
                    'client': row.get('client_name', 'No Name')
                })
            if len(data) < limit: break
            offset += limit
        except: break
    return active_requests

def analyze_dialog_for_load(item, target_start, target_end):
    """Analyzes a single dialog to find operator participation and timestamps"""
    req_id = item['req_id']
    url = f"{BASE_URL}/requests/{req_id}/messages"
    
    result = {
        'req_id': req_id,
        'participations': set(), # Which operators participated
        'active_hours': []       # List of (Operator, Hour) tuples for Heatmap
    }

    try:
        r = requests.get(url, headers=HEADERS, params={"limit": 300})
        if r.status_code != 200: return None
        
        msgs = r.json().get('data', [])
        
        for m in msgs:
            ts = m.get('created')
            if not ts: continue
            
            dt_utc = pd.to_datetime(ts, unit='s')
            dt_local = dt_utc + timedelta(hours=TIME_OFFSET)
            
            msg_type = m.get('type')
            op_id = m.get('operatorID') or m.get('operator_id')
            
            # Check if it's a human operator outgoing message
            if msg_type == 'out' and op_id and op_id != 0 and op_id != 310507:
                # Check date boundaries
                if target_start <= dt_local <= target_end:
                    result['participations'].add(op_id)
                    result['active_hours'].append({
                        'op_id': op_id,
                        'hour': dt_local.hour,
                        'dept': find_department_smart(OPERATORS_MAP_CACHE.get(op_id, ""))
                    })
        
        if not result['participations']: return None
        return result
    except: return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_api_data_range(start_date, end_date):
    """Loads and processes API data for the selected date range"""
    
    # 1. Get all dialog lists for the range
    all_requests = []
    dates = pd.date_range(start_date, end_date).strftime('%Y-%m-%d').tolist()
    
    if len(dates) > 31:
        st.error("Date range too large for API loading (>31 days). Please reduce range.")
        return pd.DataFrame(), pd.DataFrame()

    with st.spinner(f"Fetching dialog lists from API for {len(dates)} days..."):
        for d in dates:
            all_requests.extend(fetch_api_stats_for_date(d))
    
    if not all_requests:
        return pd.DataFrame(), pd.DataFrame()

    # 2. Analyze messages for participation & load
    target_start = pd.to_datetime(f"{start_date} 00:00:00")
    target_end = pd.to_datetime(f"{end_date} 23:59:59")
    
    chat_data = []    # One row per chat per operator (for counts)
    heatmap_data = [] # One row per message/hour (for heatmap)
    
    progress_bar = st.progress(0, text="Scanning chat history for accurate load data...")
    total_reqs = len(all_requests)
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(analyze_dialog_for_load, item, target_start, target_end): item for item in all_requests}
        
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            if completed % 25 == 0:
                progress_bar.progress(min(completed / total_reqs, 1.0))

            if res:
                # Process Participation (for KPI & Dept Analysis)
                for op_id in res['participations']:
                    op_name = OPERATORS_MAP_CACHE.get(op_id, f"ID {op_id}")
                    dept = find_department_smart(op_name)
                    
                    chat_data.append({
                        'req_id': res['req_id'],
                        'Operator': op_name,
                        'Department': dept
                    })
                
                # Process Heatmap Data
                for entry in res['active_hours']:
                    heatmap_data.append({
                        'Department': entry['dept'],
                        'Hour': entry['hour']
                    })
    
    progress_bar.empty()
    return pd.DataFrame(chat_data), pd.DataFrame(heatmap_data)

# ==========================================
# 5. MAIN APP LOGIC
# ==========================================

# --- Load Google Sheet ---
df_sheet = load_sheet_data()
if df_sheet.empty:
    st.stop()

# --- Filters ---
st.sidebar.title("Filters")
min_date = df_sheet['Дата'].min().date()
max_date = df_sheet['Дата'].max().date()
date_range = st.sidebar.date_input("Period", value=(max_date, max_date), min_value=min_date, max_value=max_date)

if len(date_range) != 2:
    st.warning("Select a date range.")
    st.stop()

start_date, end_date = date_range

# Filter Sheet Data
mask_sheet = (df_sheet['Дата'].dt.date >= start_date) & (df_sheet['Дата'].dt.date <= end_date)
df_sheet_filtered = df_sheet.loc[mask_sheet].copy()

if st.sidebar.button("Logout"):
    st.session_state["password_correct"] = False
    st.rerun()

st.sidebar.info(f"Data for: {start_date} — {end_date}")

# --- Load API Data (Hybrid Integration) ---
df_api_chats, df_api_heatmap = load_api_data_range(start_date, end_date)

# ==========================================
# 6. CALCULATIONS (THE HYBRID MATH)
# ==========================================

# 1. API Human Chats (Total Unique Dialogs handled by humans)
if not df_api_chats.empty:
    # Filter out Trainers/System if needed for "Total"
    # Keeping it simple: Total Human Chats
    total_human_chats = df_api_chats['req_id'].nunique()
else:
    total_human_chats = 0

# 2. Sheet Data (Bot & Auth)
is_auth = (
    df_sheet_filtered['Тип обращения'].str.contains('Авторизация', case=False, na=False) | 
    df_sheet_filtered['Статус'].str.contains('Авторизация', case=False, na=False)
)
count_auth = len(df_sheet_filtered[is_auth])

# Bot Closed (Status='Закрыл', excluding Auth)
is_bot_closed = (df_sheet_filtered['Статус'] == 'Закрыл') & (~is_auth)
count_bot_closed = len(df_sheet_filtered[is_bot_closed])

# Bot Transfer (For efficiency chart)
count_bot_transfer = len(df_sheet_filtered[df_sheet_filtered['Статус'] == 'Перевод'])

# 3. GRAND TOTAL
# Sum of: Real Human Chats (API) + Bot Closed (Sheet) + Auth (Sheet)
TOTAL_HYBRID_CHATS = total_human_chats + count_bot_closed + count_auth

# ==========================================
# 7. DASHBOARD UI
# ==========================================
st.title(f"AI Report ({start_date} — {end_date})")

tabs = st.tabs(["KPI", "Workload", "Dept Analysis", "Categories", "Raw Data"])

# --- TAB 1: KPI ---
with tabs[0]:
    st.subheader("Key Metrics")
    
    c1, c2, c3, c4 = st.columns(4)
    
    def pct(x, total): return f"{(x/total*100):.1f}%" if total > 0 else "0%"
    
    c1.metric("Bot (Closed)", count_bot_closed, delta=pct(count_bot_closed, TOTAL_HYBRID_CHATS))
    c2.metric("Bot (Transferred)", count_bot_transfer, delta_color="inverse")
    c3.metric("Authorization", count_auth, delta=pct(count_auth, TOTAL_HYBRID_CHATS))
    c4.metric("TOTAL CHATS", TOTAL_HYBRID_CHATS, help="API (Humans) + Sheet (Bot Closed + Auth)")
    
    st.divider()
    kc1, kc2 = st.columns(2)
    
    with kc1:
        st.write("**Bot Efficiency (where participated)**")
        involved = count_bot_closed + count_bot_transfer
        if involved > 0:
            fig1, ax1 = plt.subplots(figsize=(3, 3))
            ax1.pie([count_bot_closed, count_bot_transfer], labels=['Closed', 'Transferred'], 
                    autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=90)
            st.pyplot(fig1, use_container_width=False)
        else:
            st.info("No bot data.")

    with kc2:
        st.write("**Automation Rate (of Total Flow)**")
        # Formula: (Bot Closed + Auth) / Total Hybrid Chats
        automated_volume = count_bot_closed + count_auth
        auto_rate = (automated_volume / TOTAL_HYBRID_CHATS) if TOTAL_HYBRID_CHATS > 0 else 0
        
        st.progress(auto_rate)
        st.metric("Percentage", f"{auto_rate*100:.1f}%")

# --- TAB 2: WORKLOAD ---
with tabs[1]:
    st.subheader("Workload Analysis")
    
    if df_api_chats.empty:
        st.warning("API data is loading or empty. Please wait or check date range.")
    else:
        # 1. Count Table (Source: API)
        # Exclude 'Other/Not Found' and 'Trainer' for the table if you want
        valid_api_depts = df_api_chats[~df_api_chats['Department'].isin(['Trainer', 'Other / Not Found'])]
        dept_counts = valid_api_depts.groupby('Department')['req_id'].nunique().sort_values(ascending=False).reset_index(name='Count')
        
        col_L, col_R = st.columns([1, 3])
        
        with col_L:
            st.write("Chats by Dept (API)")
            st.dataframe(dept_counts, hide_index=True, use_container_width=True)
        
        with col_R:
            st.write("Heatmap: Dept Activity (API Timestamps)")
            if not df_api_heatmap.empty:
                # Pivot for Heatmap
                hm_matrix = df_api_heatmap.groupby(['Department', 'Hour']).size().unstack(fill_value=0).reindex(columns=range(24), fill_value=0)
                
                # Sort by total activity
                hm_matrix['Total'] = hm_matrix.sum(axis=1)
                hm_matrix = hm_matrix.sort_values('Total', ascending=False).drop(columns='Total')
                
                fig, ax = plt.subplots(figsize=(10, len(hm_matrix)*0.6 + 1.5))
                sns.heatmap(hm_matrix, annot=True, fmt="d", cmap="YlOrRd", cbar=False, ax=ax)
                st.pyplot(fig)
            else:
                st.info("No timestamp data available.")

    st.divider()
    st.subheader("Topics by Time (Source: Google Sheet)")
    
    # 2. Topic Heatmap (Source: Sheet)
    topics_df = df_sheet_filtered[~is_auth].copy()
    topics_df = topics_df[~topics_df['Тип обращения'].isin(['-', 'nan', 'NaT'])]
    
    if not topics_df.empty:
        # Top 15 topics
        top_topics = topics_df['Тип обращения'].value_counts().head(15).index
        topics_hm_data = topics_df[topics_df['Тип обращения'].isin(top_topics)]
        
        hm_topic = topics_hm_data.groupby(['Тип обращения', 'Час']).size().unstack(fill_value=0).reindex(columns=range(24), fill_value=0)
        hm_topic['Sum'] = hm_topic.sum(axis=1)
        hm_topic = hm_topic.sort_values('Sum', ascending=False).drop(columns='Sum')
        
        fig2, ax2 = plt.subplots(figsize=(12, len(hm_topic)*0.6 + 1))
        sns.heatmap(hm_topic, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax2)
        st.pyplot(fig2)
    else:
        st.info("No topic data in Sheet.")

# --- TAB 3: DEPT ANALYSIS ---
with tabs[2]:
    st.subheader("Department Deep Dive")
    
    # Get List of Departments from API (True Source)
    if not df_api_chats.empty:
        available_depts = sorted(df_api_chats[~df_api_chats['Department'].isin(['Trainer', 'Other / Not Found'])]['Department'].unique())
    else:
        available_depts = []
        
    selected_dept = st.selectbox("Select Department", available_depts)
    
    if selected_dept and not df_api_chats.empty:
        # A. Total Chats (API)
        api_dept_data = df_api_chats[df_api_chats['Department'] == selected_dept]
        total_api_count = api_dept_data['req_id'].nunique()
        
        # B. Topics (Sheet)
        # Filter Sheet by the MAPPED department
        sheet_dept_data = df_sheet_filtered[df_sheet_filtered['Отдел'] == selected_dept].copy()
        
        # Count Topics
        sheet_dept_data['Тип обращения'] = sheet_dept_data['Тип обращения'].replace(['-', 'nan'], 'Uncategorized')
        topic_counts = sheet_dept_data['Тип обращения'].value_counts().reset_index()
        topic_counts.columns = ['Category', 'Count']
        
        total_categorized = topic_counts['Count'].sum()
        
        # C. Calculate Unknown
        # If API says 350 chats, but Sheet has topics for 250, then 100 are Unknown
        unknown_count = total_api_count - total_categorized
        
        # Display
        st.metric(f"Total Chats for {selected_dept} (API)", total_api_count)
        
        if unknown_count > 0:
            # Add "Unknown" row
            new_row = pd.DataFrame([{'Category': '❓ Unknown / Uncategorized', 'Count': unknown_count}])
            topic_counts = pd.concat([topic_counts, new_row], ignore_index=True)
        elif unknown_count < 0:
            st.warning(f"Note: Sheet has {abs(unknown_count)} more categorized chats than API found. (Likely Bot closed chats)")
        
        # Calculate %
        topic_counts['Share'] = (topic_counts['Count'] / topic_counts['Count'].sum() * 100).map('{:.1f}%'.format)
        
        st.write(f"Topic Breakdown (Total: {topic_counts['Count'].sum()})")
        st.dataframe(topic_counts, use_container_width=True, hide_index=True)
        
        # D. Employee Breakdown (From API)
        with st.expander("Employees in this Department"):
            emp_stats = api_dept_data.groupby('Operator')['req_id'].nunique().sort_values(ascending=False).reset_index(name='Chats')
            st.dataframe(emp_stats, use_container_width=True)

# --- TAB 4: CATEGORIES ---
with tabs[3]:
    st.subheader("Bot Categories")
    # Existing Logic
    ai_df = df_sheet_filtered[df_sheet_filtered['Статус'].isin(['Закрыл', 'Перевод'])].copy()
    if not ai_df.empty:
        stats = ai_df.groupby('Тип обращения')['Статус'].value_counts().unstack(fill_value=0)
        for c in ['Закрыл', 'Перевод']: 
            if c not in stats.columns: stats[c] = 0
        stats['Total'] = stats['Закрыл'] + stats['Перевод']
        stats['Bot(✓)'] = (stats['Закрыл']/stats['Total']*100).map('{:.1f}%'.format)
        stats['Bot(→)'] = (stats['Перевод']/stats['Total']*100).map('{:.1f}%'.format)
        
        tr_df = ai_df[ai_df['Статус'] == 'Перевод']
        reasons = ['Требует сценарий', 'Не знает ответ', 'Лимит сообщений']
        r_counts = pd.DataFrame() if tr_df.empty else tr_df.groupby('Тип обращения')['Причина перевода'].value_counts().unstack(fill_value=0)
        for r in reasons: 
            if r not in r_counts.columns: r_counts[r] = 0
        stats = stats.join(r_counts, how='left').fillna(0)
        
        def fmt_r(row):
            tot = row['Перевод']
            if tot == 0: return "-"
            res = [f"• {r}: {(row.get(r,0)/tot*100):.0f}%" for r in reasons if row.get(r,0) > 0]
            return "\n".join(res) if res else "• Other"
        
        stats['Reasons'] = stats.apply(fmt_r, axis=1)
        final = stats[['Total', 'Bot(✓)', 'Bot(→)', 'Reasons']].sort_values('Total', ascending=False).reset_index()
        st.dataframe(final, use_container_width=True, hide_index=True, height=600, column_config={"Reasons": st.column_config.TextColumn(width="medium")})
    else:
        st.info("No bot data.")

# --- TAB 5: RAW DATA ---
with tabs[4]:
    st.subheader("Raw Data")
    st.write("Google Sheet Data (Filtered)")
    st.dataframe(df_sheet_filtered, use_container_width=True)
    st.write("API Data (Processed)")
    st.dataframe(df_api_chats, use_container_width=True)