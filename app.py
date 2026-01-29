import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. КОНФИГУРАЦИЯ И СТИЛИ
# ==========================================
st.set_page_config(
    page_title="Корпоративная Аналитика (Hybrid)",
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
# 2. НАСТРОЙКИ API И МАППИНГ
# ==========================================
API_TOKEN = "cb96240069dfaf99fee34e7bfb1c8b"
BASE_URL = "https://api.chat2desk.com/v1"
HEADERS = {"Authorization": API_TOKEN}
TIME_OFFSET = 3  # UTC+3

# --- Маппинг операторов ---
DEPARTMENT_MAPPING = {
    "Никита Приходько": "Concierge", 
    "Алина Федулова": "Тренер",
    "Илья Аврамов": "Appointment",
    "Виктория Суворова": "Appointment",
    "Кирилл Минаев": "Appointment",
    "Мария Попова": "Без отдела",
    "Станислав Басов": "Claims",
    "Милена Говорова": "Без отдела",
    "Надежда Смирнова": "Сопровождение",
    "Ирина Вережан": "Claims",
    "Наталья Половникова": "Claims",
    "Администратор": "Без отдела",
    "Владимир Асатрян": "Без отдела",
    "Екатерина Ермакова": "Без отдела",
    "Константин Гетман": "SMM",
    "Екатерина Анисимова": "Без отдела",
    "Оля Трущелева": "Без отдела",
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
    "Ксения Бухонина": "Сопровождение",
    "Елизавета Давыденко": "Сопровождение",
    "Екатерина Кондратьева": "Сопровождение",
    "Ксения Гаврилова": "Claims",
    "Снежана Ефимова": "Сопровождение",
    "Анастасия Карпеева": "Claims",
    "Кристина Любина": "Сопровождение",
    "Наталья Серебрякова": "Сопровождение",
    "Константин Клишин": "Claims",
    "Наталья Баландина": "Claims",
    "Даниил Гусев": "Appointment",
    "Анна Власенкова": "SMM",
    "Регина Арендт": "Сопровождение",
    "Екатерина Щукина": "Сопровождение",
    "Ксения Кривко": "Claims",
    "Вероника Софронова": "SMM",
    "Юрий Кобелев": "Claims",
    "Арина Прохорова": "SMM"
}

# --- Маппинг отделов из Гугл Таблицы ---
SHEET_DEPT_MAPPING = {
    'Cleaner_Payments': 'Сопровождение',
    'Penalty': 'Сопровождение',
    'Operations': 'Сопровождение',
    'Storage': 'Сопровождение',
    'Concierge': 'Concierge',
    'SMM': 'SMM',
    'Deep_support': 'Deep_support',
    'Appointment': 'Appointment',
    'Claims': 'Claims'
}

OPERATORS_MAP_CACHE = {310507: "Бот AI", 0: "Система"}

# --- Вспомогательные функции ---
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
    return "Прочие / Не найдено"

# ==========================================
# 3. АВТОРИЗАЦИЯ
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### Вход в систему")
            with st.form("credentials"):
                password = st.text_input("Введите пароль", type="password")
                submit_button = st.form_submit_button("Войти")
                
                if submit_button:
                    try:
                        secret_password = st.secrets["PASSWORD"]
                    except (FileNotFoundError, KeyError):
                        secret_password = "admin" 
                    
                    if password == secret_password:
                        st.session_state["password_correct"] = True
                        st.rerun()
                    else:
                        st.error("Неверный пароль")
        return False
    return True

if not check_password():
    st.stop()

# ==========================================
# 4. ЗАГРУЗКА ДАННЫХ (ГИБРИД)
# ==========================================

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
        
        # Применяем маппинг сразу
        df['Отдел_Норм'] = df['Отдел'].map(SHEET_DEPT_MAPPING).fillna(df['Отдел'])
        df['Час'] = df['Дата'].dt.hour
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки таблицы: {e}")
        return pd.DataFrame()

def fetch_api_stats_for_date(date_str):
    active_requests = []
    limit = 200
    offset = 0
    
    # Ленивая загрузка операторов
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
    """
    КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Корректная обработка ответа API (список vs словарь)
    """
    req_id = item['req_id']
    url = f"{BASE_URL}/requests/{req_id}/messages"
    
    result = {
        'req_id': req_id,
        'participations': set(), 
        'active_hours': []       
    }

    try:
        r = requests.get(url, headers=HEADERS, params={"limit": 300})
        
        # --- ФИКС НАЧАЛО ---
        # API может вернуть список напрямую ИЛИ словарь с ключом 'data'
        json_data = r.json()
        if isinstance(json_data, list):
            msgs = json_data
        else:
            msgs = json_data.get('data', [])
        # --- ФИКС КОНЕЦ ---
        
        if not msgs: return None
        
        for m in msgs:
            ts = m.get('created')
            if not ts: continue
            
            dt_utc = pd.to_datetime(ts, unit='s')
            dt_local = dt_utc + timedelta(hours=TIME_OFFSET)
            
            msg_type = m.get('type')
            op_id = m.get('operatorID') or m.get('operator_id')
            
            # Фильтр: Исходящее от человека
            if msg_type == 'out' and op_id and op_id != 0 and op_id != 310507:
                # Проверка даты
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
    all_requests = []
    dates = pd.date_range(start_date, end_date).strftime('%Y-%m-%d').tolist()
    
    if len(dates) > 31:
        st.error("Слишком большой диапазон дат (>31 дней).")
        return pd.DataFrame(), pd.DataFrame()

    with st.spinner(f"Загрузка списков диалогов API ({len(dates)} дн.)..."):
        for d in dates:
            all_requests.extend(fetch_api_stats_for_date(d))
    
    if not all_requests:
        return pd.DataFrame(), pd.DataFrame()

    target_start = pd.to_datetime(f"{start_date} 00:00:00")
    target_end = pd.to_datetime(f"{end_date} 23:59:59")
    
    chat_data = []    
    heatmap_data = [] 
    
    progress_bar = st.progress(0, text="Сканирование сообщений API...")
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
                for op_id in res['participations']:
                    op_name = OPERATORS_MAP_CACHE.get(op_id, f"ID {op_id}")
                    dept = find_department_smart(op_name)
                    
                    chat_data.append({
                        'req_id': res['req_id'],
                        'Operator': op_name,
                        'Department': dept
                    })
                
                for entry in res['active_hours']:
                    heatmap_data.append({
                        'Department': entry['dept'],
                        'Hour': entry['hour']
                    })
    
    progress_bar.empty()
    return pd.DataFrame(chat_data), pd.DataFrame(heatmap_data)

# ==========================================
# 5. ОСНОВНОЙ ИНТЕРФЕЙС
# ==========================================

# --- Загрузка Таблицы ---
df_sheet = load_sheet_data()
if df_sheet.empty:
    st.stop()

# --- Фильтры ---
st.sidebar.title("Фильтры")
min_date = df_sheet['Дата'].min().date()
max_date = df_sheet['Дата'].max().date()
date_range = st.sidebar.date_input("Период", value=(max_date, max_date), min_value=min_date, max_value=max_date)

if len(date_range) != 2:
    st.warning("Выберите диапазон.")
    st.stop()

start_date, end_date = date_range

mask_sheet = (df_sheet['Дата'].dt.date >= start_date) & (df_sheet['Дата'].dt.date <= end_date)
df_sheet_filtered = df_sheet.loc[mask_sheet].copy()

if st.sidebar.button("Выйти"):
    st.session_state["password_correct"] = False
    st.rerun()

# --- Загрузка API ---
df_api_chats, df_api_heatmap = load_api_data_range(start_date, end_date)

# ==========================================
# 6. РАСЧЕТ KPI
# ==========================================

# 1. API (Люди)
if not df_api_chats.empty:
    api_human_chats = df_api_chats['req_id'].nunique()
else:
    api_human_chats = 0

# 2. Таблица (Бот и Авторизация)
is_auth = (
    df_sheet_filtered['Тип обращения'].str.contains('Авторизация', case=False, na=False) | 
    df_sheet_filtered['Статус'].str.contains('Авторизация', case=False, na=False)
)
count_auth = len(df_sheet_filtered[is_auth])

is_bot_closed = (df_sheet_filtered['Статус'] == 'Закрыл') & (~is_auth)
count_bot_closed = len(df_sheet_filtered[is_bot_closed])

count_bot_transfer = len(df_sheet_filtered[df_sheet_filtered['Статус'] == 'Перевод'])

# 3. ИТОГ
TOTAL_HYBRID_CHATS = api_human_chats + count_bot_closed + count_auth

# ==========================================
# 7. DASHBOARD
# ==========================================
st.title(f"AI Report ({start_date} — {end_date})")

tabs = st.tabs(["KPI", "Нагрузка", "Анализ отдела", "Категории", "Сырые данные"])

# --- TAB 1: KPI ---
with tabs[0]:
    st.subheader("Ключевые показатели")
    
    c1, c2, c3, c4 = st.columns(4)
    
    def pct(x, total): return f"{(x/total*100):.1f}%" if total > 0 else "0%"
    
    c1.metric("Бот (Закрыл успешно)", count_bot_closed, delta=pct(count_bot_closed, TOTAL_HYBRID_CHATS))
    c2.metric("Бот (Перевел на спеца)", count_bot_transfer, delta_color="inverse")
    c3.metric("Авторизация", count_auth, delta=pct(count_auth, TOTAL_HYBRID_CHATS))
    c4.metric("ВСЕГО ЧАТОВ", TOTAL_HYBRID_CHATS, help="API (Люди) + Таблица (Бот + Авт)")
    
    st.divider()
    kc1, kc2 = st.columns(2)
    
    with kc1:
        st.write("**Эффективность бота (где принимал участие)**")
        involved = count_bot_closed + count_bot_transfer
        if involved > 0:
            fig1, ax1 = plt.subplots(figsize=(3, 3))
            ax1.pie([count_bot_closed, count_bot_transfer], labels=['Закрыл', 'Перевел'], 
                    autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=90)
            st.pyplot(fig1, use_container_width=False)
        else:
            st.info("Нет данных о работе бота.")

    with kc2:
        st.write("**Автоматизация (от общего потока)**")
        # Формула: (Бот + Авт) / ВСЕГО
        automated_volume = count_bot_closed + count_auth
        auto_rate = (automated_volume / TOTAL_HYBRID_CHATS) if TOTAL_HYBRID_CHATS > 0 else 0
        
        st.progress(auto_rate)
        st.metric("Процент автоматизации", f"{auto_rate*100:.1f}%")

# --- TAB 2: НАГРУЗКА ---
with tabs[1]:
    st.subheader("Нагрузка")
    
    if df_api_chats.empty:
        st.warning("Данные API загружаются или отсутствуют.")
    else:
        # 1. Таблица (API)
        valid_api_depts = df_api_chats[~df_api_chats['Department'].isin(['Тренер', 'Прочие / Не найдено'])]
        dept_counts = valid_api_depts.groupby('Department')['req_id'].nunique().sort_values(ascending=False).reset_index(name='Кол-во')
        
        col_L, col_R = st.columns([1, 3])
        
        with col_L:
            st.write("Чаты по отделам (API)")
            st.dataframe(dept_counts, hide_index=True, use_container_width=True)
        
        with col_R:
            st.write("Тепловая карта: Активность отделов (API)")
            if not df_api_heatmap.empty:
                hm_matrix = df_api_heatmap.groupby(['Department', 'Hour']).size().unstack(fill_value=0).reindex(columns=range(24), fill_value=0)
                hm_matrix['Total'] = hm_matrix.sum(axis=1)
                hm_matrix = hm_matrix.sort_values('Total', ascending=False).drop(columns='Total')
                
                fig, ax = plt.subplots(figsize=(10, len(hm_matrix)*0.6 + 1.5))
                sns.heatmap(hm_matrix, annot=True, fmt="d", cmap="YlOrRd", cbar=False, ax=ax)
                st.pyplot(fig)
            else:
                st.info("Нет данных по времени.")

    st.divider()
    st.subheader("Тематика по времени (Google Sheets)")
    
    topics_df = df_sheet_filtered[~is_auth].copy()
    topics_df = topics_df[~topics_df['Тип обращения'].isin(['-', 'nan'])]
    
    if not topics_df.empty:
        top_topics = topics_df['Тип обращения'].value_counts().head(15).index
        topics_hm_data = topics_df[topics_df['Тип обращения'].isin(top_topics)]
        
        hm_topic = topics_hm_data.groupby(['Тип обращения', 'Час']).size().unstack(fill_value=0).reindex(columns=range(24), fill_value=0)
        hm_topic['Sum'] = hm_topic.sum(axis=1)
        hm_topic = hm_topic.sort_values('Sum', ascending=False).drop(columns='Sum')
        
        fig2, ax2 = plt.subplots(figsize=(12, len(hm_topic)*0.6 + 1))
        sns.heatmap(hm_topic, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax2)
        st.pyplot(fig2)
    else:
        st.info("Нет данных по тематикам.")

# --- TAB 3: АНАЛИЗ ОТДЕЛА ---
with tabs[2]:
    st.subheader("Детальный анализ отдела")
    
    if not df_api_chats.empty:
        available_depts = sorted(df_api_chats[~df_api_chats['Department'].isin(['Тренер', 'Прочие / Не найдено'])]['Department'].unique())
    else:
        available_depts = []
        
    selected_dept = st.selectbox("Выберите отдел", available_depts)
    
    if selected_dept and not df_api_chats.empty:
        # A. Всего чатов (API)
        api_dept_data = df_api_chats[df_api_chats['Department'] == selected_dept]
        total_api_count = api_dept_data['req_id'].nunique()
        
        # B. Тематики (Таблица)
        sheet_dept_data = df_sheet_filtered[df_sheet_filtered['Отдел_Норм'] == selected_dept].copy()
        
        sheet_dept_data['Тип обращения'] = sheet_dept_data['Тип обращения'].replace(['-', 'nan'], 'Не размечено')
        topic_counts = sheet_dept_data['Тип обращения'].value_counts().reset_index()
        topic_counts.columns = ['Категория', 'Кол-во']
        
        total_categorized = topic_counts['Кол-во'].sum()
        
        # C. Неизвестные
        unknown_count = total_api_count - total_categorized
        
        st.metric(f"Всего чатов в {selected_dept} (API)", total_api_count)
        
        if unknown_count > 0:
            new_row = pd.DataFrame([{'Категория': '❓ Неизвестные / Без категории', 'Кол-во': unknown_count}])
            topic_counts = pd.concat([topic_counts, new_row], ignore_index=True)
        elif unknown_count < 0:
            st.warning(f"В таблице размечено на {abs(unknown_count)} чатов больше, чем найдено в API (возможно, закрыты ботом).")
        
        topic_counts['Доля'] = (topic_counts['Кол-во'] / topic_counts['Кол-во'].sum() * 100).map('{:.1f}%'.format)
        
        st.write(f"Разбивка по темам (Всего: {topic_counts['Кол-во'].sum()})")
        st.dataframe(topic_counts, use_container_width=True, hide_index=True)
        
        with st.expander("Сотрудники отдела"):
            emp_stats = api_dept_data.groupby('Operator')['req_id'].nunique().sort_values(ascending=False).reset_index(name='Чатов')
            st.dataframe(emp_stats, use_container_width=True)

# --- TAB 4: КАТЕГОРИИ ---
with tabs[3]:
    st.subheader("Категории бота")
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
        r_counts = pd.DataFrame() if tr_