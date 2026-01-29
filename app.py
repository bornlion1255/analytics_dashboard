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
    page_title="Корпоративная Аналитика AI + API",
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
# 2. БЕЗОПАСНАЯ АВТОРИЗАЦИЯ (SECRETS)
# ==========================================
def check_password():
    """Проверка пароля через st.secrets"""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### Вход в систему")
            with st.form("credentials"):
                password = st.text_input("Введите пароль доступа", type="password")
                submit_button = st.form_submit_button("Войти")
                
                if submit_button:
                    try:
                        # Берем пароль из файла .streamlit/secrets.toml
                        # Формат в файле: PASSWORD = "ваш_пароль"
                        secret_password = st.secrets["PASSWORD"]
                    except FileNotFoundError:
                        st.error("Не настроены Secrets! Добавьте пароль в настройках приложения.")
                        return False
                    except KeyError:
                        st.error("В secrets.toml не найден ключ 'PASSWORD'.")
                        return False

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
# 3. API CONFIG & MAPPING (ЯДРО)
# ==========================================
API_TOKEN = "cb96240069dfaf99fee34e7bfb1c8b" # Можно тоже убрать в st.secrets["API_TOKEN"]
BASE_URL = "https://api.chat2desk.com/v1"
HEADERS = {"Authorization": API_TOKEN}
TIME_OFFSET = 3  # UTC+3

# Маппинг операторов
DEPARTMENT_MAPPING = {
    # Обновление по запросу
    "Никита Приходько": "Concierge", 
    "Алина Федулова": "Тренер",
    
    # Остальные
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

# Маппинг отделов из Гугл Таблицы
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

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ API ---
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
# 4. ЛОГИКА ЗАГРУЗКИ ДАННЫХ
# ==========================================

# --- 4.1 GOOGLE SHEET ---
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
        
        # Нормализация отделов в таблице
        df['Отдел_Норм'] = df['Отдел'].map(SHEET_DEPT_MAPPING).fillna(df['Отдел'])
        
        df['Час'] = df['Дата'].dt.hour
        return df
    except Exception as e:
        st.error(f"Ошибка Google Sheet: {e}")
        return pd.DataFrame()

# --- 4.2 CHAT2DESK API ---
def fetch_api_stats(date_str):
    """Получает список диалогов за конкретную дату"""
    active_requests = []
    limit = 200
    offset = 0
    
    # Подгружаем имена операторов
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
                rating = row.get('rating_scale_score')
                if rating == 0 or rating == '0': rating = None
                active_requests.append({
                    'req_id': row['request_id'],
                    'client': row.get('client_name', 'No Name'),
                    'rating': rating,
                    'date_str': date_str
                })
            if len(data) < limit: break
            offset += limit
        except: break
    return active_requests

def analyze_dialog_hybrid(item, target_start, target_end):
    """Анализ одного диалога (Точный подсчет + Часы активности)"""
    req_id = item['req_id']
    url = f"{BASE_URL}/requests/{req_id}/messages"
    
    stats = {
        'req_id': req_id,
        'rating': item['rating'],
        'participations': set(),
        'operator_times': [] 
    }

    try:
        r = requests.get(url, headers=HEADERS, params={"limit": 300})
        if r.status_code != 200: return None
        
        msgs = r.json().get('data', [])
        if not msgs: return None
        
        for m in msgs:
            ts = m.get('created')
            if not ts: continue
            
            dt_utc = pd.to_datetime(ts, unit='s')
            dt_local = dt_utc + timedelta(hours=TIME_OFFSET)
            
            msg_type = m.get('type')
            op_id = m.get('operatorID') or m.get('operator_id')
            
            # Фильтр: Исходящее от человека (out)
            if msg_type == 'out' and op_id and op_id != 0 and op_id != 310507:
                if target_start <= dt_local <= target_end:
                    stats['participations'].add(op_id)
                    stats['operator_times'].append({
                        'op_id': op_id,
                        'hour': dt_local.hour
                    })
        
        if not stats['participations']: return None
        return stats
    except: return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_api_data_range(start_date, end_date):
    """Главная функция загрузки API за диапазон"""
    
    all_requests = []
    dates = pd.date_range(start_date, end_date).strftime('%Y-%m-%d').tolist()
    
    if len(dates) > 31:
        st.error("Диапазон дат для API слишком большой (>31 дней).")
        return pd.DataFrame()

    with st.spinner(f"Загрузка диалогов из API ({len(dates)} дн.)..."):
        for d in dates:
            all_requests.extend(fetch_api_stats(d))
    
    if not all_requests:
        return pd.DataFrame()

    target_start = pd.to_datetime(f"{start_date} 00:00:00")
    target_end = pd.to_datetime(f"{end_date} 23:59:59")
    
    processed_data = []
    
    progress_text = "Сканирование активности операторов..."
    my_bar = st.progress(0, text=progress_text)
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(analyze_dialog_hybrid, item, target_start, target_end): item for item in all_requests}
        
        total = len(futures)
        completed = 0
        
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            if completed % 20 == 0:
                 my_bar.progress(min(completed / total, 1.0), text=f"{progress_text} {completed}/{total}")

            if res:
                for op_id in res['participations']:
                    op_name = OPERATORS_MAP_CACHE.get(op_id, f"ID {op_id}")
                    dept = find_department_smart(op_name)
                    
                    op_hours = [x['hour'] for x in res['operator_times'] if x['op_id'] == op_id]
                    
                    processed_data.append({
                        'req_id': res['req_id'],
                        'operator_id': op_id,
                        'Оператор': op_name,
                        'Отдел': dept,
                        'rating': res['rating'],
                        'hours_active': op_hours
                    })
    
    my_bar.empty()
    return pd.DataFrame(processed_data)

# ==========================================
# 5. ОСНОВНОЙ ИНТЕРФЕЙС
# ==========================================

# --- Загрузка Гугл Таблицы ---
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

# --- Фильтрация Гугл Таблицы ---
mask_sheet = (df_sheet['Дата'].dt.date >= start_date) & (df_sheet['Дата'].dt.date <= end_date)
df_sheet_filtered = df_sheet.loc[mask_sheet].copy()

if st.sidebar.button("Выйти"):
    st.session_state["password_correct"] = False
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(f"Данные за {start_date} — {end_date}")

# --- Загрузка API (Точные данные) ---
df_api = load_api_data_range(start_date, end_date)

# ==========================================
# 6. РАСЧЕТ KPI (ГИБРИД)
# ==========================================

# 1. API (Люди)
if not df_api.empty:
    api_human_chats = df_api['req_id'].nunique()
    df_api['rating'] = pd.to_numeric(df_api['rating'], errors='coerce')
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
TOTAL_CHATS_REAL = api_human_chats + count_bot_closed + count_auth

# ==========================================
# 7. ВИЗУАЛИЗАЦИЯ
# ==========================================
st.title(f"Отчетность AI + API ({start_date} — {end_date})")
st.caption("Данные объединяются из Chat2Desk API (работа людей) и Google Sheets (классификация бота)")

tabs = st.tabs(["KPI", "Нагрузка", "Анализ отдела", "Категории", "База данных"])

# --- TAB 1: KPI ---
with tabs[0]:
    st.subheader("Ключевые показатели")
    
    c1, c2, c3, c4 = st.columns(4)
    def pct(x, total): return f"{(x/total*100):.1f}%" if total > 0 else "0%"
    
    c1.metric("Бот (Закрыл успешно)", count_bot_closed, delta=pct(count_bot_closed, TOTAL_CHATS_REAL))
    c2.metric("Бот (Перевел на спеца)", count_bot_transfer, delta_color="inverse")
    c3.metric("Авторизация", count_auth, delta=pct(count_auth, TOTAL_CHATS_REAL))
    c4.metric("ВСЕГО ЧАТОВ", TOTAL_CHATS_REAL, help="API (Люди) + Бот (Закрыл) + Авторизация")
    
    st.divider()
    kc1, kc2 = st.columns(2)
    with kc1:
        st.write("**Эффективность бота (где принимал участие)**")
        involved_total = count_bot_closed + count_bot_transfer
        if involved_total > 0:
            fig1, ax1 = plt.subplots(figsize=(3, 3))
            ax1.pie([count_bot_closed, count_bot_transfer], labels=['Закрыл', 'Перевел'], 
                    autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=90)
            st.pyplot(fig1, use_container_width=False)
        else:
            st.info("Нет данных о работе бота")

    with kc2:
        st.write("**Автоматизация (от общего потока)**")
        auto_count = count_bot_closed + count_auth
        auto_rate = (auto_count / TOTAL_CHATS_REAL) if TOTAL_CHATS_REAL > 0 else 0
        st.progress(auto_rate)
        st.metric("Процент автоматизации", f"{auto_rate*100:.1f}%")

# --- TAB 2: НАГРУЗКА ---
with tabs[1]:
    st.subheader("Нагрузка")
    
    if df_api.empty:
        st.warning("Нет данных API для построения точной нагрузки.")
    else:
        # Фильтруем "Прочих"
        load_df = df_api[~df_api['Отдел'].isin(["Прочие / Не найдено", "Тренер", "System / Bot"])].copy()
        dept_counts = load_df.groupby('Отдел')['req_id'].nunique().sort_values(ascending=False).reset_index(name='Кол-во')
        
        lc1, lc2 = st.columns([1, 3])
        with lc1:
            st.write("По отделам (Человеческие чаты)")
            st.dataframe(dept_counts, hide_index=True, use_container_width=True)
            
        with lc2:
            st.write("Тепловая карта: Активность отделов (API)")
            heatmap_data = []
            for _, row in load_df.iterrows():
                dept = row['Отдел']
                for h in row['hours_active']:
                    heatmap_data.append({'Отдел': dept, 'Час': h})
            
            if heatmap_data:
                hm_df = pd.DataFrame(heatmap_data)
                hm_matrix = hm_df.groupby(['Отдел', 'Час']).size().unstack(fill_value=0).reindex(columns=range(24), fill_value=0)
                hm_matrix['Sum'] = hm_matrix.sum(axis=1)
                hm_matrix = hm_matrix.sort_values('Sum', ascending=False).drop(columns='Sum')
                
                fig, ax = plt.subplots(figsize=(10, len(hm_matrix)*0.8 + 1))
                sns.heatmap(hm_matrix, annot=True, fmt="d", cmap="YlOrRd", cbar=False, ax=ax)
                st.pyplot(fig)
            else:
                st.info("Нет данных по часам")

    st.divider()
    st.subheader("Тематика обращений (Google Sheets)")
    topics_df = df_sheet_filtered[~is_auth].copy()
    topics_df = topics_df[topics_df['Тип обращения'] != '-']
    topics_df = topics_df[topics_df['Тип обращения'] != 'nan']
    
    if not topics_df.empty:
        top_topics = topics_df['Тип обращения'].value_counts().head(15).index
        topics_hm = topics_df[topics_df['Тип обращения'].isin(top_topics)]
        
        hm_topic_data = topics_hm.groupby(['Тип обращения', 'Час']).size().unstack(fill_value=0).reindex(columns=range(24), fill_value=0)
        hm_topic_data['Sum'] = hm_topic_data.sum(axis=1)
        hm_topic_data = hm_topic_data.sort_values('Sum', ascending=False).drop(columns='Sum')
        
        fig2, ax2 = plt.subplots(figsize=(12, len(hm_topic_data)*0.6 + 1))
        sns.heatmap(hm_topic_data, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax2)
        st.pyplot(fig2)

# --- TAB 3: АНАЛИЗ ОТДЕЛА ---
with tabs[2]:
    st.subheader("Анализ отдела (Сверка API vs Темы)")
    
    if not df_api.empty:
        api_depts = sorted(df_api[~df_api['Отдел'].isin(["Прочие / Не найдено", "Тренер", "System / Bot"])]['Отдел'].unique())
    else:
        api_depts = []
        
    sel_dept = st.selectbox("Выберите отдел", api_depts)
    
    if sel_dept and not df_api.empty:
        api_dept_df = df_api[df_api['Отдел'] == sel_dept]
        total_api_chats = api_dept_df['req_id'].nunique()
        
        sheet_dept_df = df_sheet_filtered[df_sheet_filtered['Отдел_Норм'] == sel_dept].copy()
        
        sheet_dept_df['Тип обращения'] = sheet_dept_df['Тип обращения'].replace(['-', 'nan'], 'Не размечено')
        cat_counts = sheet_dept_df['Тип обращения'].value_counts().reset_index()
        cat_counts.columns = ['Категория', 'Кол-во']
        
        total_categorized = cat_counts['Кол-во'].sum()
        unknown_count = total_api_chats - total_categorized
        
        st.metric(f"Всего чатов в отделе {sel_dept} (API)", total_api_chats)
        
        if unknown_count > 0:
            new_row = pd.DataFrame([{'Категория': '❓ Неизвестные / Без категории', 'Кол-во': unknown_count}])
            cat_counts = pd.concat([cat_counts, new_row], ignore_index=True)
        
        cat_counts['Доля'] = (cat_counts['Кол-во'] / cat_counts['Кол-во'].sum() * 100).map('{:.1f}%'.format)
        
        st.write("Распределение по тематикам:")
        st.dataframe(cat_counts, use_container_width=True, hide_index=True)
        
        with st.expander(f"Сотрудники отдела {sel_dept}"):
            spec_stats = api_dept_df.groupby('Оператор').agg(
                Чатов=('req_id', 'nunique'),
                Ср_Оценка=('rating', 'mean')
            ).sort_values('Чатов', ascending=False).reset_index()
            spec_stats['Ср_Оценка'] = spec_stats['Ср_Оценка'].fillna(0).round(2)
            st.dataframe(spec_stats, use_container_width=True)

# --- TAB 4: CATEGORIES ---
with tabs[3]:
    st.subheader("Категории (Бот)")
    ai_df = df_sheet_filtered[df_sheet_filtered['Статус'].isin(['Закрыл', 'Перевод'])].copy()
    if not ai_df.empty:
        stats = ai_df.groupby('Тип обращения')['Статус'].value_counts().unstack(fill_value=0)
        for c in ['Закрыл', 'Перевод']: 
            if c not in stats.columns: stats[c] = 0
        stats['Total'] = stats['Закрыл'] + stats['Перевод']
        stats['Бот(✓)'] = (stats['Закрыл']/stats['Total']*100).map('{:.1f}%'.format)
        stats['Бот(→)'] = (stats['Перевод']/stats['Total']*100).map('{:.1f}%'.format)
        
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
            return "\n".join(res) if res else "• Другая"
        
        stats['Причины'] = stats.apply(fmt_r, axis=1)
        final = stats[['Total', 'Бот(✓)', 'Бот(→)', 'Причины']].sort_values('Total', ascending=False).reset_index()
        st.dataframe(final, use_container_width=True, hide_index=True, height=600, column_config={"Причины": st.column_config.TextColumn(width="medium")})
    else:
        st.info("Нет данных о работе бота")

# --- TAB 5: RAW DATA ---
with tabs[4]:
    st.subheader("Сырые данные")
    st.write("Google Sheets Filtered")
    st.dataframe(df_sheet_filtered, use_container_width=True, height=400)
    st.write("API Processed Data")
    st.dataframe(df_api, use_container_width=True, height=400)