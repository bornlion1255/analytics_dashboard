import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. КОНФИГУРАЦИЯ И НАСТРОЙКИ
# ==========================================
st.set_page_config(page_title="SLA Dashboard Hybrid", layout="wide")

# CSS для таблиц
st.markdown("""
    <style>
    .stDataFrame td { white-space: pre-wrap !important; vertical-align: top !important; }
    </style>
""", unsafe_allow_html=True)

# API CHAT2DESK
API_TOKEN = "cb96240069dfaf99fee34e7bfb1c8b"
BASE_URL = "https://api.chat2desk.com/v1"
HEADERS = {"Authorization": API_TOKEN}
MAX_WORKERS = 20
TIME_OFFSET = 3  # UTC+3

# GOOGLE SHEET
SHEET_ID = "123VexBVR3y9o6f6pnJKJAWV47PBpT0uhnCL9JSGwIBo"
GID = "465082032"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

# СПРАВОЧНИК ОТДЕЛОВ (Ваш оригинальный + Группировка)
OPERATORS_MAP = {310507: "Бот AI", 0: "Система"}
DEPARTMENT_MAPPING = {
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
    "Никита Приходько": "Concierge", # ИЗМЕНЕНО ПО ЗАПРОСУ
    "Анна Власенкова": "SMM",
    "Регина Арендт": "Сопровождение",
    "Екатерина Щукина": "Сопровождение",
    "Ксения Кривко": "Claims",
    "Вероника Софронова": "SMM",
    "Юрий Кобелев": "Claims",
    "Арина Прохорова": "SMM"
}

# Группировка микро-отделов в Сопровождение
CUSTOM_GROUPING = {
    "Cleaner_Payments": "Сопровождение",
    "Penalty": "Сопровождение",
    "Operations": "Сопровождение",
    "Storage": "Сопровождение"
}

# ==========================================
# 2. БЕЗОПАСНОСТЬ (ПАРОЛЬ)
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if not st.session_state["password_correct"]:
        with st.form("credentials"):
            password = st.text_input("Введите пароль доступа", type="password")
            submit = st.form_submit_button("Войти")
            if submit:
                # Пароль берем из Secrets или хардкод
                secret_pass = st.secrets.get("PASSWORD", "Qleanlife1!")
                if password == secret_pass:
                    st.session_state["password_correct"] = True
                    st.rerun()
                else:
                    st.error("Неверный пароль")
        return False
    return True

if not check_password():
    st.stop()

# ==========================================
# 3. ФУНКЦИИ СКРИПТА (API)
# ==========================================
def normalize_text(text):
    if not text: return ""
    return str(text).lower().strip().replace("ё", "е")

def find_department_smart(api_name_full):
    # Сначала проверяем точное совпадение
    clean_api = normalize_text(api_name_full)
    for name, dept in DEPARTMENT_MAPPING.items():
        if normalize_text(name) == clean_api:
            return dept
    # Потом частичное
    for name_key, dept in DEPARTMENT_MAPPING.items():
        parts = normalize_text(name_key).split()
        if not parts: continue
        if all(part in clean_api for part in parts):
            return dept
    return "Не определен"

def process_single_dialog(item, target_start, target_end):
    """Анализ одного диалога (из вашего скрипта)"""
    req_id = item['req_id']
    try:
        r = requests.get(f"{BASE_URL}/requests/{req_id}/messages", headers=HEADERS, params={"limit": 300})
        if r.status_code != 200: return None
        json_data = r.json()
        msgs = json_data if isinstance(json_data, list) else json_data.get('data', [])
        msgs.sort(key=lambda x: x.get('created', 0))
        
        participations = set()
        msg_times = [] # Для тепловой карты (время ответов)
        
        for m in msgs:
            ts = m.get('created')
            if not ts: continue
            dt_utc = pd.to_datetime(ts, unit='s')
            dt_local = dt_utc + timedelta(hours=TIME_OFFSET)
            
            msg_type = m.get('type')
            op_id = m.get('operatorID') or m.get('operator_id')
            
            # Логика из вашего скрипта: Если пишет ОПЕРАТОР и попадает в дату
            if msg_type == 'out' and op_id and op_id != 0 and op_id != 310507:
                 if target_start <= dt_local <= target_end:
                     participations.add(op_id)
                     msg_times.append(dt_local.hour)
                     
        return {
            'req_id': req_id,
            'participations': list(participations),
            'hours': list(set(msg_times)) # В какие часы отвечали
        }
    except:
        return None

@st.cache_data(ttl=3600)
def load_api_data(date_str):
    """Основная функция загрузки данных через API"""
    target_start = pd.to_datetime(f"{date_str} 00:00:00")
    target_end = pd.to_datetime(f"{date_str} 23:59:59")
    
    # 1. Загрузка имен операторов
    try:
        r = requests.get(f"{BASE_URL}/operators", headers=HEADERS, params={"limit": 1000})
        for op in r.json().get('data', []):
            name = f"{op.get('first_name', '')} {op.get('last_name', '')}".strip()
            if not name: name = op.get('email', str(op['id']))
            OPERATORS_MAP[op['id']] = name
    except: pass

    # 2. Получение списка обращений
    active_requests = []
    limit = 200
    offset = 0
    # Ограничитель на всякий случай, чтобы не зависнуть навечно
    while offset < 5000:
        try:
            params = {"report": "request_stats", "date": date_str, "limit": limit, "offset": offset}
            r = requests.get(f"{BASE_URL}/statistics", headers=HEADERS, params=params)
            data = r.json().get('data', [])
            if not data: break
            
            for row in data:
                active_requests.append({'req_id': row['request_id']})
            
            if len(data) < limit: break
            offset += limit
        except: break
        
    # 3. Детальный анализ (Multithreading)
    final_rows = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(active_requests)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_dialog, item, target_start, target_end): item for item in active_requests}
        
        for future in as_completed(futures):
            res = future.result()
            if res and res['participations']:
                # Разворачиваем: одна строка = один оператор в чате
                for op_id in res['participations']:
                    op_name = OPERATORS_MAP.get(op_id, f"ID {op_id}")
                    dept = find_department_smart(op_name)
                    
                    # Группировка микро-отделов
                    if dept in CUSTOM_GROUPING:
                        dept = CUSTOM_GROUPING[dept]
                    
                    # Фильтр "Тренер" (Исключаем)
                    if dept == "Тренер":
                        continue
                        
                    for h in res['hours']:
                         final_rows.append({
                            'req_id': res['req_id'],
                            'operator_id': op_id,
                            'Оператор': op_name,
                            'Отдел': dept,
                            'Час': h
                        })
            
            completed += 1
            if total > 0:
                progress_bar.progress(min(completed / total, 1.0))
            status_text.text(f"Обработано API: {completed}/{total}")
            
    progress_bar.empty()
    status_text.empty()
    
    df = pd.DataFrame(final_rows)
    return df

# ==========================================
# 4. ЗАГРУЗКА GOOGLE SHEET (ДЛЯ ТЕМ И БОТА)
# ==========================================
@st.cache_data(ttl=600)
def load_gsheet_data():
    try:
        df = pd.read_csv(SHEET_URL)
        df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Дата'])
        for col in ['Отдел', 'Статус', 'Тип обращения']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        df['Час'] = df['Дата'].dt.hour
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки Google Sheet: {e}")
        return pd.DataFrame()

# ==========================================
# 5. ОСНОВНАЯ ЛОГИКА
# ==========================================
st.sidebar.title("Фильтры")

# Загружаем GSheet для дат
df_gsheet_all = load_gsheet_data()
if not df_gsheet_all.empty:
    min_date = df_gsheet_all['Дата'].min().date()
    max_date = df_gsheet_all['Дата'].max().date()
    # По умолчанию сегодня или макс дата
    default_date = max_date
else:
    default_date = datetime.now().date()

selected_date = st.sidebar.date_input("Выберите дату для анализа", value=default_date)

# Кнопка запуска анализа
if st.sidebar.button("Запустить анализ (API)"):
    st.session_state['run_analysis'] = True
    st.cache_data.clear() # Чистим кэш чтобы получить свежие данные

if 'run_analysis' not in st.session_state:
    st.info("👈 Выберите дату и нажмите 'Запустить анализ', чтобы собрать точные данные из Chat2Desk.")
    st.stop()

# --- ЗАГРУЗКА ДАННЫХ ---
date_str = selected_date.strftime("%Y-%m-%d")

# 1. API Данные (Люди)
df_api = load_api_data(date_str)

# 2. GSheet Данные (Бот, Авторизация, Темы) - фильтруем по выбранной дате
mask_gsheet = (df_gsheet_all['Дата'].dt.date == selected_date)
df_gsheet = df_gsheet_all[mask_gsheet].copy()

# ==========================================
# 6. РАСЧЕТЫ KPI
# ==========================================

# А. Считаем людей (уникальные чаты из API)
# Один чат может быть у нескольких операторов, для Total берем уникальные req_id
if not df_api.empty:
    count_human_chats = df_api['req_id'].nunique()
else:
    count_human_chats = 0

# Б. Считаем Бота (из GSheet)
# Статус = Закрыл ИЛИ Тип = Авторизация
# Нужно быть аккуратным, чтобы не посчитать дважды, если и там и там
# Но обычно Авторизация это отдельный кейс. Будем считать как вы написали.

# 1. Бот закрыл (по Статусу)
bot_closed_mask = (df_gsheet['Статус'].str.lower() == 'закрыл')
count_bot_closed = len(df_gsheet[bot_closed_mask])

# 2. Авторизация (по Типу)
auth_mask = (df_gsheet['Тип обращения'].str.contains('Авторизация пройдена', case=False, na=False))
count_auth = len(df_gsheet[auth_mask])

# В. ИТОГО ВСЕГО
total_chats_day = count_human_chats + count_bot_closed + count_auth

st.title(f"📊 Отчетность SLA ({date_str})")

tabs = st.tabs(["KPI", "Нагрузка", "Анализ отдела", "Категории", "База данных"])

# --- TAB 1: KPI ---
with tabs[0]:
    st.subheader("Сводная статистика за день")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Всего чатов (Human + Bot + Auth)", total_chats_day)
    col2.metric("Обработано людьми (API)", count_human_chats)
    col3.metric("Закрыто ботом", count_bot_closed)
    col4.metric("Авторизация", count_auth)
    
    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Эффективность бота, где он принимал участие")
        # База: Общее кол-во чатов. Часть: Бот закрыл
        if total_chats_day > 0:
            bot_share = count_bot_closed / total_chats_day
            st.metric("Доля закрытия ботом", f"{bot_share*100:.1f}%")
            
            # Пирог
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            labels = ['Люди', 'Бот (Закрыл)', 'Авторизация']
            sizes = [count_human_chats, count_bot_closed, count_auth]
            colors = ['#66b3ff', '#ff9999', '#99ff99']
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            st.pyplot(fig1, use_container_width=False)
        else:
            st.write("Нет данных.")

# --- TAB 2: НАГРУЗКА ---
with tabs[1]:
    st.subheader("1. Нагрузка по отделам (Данные скрипта)")
    # Считаем уникальные чаты на отдел
    if not df_api.empty:
        # Один req_id может быть у нескольких операторов одного отдела, считаем уникальные req_id внутри отдела
        dept_load = df_api.groupby('Отдел')['req_id'].nunique().sort_values(ascending=False).reset_index()
        dept_load.columns = ['Отдел', 'Кол-во чатов']
        
        c_table, c_heat = st.columns([1, 2])
        with c_table:
            st.dataframe(dept_load, hide_index=True, use_container_width=True)
            
        with c_heat:
            st.write("**Тепловая карта: Отдел vs Час (Нагрузка)**")
            # Группируем: Отдел, Час -> Count unique req_id
            hm_data = df_api.groupby(['Отдел', 'Час'])['req_id'].nunique().unstack(fill_value=0)
            hm_data = hm_data.reindex(columns=range(24), fill_value=0)
            
            fig, ax = plt.subplots(figsize=(10, len(hm_data)*0.5+2))
            sns.heatmap(hm_data, annot=True, fmt="d", cmap="YlOrRd", cbar=False, ax=ax)
            ax.set_xlabel("Час дня")
            st.pyplot(fig)
            
    else:
        st.warning("Нет данных API для нагрузки.")
        
    st.divider()
    
    st.subheader("2. Тематика обращений по времени (Данные GSheet)")
    # Берем типы обращений из GSheet (исключая прочерки и авторизацию для чистоты, или оставляем все)
    # Обычно "-" убирают
    topics_df = df_gsheet[~df_gsheet['Тип обращения'].isin(['-', 'Авторизация пройдена'])].copy()
    
    if not topics_df.empty:
        # Топ 15 тематик для читаемости карты
        top_topics = topics_df['Тип обращения'].value_counts().nlargest(15).index
        topics_df_top = topics_df[topics_df['Тип обращения'].isin(top_topics)]
        
        hm_topic = topics_df_top.groupby(['Тип обращения', 'Час']).size().unstack(fill_value=0)
        hm_topic = hm_topic.reindex(columns=range(24), fill_value=0)
        
        # Сортировка по общему кол-ву
        hm_topic['Total'] = hm_topic.sum(axis=1)
        hm_topic = hm_topic.sort_values('Total', ascending=False).drop(columns='Total')
        
        fig2, ax2 = plt.subplots(figsize=(12, len(hm_topic)*0.6+2))
        sns.heatmap(hm_topic, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax2)
        ax2.set_xlabel("Час дня")
        st.pyplot(fig2)
    else:
        st.write("Нет данных по тематикам в Google Sheet.")

# --- TAB 3: АНАЛИЗ ОТДЕЛА ---
with tabs[2]:
    st.subheader("Детальный анализ по отделу (Сравнение API и Темы)")
    
    if not df_api.empty:
        all_depts = sorted(df_api['Отдел'].unique())
        selected_dept = st.selectbox("Выберите отдел", all_depts)
        
        if selected_dept:
            # 1. Данные API (Точное число чатов)
            # Фильтруем API по отделу и считаем уникальные
            dept_api_data = df_api[df_api['Отдел'] == selected_dept]
            total_chats_api = dept_api_data['req_id'].nunique()
            
            # 2. Данные GSheet (Темы)
            # Фильтруем GSheet по отделу
            # ВНИМАНИЕ: В GSheet названия отделов могут отличаться. 
            # Предполагаем, что в GSheet они уже нормализованы или используем маппинг.
            # Если в GSheet названия старые, фильтрация может не сработать. 
            # Пробуем найти прямое совпадение.
            dept_gsheet_data = df_gsheet[df_gsheet['Отдел'] == selected_dept].copy()
            
            # Считаем категории
            cat_counts = dept_gsheet_data['Тип обращения'].value_counts().reset_index()
            cat_counts.columns = ['Категория', 'Кол-во']
            
            # Убираем "-" из подсчета известных тем
            known_topics_count = dept_gsheet_data[dept_gsheet_data['Тип обращения'] != '-'].shape[0]
            
            # 3. Вычисляем "Неизвестные"
            # Логика: Всего (API) - Известные (GSheet)
            # Если в GSheet тем больше чем в API (ошибки учета), то неизвестных 0
            unknown_count = max(0, total_chats_api - known_topics_count)
            
            # Формируем итоговую таблицу
            st.write(f"📊 Всего чатов в отделе **{selected_dept}** (по данным скрипта): **{total_chats_api}**")
            
            final_stats = []
            
            # Добавляем известные темы
            for _, row in cat_counts.iterrows():
                cat = row['Категория']
                cnt = row['Кол-во']
                if cat == '-': continue # Пропускаем прочерки, заменим их на calculated unknown
                final_stats.append({'Категория': cat, 'Кол-во': cnt})
            
            # Добавляем расчетные неизвестные
            if unknown_count > 0:
                final_stats.append({'Категория': 'Неизвестные обращения (разница)', 'Кол-во': unknown_count})
            
            df_res = pd.DataFrame(final_stats)
            if not df_res.empty:
                df_res['Доля'] = (df_res['Кол-во'] / total_chats_api * 100).map('{:.1f}%'.format)
                df_res = df_res.sort_values('Кол-во', ascending=False)
                st.dataframe(df_res, use_container_width=True, hide_index=True)
            else:
                st.write("Нет данных по категориям.")

# --- TAB 4: КАТЕГОРИИ (BOT) ---
with tabs[3]:
    st.subheader("Детализация по категориям (Бот)")
    # Тут логика остается старая, чисто по GSheet, так как скрипт не знает причин перевода
    
    ai_df = df_gsheet[df_gsheet['Статус'].isin(['Закрыл', 'Перевод'])].copy()
    
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
        st.write("Нет данных о работе бота в GSheet.")

# --- TAB 5: БАЗА ДАННЫХ ---
with tabs[4]:
    st.subheader("Сырые данные (Google Sheet)")
    st.dataframe(df_gsheet, use_container_width=True)
    
    if not df_api.empty:
        st.subheader("Сырые данные (Обработанный API скрипт)")
        st.dataframe(df_api, use_container_width=True)