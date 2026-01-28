import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. НАСТРОЙКИ СТРАНИЦЫ
# ==========================================
st.set_page_config(
    page_title="Корпоративная Аналитика",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stDataFrame td {
        white-space: pre-wrap !important;
        vertical-align: top !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. БЕЗОПАСНАЯ АВТОРИЗАЦИЯ
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
                    secret_password = st.secrets["PASSWORD"]
                except FileNotFoundError:
                    st.error("Не настроены Secrets! Добавьте пароль в настройках приложения.")
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
# 3. ЗАГРУЗКА ДАННЫХ
# ==========================================
@st.cache_data(ttl=600)
def load_data():
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
        
        df['Час'] = df['Дата'].dt.hour
        return df
    except Exception as e:
        st.error(f"Ошибка подключения к данным: {e}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

# ==========================================
# 4. ФИЛЬТРЫ
# ==========================================
st.sidebar.title("Фильтры")

min_date = df['Дата'].min().date()
max_date = df['Дата'].max().date()

date_range = st.sidebar.date_input("Период", value=(min_date, max_date), min_value=min_date, max_value=max_date)

if len(date_range) != 2:
    st.warning("Выберите корректный диапазон дат.")
    st.stop()

start_date, end_date = date_range
mask = (df['Дата'].dt.date >= start_date) & (df['Дата'].dt.date <= end_date)
df_filtered = df.loc[mask].copy()

st.sidebar.markdown("---")
st.sidebar.write(f"Записей: {len(df_filtered)}")

if st.sidebar.button("Выйти"):
    st.session_state["password_correct"] = False
    st.rerun()

# ==========================================
# 5. ПОДГОТОВКА ДАННЫХ
# ==========================================
is_auth = (
    df_filtered['Тип обращения'].str.contains('Авторизация', case=False, na=False) | 
    df_filtered['Статус'].str.contains('Авторизация', case=False, na=False)
)
exclude_depts = ['-', 'Меню клинер Deep', 'Меню Курьера']
is_excluded_dept = df_filtered['Отдел'].isin(exclude_depts)
valid_mask = (~is_excluded_dept) | (is_auth)
valid_df = df_filtered[valid_mask].copy()

# ==========================================
# 6. ИНТЕРФЕЙС
# ==========================================
st.title(f"Отчетность AI ({start_date} — {end_date})")
tabs = st.tabs(["KPI", "Нагрузка", "Анализ отдела", "Категории", "База данных"])

# --- TAB 1: KPI ---
with tabs[0]:
    st.subheader("Ключевые показатели")
    auth_mask_final = (valid_df['Тип обращения'].str.contains('Авторизация', case=False, na=False) | valid_df['Статус'].str.contains('Авторизация', case=False, na=False))
    count_auth = len(valid_df[auth_mask_final])
    count_bot_closed = len(valid_df[(valid_df['Статус'] == 'Закрыл') & (~auth_mask_final)])
    count_bot_transfer = len(valid_df[valid_df['Статус'] == 'Перевод'])
    total_valid = len(valid_df)
    
    c1, c2, c3, c4 = st.columns(4)
    def pct(x, total): return f"{(x/total*100):.1f}%" if total > 0 else "0%"
    c1.metric("Бот (Успешно)", count_bot_closed, delta=pct(count_bot_closed, total_valid))
    c2.metric("Бот (На спеца)", count_bot_transfer, delta_color="inverse", delta=pct(count_bot_transfer, total_valid))
    c3.metric("Авторизация", count_auth, delta=pct(count_auth, total_valid))
    c4.metric("Всего", total_valid)
    
    st.divider()
    kc1, kc2 = st.columns(2)
    with kc1:
        st.write("**Эффективность бота**")
        if (count_bot_closed + count_bot_transfer) > 0:
            fig1, ax1 = plt.subplots(figsize=(3, 3))
            ax1.pie([count_bot_closed, count_bot_transfer], labels=['Закрыл', 'Перевел'], autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=90)
            st.pyplot(fig1, use_container_width=False)
    with kc2:
        st.write("**Автоматизация**")
        auto_rate = ((count_bot_closed + count_auth) / total_valid) if total_valid > 0 else 0
        st.progress(auto_rate)
        st.metric("Процент", f"{auto_rate*100:.1f}%")

# --- TAB 2: LOAD ---
with tabs[1]:
    st.subheader("Нагрузка")
    depts_to_hide = exclude_depts + ['Бот']
    load_df = valid_df[~valid_df['Отдел'].isin(depts_to_hide)].copy()
    if not load_df.empty:
        lc1, lc2 = st.columns([1, 3])
        with lc1:
            st.dataframe(load_df['Отдел'].value_counts().reset_index(name='Кол-во'), hide_index=True, use_container_width=True)
        with lc2:
            hm_data = load_df.groupby(['Отдел', 'Час']).size().unstack(fill_value=0).reindex(columns=range(24), fill_value=0)
            hm_data['Total'] = hm_data.sum(axis=1)
            hm_data = hm_data.sort_values('Total', ascending=False).drop(columns='Total')
            fig, ax = plt.subplots(figsize=(10, len(hm_data)*0.6+1.5))
            sns.heatmap(hm_data, annot=True, fmt="d", cmap="YlOrRd", cbar=False, ax=ax)
            st.pyplot(fig)

# --- TAB 3: DEPT ANALYSIS ---
with tabs[2]:
    st.subheader("Анализ отдела")
    all_depts = sorted([d for d in df_filtered['Отдел'].unique() if d not in exclude_depts and d != 'Бот'])
    sel_dept = st.selectbox("Отдел", all_depts)
    if sel_dept:
        d_df = df_filtered[df_filtered['Отдел'] == sel_dept].copy()
        d_df['Тип обращения'] = d_df['Тип обращения'].replace('-', 'Без участия AI, прямая маршрутизация')
        cat_c = d_df['Тип обращения'].value_counts().reset_index()
        cat_c.columns = ['Категория', 'Кол-во']
        cat_c['Доля'] = (cat_c['Кол-во'] / cat_c['Кол-во'].sum() * 100).map('{:.1f}%'.format)
        st.dataframe(cat_c, use_container_width=True, hide_index=True)

# --- TAB 4: CATEGORIES ---
with tabs[3]:
    st.subheader("Категории (Бот)")
    ai_df = valid_df[valid_df['Статус'].isin(['Закрыл', 'Перевод'])].copy()
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

# --- TAB 5: RAW DATA ---
with tabs[4]:
    st.subheader("База данных")
    st.dataframe(df_filtered, use_container_width=True, height=700)