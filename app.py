import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. НАСТРОЙКИ И АВТОРИЗАЦИЯ
# ==========================================
st.set_page_config(
    page_title="Корпоративная Аналитика AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS для строгих таблиц
st.markdown("""
    <style>
    .stDataFrame td {
        white-space: pre-wrap !important;
        vertical-align: top !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- ЛОГИКА ПАРОЛЯ ---
def check_password():
    """Простая проверка пароля"""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.markdown("### Вход в систему")
        password = st.text_input("Введите пароль доступа", type="password")
        
        if st.button("Войти"):
            # === ПАРОЛЬ МЕНЯТЬ ТУТ ===
            if password == "Qleanlife1!": 
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Неверный пароль")
        return False
    return True

if not check_password():
    st.stop()

# ==========================================
# 2. ЗАГРУЗКА ДАННЫХ
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
        
        # Очистка строк
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
# 3. БОКОВАЯ ПАНЕЛЬ
# ==========================================
st.sidebar.title("Фильтры")

# Даты
min_date = df['Дата'].min().date()
max_date = df['Дата'].max().date()

date_range = st.sidebar.date_input(
    "Период",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) != 2:
    st.warning("Выберите корректный диапазон дат.")
    st.stop()

start_date, end_date = date_range
mask = (df['Дата'].dt.date >= start_date) & (df['Дата'].dt.date <= end_date)
df_filtered = df.loc[mask].copy()

st.sidebar.markdown("---")
st.sidebar.write(f"Загружено записей: {len(df_filtered)}")

if st.sidebar.button("Выйти из системы"):
    st.session_state["password_correct"] = False
    st.rerun()

# ==========================================
# 4. ПОДГОТОВКА ДАННЫХ (ОБЩАЯ)
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
# 5. ИНТЕРФЕЙС
# ==========================================
st.title(f"Отчетность AI ({start_date} — {end_date})")

# Создаем 5 вкладок
tabs = st.tabs(["KPI", "Нагрузка", "Анализ отдела", "Категории", "Все данные (Таблица)"])

# --- ВКЛАДКА 1: KPI ---
with tabs[0]:
    st.subheader("Ключевые показатели эффективности")
    
    # Счетчики
    auth_mask_final = (
        valid_df['Тип обращения'].str.contains('Авторизация', case=False, na=False) | 
        valid_df['Статус'].str.contains('Авторизация', case=False, na=False)
    )
    count_auth = len(valid_df[auth_mask_final])
    count_bot_closed = len(valid_df[(valid_df['Статус'] == 'Закрыл') & (~auth_mask_final)])
    count_bot_transfer = len(valid_df[valid_df['Статус'] == 'Перевод'])
    total_valid = len(valid_df)
    
    col1, col2, col3, col4 = st.columns(4)
    def pct(x, total): return f"{(x/total*100):.1f}%" if total > 0 else "0%"

    col1.metric("Бот (Успешно)", count_bot_closed, delta=pct(count_bot_closed, total_valid))
    col2.metric("Бот (На спеца)", count_bot_transfer, delta_color="inverse", delta=pct(count_bot_transfer, total_valid))
    col3.metric("Авторизация", count_auth, delta=pct(count_auth, total_valid))
    col4.metric("Всего заявок", total_valid)
    
    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("Эффективность стандартного бота (где он участвовал)")
        bot_participated = count_bot_closed + count_bot_transfer
        if bot_participated > 0:
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            ax1.pie([count_bot_closed, count_bot_transfer], labels=['Закрыл', 'Перевел'], 
                    autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=90)
            st.pyplot(fig1, use_container_width=False)
    
    with c2:
        st.write("Общая автоматизация (Бот + Авторизация)")
        total_auto = count_bot_closed + count_auth
        auto_rate = (total_auto / total_valid) if total_valid > 0 else 0
        st.progress(auto_rate)
        st.metric("Процент автоматизации", f"{auto_rate*100:.1f}%")

# --- ВКЛАДКА 2: НАГРУЗКА ---
with tabs[1]:
    st.subheader("Распределение нагрузки по отделам")
    
    depts_to_hide = exclude_depts + ['Бот']
    workload_df = valid_df[~valid_df['Отдел'].isin(depts_to_hide)].copy()
    
    if not workload_df.empty:
        col_table, col_heatmap = st.columns([1, 3])
        with col_table:
            dept_counts = workload_df['Отдел'].value_counts().reset_index()
            dept_counts.columns = ['Отдел', 'Кол-во']
            st.dataframe(dept_counts, hide_index=True, use_container_width=True)
            
        with col_heatmap:
            heatmap_data = workload_df.groupby(['Отдел', 'Час']).size().unstack(fill_value=0)
            heatmap_data = heatmap_data.reindex(columns=range(24), fill_value=0)
            heatmap_data['Total'] = heatmap_data.sum(axis=1)
            heatmap_data = heatmap_data.sort_values('Total', ascending=False).drop(columns='Total')
            
            fig, ax = plt.subplots(figsize=(10, len(heatmap_data) * 0.6 + 1.5))
            sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlOrRd", linewidths=.5, ax=ax, cbar=False)
            ax.set_xlabel("Час дня")
            ax.set_ylabel("")
            st.pyplot(fig)

# --- ВКЛАДКА 3: АНАЛИЗ ОТДЕЛА (НОВАЯ) ---
with tabs[2]:
    st.subheader("Детальный анализ по отделу")
    
    # Список отделов (убираем мусор)
    all_depts = sorted([d for d in df_filtered['Отдел'].unique() if d not in exclude_depts and d != 'Бот'])
    
    selected_dept = st.selectbox("Выберите отдел:", all_depts)
    
    if selected_dept:
        # Фильтруем данные по отделу
        dept_df = df_filtered[df_filtered['Отдел'] == selected_dept].copy()
        
        # Меняем "-" на понятный текст
        dept_df['Тип обращения'] = dept_df['Тип обращения'].replace('-', 'Без участия AI, прямая маршрутизация')
        
        # Считаем категории
        cat_counts = dept_df['Тип обращения'].value_counts().reset_index()
        cat_counts.columns = ['Категория обращения', 'Количество']
        cat_counts['Доля'] = (cat_counts['Количество'] / cat_counts['Количество'].sum() * 100).map('{:.1f}%'.format)
        
        st.write(f"Всего обращений в **{selected_dept}**: {len(dept_df)}")
        st.dataframe(cat_counts, use_container_width=True, hide_index=True)

# --- ВКЛАДКА 4: КАТЕГОРИИ ---
with tabs[3]:
    st.subheader("Детализация по категориям (Бот)")
    
    ai_df = valid_df[valid_df['Статус'].isin(['Закрыл', 'Перевод'])].copy()
    
    if not ai_df.empty:
        stats = ai_df.groupby('Тип обращения')['Статус'].value_counts().unstack(fill_value=0)
        if 'Закрыл' not in stats.columns: stats['Закрыл'] = 0
        if 'Перевод' not in stats.columns: stats['Перевод'] = 0
        
        stats['Кол-во'] = stats['Закрыл'] + stats['Перевод']
        stats['Бот(✓)'] = (stats['Закрыл'] / stats['Кол-во'] * 100).map('{:.1f}%'.format)
        stats['Бот(→)'] = (stats['Перевод'] / stats['Кол-во'] * 100).map('{:.1f}%'.format)
        
        transfers = ai_df[ai_df['Статус'] == 'Перевод']
        target_reasons = ['Требует сценарий', 'Не знает ответ', 'Лимит сообщений']
        
        if not transfers.empty:
            reason_counts = transfers.groupby('Тип обращения')['Причина перевода'].value_counts().unstack(fill_value=0)
        else:
            reason_counts = pd.DataFrame()

        for r in target_reasons:
            if r not in reason_counts.columns: reason_counts[r] = 0
            
        stats = stats.join(reason_counts, how='left').fillna(0)
        
        def format_reasons(row):
            total_transfers = row['Перевод']
            if total_transfers == 0: return "-"
            parts = []
            for r in target_reasons:
                count = row.get(r, 0)
                if count > 0:
                    pct = (count / total_transfers * 100)
                    parts.append(f"• {r}: {pct:.0f}%")
            if not parts: return "• Другая причина"
            return "\n".join(parts)
            
        stats['Причины перевода'] = stats.apply(format_reasons, axis=1)
        
        final_df = stats[['Кол-во', 'Бот(✓)', 'Бот(→)', 'Причины перевода']].sort_values('Кол-во', ascending=False).reset_index()
        
        st.dataframe(
            final_df, 
            use_container_width=True,
            column_config={"Причины перевода": st.column_config.TextColumn("Причины перевода", width="medium")},
            hide_index=True,
            height=600
        )

# --- ВКЛАДКА 5: ВСЕ ДАННЫЕ (RAW) ---
with tabs[4]:
    st.subheader("Полная база данных")
    st.markdown("Используйте заголовки столбцов для сортировки и фильтрации (значок 🔍 при наведении).")
    
    # Отображаем таблицу с включенной фильтрацией
    st.dataframe(df_filtered, use_container_width=True, height=700)