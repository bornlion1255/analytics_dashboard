import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. НАСТРОЙКИ СТРАНИЦЫ
# ==========================================
st.set_page_config(
    page_title="AI Аналитика",
    page_icon="📊",
    layout="wide"
)

# Немного CSS магии, чтобы таблицы были красивые
st.markdown("""
    <style>
    .stDataFrame td {
        white-space: pre-wrap !important;
        vertical-align: top !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Дашборд аналитики AI и Нагрузки")

# ==========================================
# 2. ЗАГРУЗКА ДАННЫХ
# ==========================================
@st.cache_data(ttl=600) # Кэш на 10 минут
def load_data():
    # Ваша ссылка на гугл таблицу
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
        st.error(f"Ошибка загрузки данных: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# ==========================================
# 3. БОКОВАЯ ПАНЕЛЬ (ФИЛЬТРЫ)
# ==========================================
st.sidebar.header("⚙️ Настройки")

# Выбор дат
min_date = df['Дата'].min().date()
max_date = df['Дата'].max().date()

date_range = st.sidebar.date_input(
    "Выберите период",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) != 2:
    st.warning("Выберите дату начала и окончания.")
    st.stop()

start_date, end_date = date_range

# Фильтр по дате
mask = (df['Дата'].dt.date >= start_date) & (df['Дата'].dt.date <= end_date)
df_filtered = df.loc[mask].copy()

st.sidebar.info(f"Записей: {len(df_filtered)}")

# ==========================================
# 4. РАСЧЕТЫ (ЛОГИКА)
# ==========================================

# 1. Авторизация
is_auth = (
    df_filtered['Тип обращения'].str.contains('Авторизация', case=False, na=False) | 
    df_filtered['Статус'].str.contains('Авторизация', case=False, na=False)
)

# 2. Исключенные отделы
exclude_depts = ['-', 'Меню клинер Deep', 'Меню Курьера']
is_excluded_dept = df_filtered['Отдел'].isin(exclude_depts)

# 3. Валидный датафрейм (Главный)
valid_mask = (~is_excluded_dept) | (is_auth)
valid_df = df_filtered[valid_mask].copy()

# СЧЕТЧИКИ
auth_mask_final = (
    valid_df['Тип обращения'].str.contains('Авторизация', case=False, na=False) | 
    valid_df['Статус'].str.contains('Авторизация', case=False, na=False)
)
count_auth = len(valid_df[auth_mask_final])
count_bot_closed = len(valid_df[(valid_df['Статус'] == 'Закрыл') & (~auth_mask_final)])
count_bot_transfer = len(valid_df[valid_df['Статус'] == 'Перевод'])
total_valid = len(valid_df)

# ==========================================
# 5. ИНТЕРФЕЙС
# ==========================================
tab1, tab2, tab3 = st.tabs(["📈 KPI и Эффективность", "🏢 Нагрузка по отделам", "🧩 Категории (Детали)"])

# --- ВКЛАДКА 1: ОБЩИЕ ПОКАЗАТЕЛИ ---
with tab1:
    st.subheader("Общее распределение по обработке")
    
    col1, col2, col3, col4 = st.columns(4)
    
    def pct(x, total): return f"{(x/total*100):.1f}%" if total > 0 else "0%"

    col1.metric("Бот (Успешно)", count_bot_closed, delta=pct(count_bot_closed, total_valid))
    col2.metric("Бот (На спеца)", count_bot_transfer, delta_color="inverse", delta=pct(count_bot_transfer, total_valid))
    col3.metric("Авторизация", count_auth, delta=pct(count_auth, total_valid))
    col4.metric("ВСЕГО ЗАЯВОК", total_valid)
    
    st.divider()
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.info("🤖 **1.1 Эффективность СТАНДАРТНОГО БОТА** (где он участвовал)")
        bot_participated = count_bot_closed + count_bot_transfer
        
        if bot_participated > 0:
            # Рисуем простой пирог
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            ax1.pie([count_bot_closed, count_bot_transfer], labels=['Закрыл', 'Перевел'], 
                    autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=90)
            st.pyplot(fig1, use_container_width=False)
        else:
            st.write("Нет данных.")

    with c2:
        st.success("⚡ **1.2 Эффективность АВТОМАТИКИ** (Бот + Авторизация)")
        total_auto = count_bot_closed + count_auth
        auto_rate = (total_auto / total_valid) if total_valid > 0 else 0
        
        st.progress(auto_rate)
        st.metric("Процент автоматизации (от всех заявок)", f"{auto_rate*100:.1f}%")
        st.caption(f"Всего закрыто: {total_auto} из {total_valid}")

# --- ВКЛАДКА 2: НАГРУЗКА ---
with tab2:
    st.subheader("Нагрузка по отделам")
    st.caption("Исключены: Бот, прочерки, технические меню")
    
    depts_to_hide = exclude_depts + ['Бот']
    workload_df = valid_df[~valid_df['Отдел'].isin(depts_to_hide)].copy()
    
    if not workload_df.empty:
        col_table, col_heatmap = st.columns([1, 3])
        
        with col_table:
            st.write("🔢 **Сводка**")
            dept_counts = workload_df['Отдел'].value_counts().reset_index()
            dept_counts.columns = ['Отдел', 'Кол-во']
            st.dataframe(dept_counts, hide_index=True, use_container_width=True)
            
        with col_heatmap:
            st.write("🔥 **Тепловая карта (Часы vs Отделы)**")
            
            heatmap_data = workload_df.groupby(['Отдел', 'Час']).size().unstack(fill_value=0)
            heatmap_data = heatmap_data.reindex(columns=range(24), fill_value=0)
            
            heatmap_data['Total'] = heatmap_data.sum(axis=1)
            heatmap_data = heatmap_data.sort_values('Total', ascending=False).drop(columns='Total')
            
            fig, ax = plt.subplots(figsize=(10, len(heatmap_data) * 0.6 + 1.5))
            sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlOrRd", linewidths=.5, ax=ax, cbar=False)
            ax.set_xlabel("Час дня")
            ax.set_ylabel("")
            st.pyplot(fig)
    else:
        st.warning("Нет данных для анализа нагрузки.")

# --- ВКЛАДКА 3: КАТЕГОРИИ ---
with tab3:
    st.subheader("Детализация по категориям обращений")
    
    ai_df = valid_df[valid_df['Статус'].isin(['Закрыл', 'Перевод'])].copy()
    
    if not ai_df.empty:
        stats = ai_df.groupby('Тип обращения')['Статус'].value_counts().unstack(fill_value=0)
        if 'Закрыл' not in stats.columns: stats['Закрыл'] = 0
        if 'Перевод' not in stats.columns: stats['Перевод'] = 0
        
        stats['Кол-во чатов'] = stats['Закрыл'] + stats['Перевод']
        stats['Бот(✓)'] = (stats['Закрыл'] / stats['Кол-во чатов'] * 100)
        stats['Бот(→)'] = (stats['Перевод'] / stats['Кол-во чатов'] * 100)
        
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
            
        stats['Причина перевода'] = stats.apply(format_reasons, axis=1)
        
        final_cols = ['Кол-во чатов', 'Бот(✓)', 'Бот(→)', 'Причина перевода']
        final_df = stats[final_cols].sort_values('Кол-во чатов', ascending=False).reset_index()
        
        final_df['Бот(✓)'] = final_df['Бот(✓)'].map('{:.1f}%'.format)
        final_df['Бот(→)'] = final_df['Бот(→)'].map('{:.1f}%'.format)
        
        st.dataframe(
            final_df, 
            use_container_width=True,
            column_config={
                "Причина перевода": st.column_config.TextColumn("Причины перевода (от переводов)", width="medium"),
                "Кол-во чатов": st.column_config.NumberColumn("Всего чатов", format="%d")
            },
            hide_index=True,
            height=800
        )
        
    else:
        st.info("Нет данных с участием бота.")