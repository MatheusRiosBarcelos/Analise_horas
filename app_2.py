import streamlit as st
import pandas as pd
import plotly.express as px


st.set_page_config(layout="wide")

ordens = pd.read_csv('ordens (4).csv', sep = ',')
pedidos = pd.read_csv('pedidos (1).csv', sep = ',')
ordens['ordem'] = ordens['ordem'].fillna(0)
ordens['data_ini'] = ordens['data_ini'].fillna(0)
ordens['ordem'] = ordens['ordem'].astype(int)
ordens.loc[:, 'Datetime_ini'] = pd.to_datetime(ordens['data_ini'] + ' ' + ordens['hora_ini'], format = 'mixed', errors='coerce')
ordens.loc[:,'Datetime_fim'] = pd.to_datetime(ordens['data_fim'] + ' ' + ordens['hora_fim'], format = 'mixed', errors='coerce')
ordens.loc[:, 'delta_time_seconds'] = (ordens['Datetime_fim'] - ordens['Datetime_ini']).dt.total_seconds()
ordens.loc[:, 'delta_time_hours'] = ordens['delta_time_seconds'] / 3600

estacao = st.sidebar.selectbox("Estação", ordens["estacao"].sort_values().unique(), index= 0,placeholder ='Escolha uma opção')
st.header(f"Análise Hora de Trabalho Mensal {estacao}")

ordens["data_ini"] = pd.to_datetime(ordens["data_ini"], format = 'mixed', errors='coerce')
ordens=ordens.sort_values("data_ini")
ordens["Ano"] = ordens["data_ini"].dt.year.astype('Int64') 
ordens["Mes"] = ordens["data_ini"].dt.month.astype('Int64')
target_year = st.sidebar.selectbox("Ano", ordens["Ano"].sort_values().unique(), index= 1 ,placeholder ='Escolha uma opção')
target_month = st.sidebar.selectbox("Mês", ordens["Mes"].sort_values().unique(), index= 0,placeholder ='Escolha uma opção')

new_df = ordens[ordens['estacao'] == estacao]
df_filtrado = new_df[new_df['Datetime_ini'].dt.month == target_month]
df_filtrado_2 = new_df[new_df['Datetime_ini'].dt.month == (target_month-1)]
hora_esperada_de_trabalho = 160
num_entries = df_filtrado.shape[0]
total_de_horas = round(df_filtrado['delta_time_hours'].sum(), 1)
total_de_horas_delta = round(df_filtrado_2['delta_time_hours'].sum(), 1)
percent_horas = round((total_de_horas/hora_esperada_de_trabalho) * 100, 1)
percent_horas_delta = round((total_de_horas_delta/hora_esperada_de_trabalho) * 100, 1)
media = round(total_de_horas/num_entries,1)
delta_1 = round(percent_horas - percent_horas_delta, 1)
col1, col2, col3 = st.columns(3)
col1.metric(f"Total de Horas da Máquina em {target_month}-{target_year}", f"{total_de_horas}H")
col2.metric('Eficiência (%)', f'{percent_horas}%', delta_1)
col3.metric("Média", f"{media}H")

target_pv = st.sidebar.selectbox("PV", pedidos["pedido"].sort_values().unique(),placeholder ='Escolha uma opção')
st.header(f'Análise Hora de Trabalho PV {target_pv}')

col4, col5 = st.columns(2)

# target_pv = st.sidebar.selectbox("PV", pedidos["pedido"].sort_values().unique(),placeholder ='Escolha uma opção')

descricao = pedidos.loc[pedidos['pedido'] == target_pv]
descricao = descricao.reset_index(drop=True)

descricao = descricao.iloc[0,14]

pedido = pedidos[pedidos['pedido'] == target_pv]
filtro_df = pedido['ordem']
ordem = ordens[ordens['ordem'].isin(filtro_df)]
soma_por_estacao = ordem.groupby('estacao')['delta_time_hours'].sum().reset_index().round(2)
soma_por_estacao.rename(columns={'delta_time_hours': 'Tempo de uso (h)'}, inplace=True)
soma_por_estacao.rename(columns={'estacao': 'Estação de Trabalho'}, inplace=True)
total_de_horas_pedido = round(ordem['delta_time_hours'].sum(),2)

fig = px.pie(soma_por_estacao, values='Tempo de uso (h)', names='Estação de Trabalho', title='Proporção de Tempo de Uso por Máquina em Cada Pedido', width=800, height=500)
fig.update_layout(title_yref='container',title_xanchor = 'center',title_x = 0.43, title_y = 0.95, legend=dict(font=dict(size=18)),font=dict(size=20), title_font=dict(size=20))
col4.plotly_chart(fig, use_container_width=True)

nova_linha = {'Estação de Trabalho': 'Total', 'Tempo de uso (h)': total_de_horas_pedido}
soma_por_estacao = pd.concat([soma_por_estacao, pd.DataFrame([nova_linha])], ignore_index=True)
with col5:
    st.markdown(f"<h1 style='font-size: 20px;'>Tabela de Horas por Estação no PV {target_pv}</h1>", unsafe_allow_html=True)

col5.dataframe(soma_por_estacao, height= 250, width= 500,hide_index=True)

st.markdown(f"<h1 style='text-align: left;'>{descricao}</h1>", unsafe_allow_html=True)








