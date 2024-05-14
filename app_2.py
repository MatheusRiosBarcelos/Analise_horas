import streamlit as st
import pandas as pd
import plotly.express as px
from datetime  import datetime as dt , timedelta

st.set_page_config(layout="wide")

ordens = pd.read_csv('ordens (4).csv', sep = ',')
pedidos = pd.read_csv('pedidos (1).csv', sep = ',')
ordens['ordem'] = ordens['ordem'].fillna(0)
ordens['data_ini'] = ordens['data_ini'].fillna(0)
ordens['hora_ini'] = ordens['hora_ini'].fillna(0)
ordens['ordem'] = ordens['ordem'].astype(int)
ordens.loc[:, 'Datetime_ini'] = pd.to_datetime(ordens['data_ini'] + ' ' + ordens['hora_ini'], format = 'mixed', errors='coerce')
ordens.loc[:,'Datetime_fim'] = pd.to_datetime(ordens['data_fim'] + ' ' + ordens['hora_fim'], format = 'mixed', errors='coerce')
ordens.loc[:, 'delta_time_seconds'] = (ordens['Datetime_fim'] - ordens['Datetime_ini']).dt.total_seconds()
ordens.loc[:, 'delta_time_hours'] = ordens['delta_time_seconds'] / 3600
ordens.loc[:,'delta_time_min'] = ordens['delta_time_seconds']/60
st.image('logo.png', width= 150)

tab1, tab2, tab3 = st.tabs(["ANÁLISE HORA DE TRABALHO MENSAL", "ANÁLISE HORA DE TRABALHO POR PV", "MÉDIA POR PEÇA"])
with tab1:
    col6, col7, col8 = st.columns(3)
    with col6:
        estacao = st.selectbox("Estação", ordens["estacao"].sort_values().unique(), index= 0,placeholder ='Escolha uma opção')
    ordens["data_ini"] = pd.to_datetime(ordens["data_ini"], format = 'mixed', errors='coerce')
    ordens=ordens.sort_values("data_ini")
    ordens["Ano"] = ordens["data_ini"].dt.year.astype('Int64') 
    ordens["Mes"] = ordens["data_ini"].dt.month.astype('Int64')
    with col7:
        target_month = st.selectbox("Mês", ordens["Mes"].sort_values().unique(), index= 0,placeholder ='Escolha uma opção')
    with col8:
        target_year = st.selectbox("Ano", ordens["Ano"].sort_values().unique(), index= 1 ,placeholder ='Escolha uma opção')
    
    new_df = ordens[ordens['estacao'] == estacao]
    df_filtrado = new_df[new_df['Datetime_ini'].dt.month == target_month]
    df_filtrado_2 = df_filtrado[df_filtrado['Datetime_ini'].dt.year == target_year]
    hora_esperada_de_trabalho = 160
    num_entries = df_filtrado.shape[0]
    total_de_horas = round(df_filtrado_2['delta_time_hours'].sum(), 1)
    percent_horas = round((total_de_horas/hora_esperada_de_trabalho) * 100, 1)
    media = round(total_de_horas/num_entries,1)
    delta_1 = round(percent_horas - 100, 1)
    
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Total de Horas da Máquina em {target_month}-{target_year}", f"{total_de_horas}H", f'{round(total_de_horas-160,1)}H')
    col2.metric('Eficiência (%)', f'{percent_horas}%', f'{delta_1}%')
    col3.metric("Média", f"{media}H")
   
    col11,col12 = st.columns([0.8,0.2])
    
    new_df_ano = new_df[new_df['Datetime_ini'].dt.year == target_year]
    ordem_2 = new_df_ano.groupby(['estacao', new_df_ano['Datetime_ini'].dt.month, 'nome_func'])['delta_time_hours'].sum().reset_index().round(2)
    ordem_2.rename(columns = {'delta_time_hours':'Tempo de uso total (H)'}, inplace = True)
    ordem_2.rename(columns = {'Datetime_ini': 'Mês'}, inplace = True)
    
    fig2 = px.bar(ordem_2, x = 'Mês', y = round((ordem_2['Tempo de uso total (H)']/hora_esperada_de_trabalho)*100,2),color='nome_func',title= 'Eficiência Mensal',text_auto='.2s', width=1300)
    fig2.update_traces(textfont_size=16, textangle=0, textposition="outside", cliponaxis=False)
    fig2.update_layout(yaxis_title = 'Eficiência (%)',legend_title_text = 'Colaborador', title_x = 0.5, title_y = 0.95,title_xanchor = 'center')
    fig2.update_xaxes(tickvals=list(range(len(ordem_2)+1)))
    col11.plotly_chart(fig2)

with tab2:
    col9,col10 = st.columns([0.2,0.8])
    with col9:
        target_pv = st.selectbox("Selecione o PV", pedidos["pedido"].sort_values().unique(),placeholder ='Escolha uma opção')
    
    col4, col5 = st.columns([3,2])

    descricao = pedidos.loc[pedidos['pedido'] == target_pv]
    descricao = descricao.reset_index(drop=True)

    descricao = descricao.iloc[0,14]

    pedido = pedidos[pedidos['pedido'] == target_pv]
    quant = pedido['quant_a_fat'].iloc[0]
    filtro_df = pedido['ordem']
    ordem = ordens[ordens['ordem'].isin(filtro_df)]
    soma_por_estacao = ordem.groupby('estacao')['delta_time_hours'].sum().reset_index().round(2)
    soma_por_estacao.rename(columns={'delta_time_hours': 'Tempo de uso total (H)'}, inplace=True)
    soma_por_estacao.rename(columns={'estacao': 'Estação de Trabalho'}, inplace=True)
    soma_por_estacao['Tempo de uso por Peça (min)'] = round((soma_por_estacao['Tempo de uso total (H)']/quant)*60,2)
    total_de_minutos_peca = round(soma_por_estacao['Tempo de uso por Peça (min)'].sum(),2)
    total_de_horas_pedido = round(soma_por_estacao['Tempo de uso total (H)'].sum(),2)

    fig = px.pie(soma_por_estacao, values='Tempo de uso total (H)', names='Estação de Trabalho', title='Proporção de Tempo de Uso por Máquina em Cada Pedido', width=800, height=500)
    fig.update_layout(title_yref='container',title_xanchor = 'center',title_x = 0.43, title_y = 0.95, legend=dict(font=dict(size=18)),font=dict(size=20), title_font=dict(size=20))
    col4.plotly_chart(fig, use_container_width=True)
    nova_linha = {'Estação de Trabalho': 'Total', 'Tempo de uso total (H)': total_de_horas_pedido,'Tempo de uso por Peça (min)': total_de_minutos_peca}
    soma_por_estacao = pd.concat([soma_por_estacao, pd.DataFrame([nova_linha])], ignore_index=True)
    with col5:
        st.markdown(f"<h1 style='font-size: 20px;'>Tabela de Horas por Estação no PV {target_pv}/Número de peças é {quant}</h1>", unsafe_allow_html=True)
    col5.dataframe(soma_por_estacao, width= 500,hide_index=True)
    with col10:
        st.markdown(f"<h1 style='text-align: left;'>{descricao}</h1>", unsafe_allow_html=True)

with tab3:
    col12, col13, col14 = st.columns(3)
    with col12:
        target_peca = st.selectbox("Peça", pedidos["descricao"].sort_values().unique(), index= 0,placeholder ='Escolha uma opção')
    pedidos_cleaned = pedidos.dropna(subset=['descricao'])
    peca = pedidos_cleaned[pedidos_cleaned['descricao'].str.contains(target_peca)]
    filtro_df_2 = peca['ordem']
    ordem_peca = ordens[ordens['ordem'].isin(filtro_df_2)]
    peca_estacao = ordem_peca.groupby('estacao')['delta_time_min'].sum().reset_index().round(2)
    total = round(peca_estacao['delta_time_min'].sum(),2)

    average_minutes = round(peca_estacao['delta_time_min'] / len(peca_estacao['estacao']),0)
    peca_estacao['time_delta'] = average_minutes.apply(lambda x: timedelta(minutes=x))
    base_datetime = pd.Timestamp.now().replace(hour=0, minute=0, second=0, microsecond=0)
    peca_estacao['result_datetime'] = base_datetime + peca_estacao['time_delta']
    peca_estacao['Tempo Médio por '] = peca_estacao['result_datetime'].dt.strftime('%H:%M')
    peca_estacao.drop('time_delta', axis=1, inplace=True)
    peca_estacao.drop('result_datetime', axis=1, inplace=True)
    media_2 = round(total / len(peca_estacao['estacao']),2)
    media_3 = round(media_2/60,2)
    col13.metric("Tempo médio para a fabricação em Horas", f"{media_3}H")
    col14.metric("Tempo médio para a fabricação em Minutos", f"{media_2}min")
    col15, col16 = st.columns([0.90,0.1])
    peca_estacao.drop('delta_time_min', axis = 1, inplace= True)
    peca_estacao.rename(columns={'estacao': 'Estação de Trabalho'}, inplace=True)
    col15.dataframe(peca_estacao, width= 1500,hide_index=True)