import streamlit as st
import pandas as pd
import plotly.express as px
from datetime  import datetime as dt , timedelta
import seaborn as sns
import numpy as np
def convert_to_HM(x):
    # Convert to float
    hours = float(x)
    # Separate the integer part (hours) and the fractional part (minutes)
    h = int(hours)
    m = int((hours - h) * 60)
    # Return in "H:M" format
    return f"{h}:{m:02d}"

def count_sundays(start_date, end_date):
    if pd.isna(start_date) or pd.isna(end_date):
        return 0
    all_dates = pd.date_range(start=start_date, end=end_date)
    sundays = all_dates[all_dates.weekday == 6]
    return len(sundays)

# def count_weekend_days(start_date, end_date):
#     if pd.isna(start_date) or pd.isna(end_date):
#         return 0
#     all_dates = pd.date_range(start=start_date, end=end_date)
#     weekends = all_dates[(all_dates.weekday == 5) | (all_dates.weekday == 6)]
#     return len(weekends)

def adjust_delta_time(row):
    if pd.notna(row['hora_ini']) and pd.notna(row['hora_fim']):
        if (row['delta_dia'] == 0 and row['hora_fim'] > pd.to_datetime('12:30:00').time() and row['hora_ini'] < pd.to_datetime('11:30:00').time()):
            row['delta_time_hours']  = row['delta_time_hours'] - 1
    return row

def adjust_delta_time_hours(row):
    if row['delta_dia'] != 0:
        row['delta_time_hours'] = row['delta_time_hours'] - ((row['delta_dia']-row['weekends_count']) * 14) - (row['weekends_count'] * 24)
    return row



st.set_page_config(layout="wide")   
st.image('logo.png', width= 150)

ordens = pd.read_csv('ordens (4).csv', sep = ',')
pedidos = pd.read_csv('pedidos (1).csv', sep = ',')
orc = pd.read_csv('orcamento_csv.csv', sep=';')

ordens['ordem'] = ordens['ordem'].fillna(0)
ordens['data_ini'] = ordens['data_ini'].fillna(0)
ordens['hora_ini'] = ordens['hora_ini'].fillna(0)
ordens['ordem'] = ordens['ordem'].astype(int)
ordens.loc[:, 'Datetime_ini'] = pd.to_datetime(ordens['data_ini'] + ' ' + ordens['hora_ini'], format = 'mixed', errors='coerce')
ordens.loc[:,'Datetime_fim'] = pd.to_datetime(ordens['data_fim'] + ' ' + ordens['hora_fim'], format = 'mixed', errors='coerce')
ordens.loc[:, 'delta_time_seconds'] = (ordens['Datetime_fim'] - ordens['Datetime_ini']).dt.total_seconds()
ordens.loc[:, 'delta_time_hours'] = ordens['delta_time_seconds'] / 3600
ordens.loc[:,'delta_time_min'] = ordens['delta_time_seconds']/60

ordens.loc[ordens['estacao'] == 'CCNC 001', 'estacao'] = 'CNC 001'
ordens.loc[ordens['estacao'] == 'CCNC001', 'estacao'] = 'CNC 001'
ordens.loc[ordens['estacao'] == 'CCNC01', 'estacao'] = 'CNC 001'
ordens.loc[ordens['estacao'] == 'PLM001', 'estacao'] = 'PLM 001'
ordens.loc[ordens['estacao'] == 'PLM 01', 'estacao'] = 'PLM 001'
ordens["data_ini"] = pd.to_datetime(ordens["data_ini"], format = 'mixed', errors='coerce')
ordens["data_fim"] = pd.to_datetime(ordens["data_fim"], format = 'mixed', errors='coerce')
ordens["Ano"] = ordens["data_ini"].dt.year.astype('Int64') 
ordens["Mes"] = ordens["data_ini"].dt.month.astype('Int64')
ordens['delta_dia'] = (ordens['data_fim'] - ordens['data_ini']).dt.days
    
ordens['weekends_count'] = ordens.apply(lambda row: count_sundays(row['data_ini'], row['data_fim']), axis=1)

ordens['hora_fim'] = pd.to_datetime(ordens['hora_fim'], format='%H:%M:%S').dt.time
ordens['hora_ini'] = pd.to_datetime(ordens['hora_ini'], format='%H:%M:%S').dt.time

ordens = ordens.apply(adjust_delta_time_hours, axis=1)
ordens = ordens.apply(adjust_delta_time, axis=1)


tab1, tab2, tab3 = st.tabs(["ANÁLISE HORA DE TRABALHO MENSAL", "ANÁLISE HORA DE TRABALHO POR PV", "MÉDIA POR PEÇA"])
with tab1:
    col6, col7, col8 = st.columns(3)
    with col6:
        estacao = st.selectbox("Estação", ordens["estacao"].sort_values().unique(), index= 0,placeholder ='Escolha uma opção')
    
    ordens = ordens[~((ordens['id'] == 17854) | (ordens['id'] == 17856) | (ordens['id'] == 17858))]
    
    ordens=ordens.sort_values("data_ini")

    with col7:
        target_month = st.selectbox("Mês", ordens["Mes"].sort_values().unique(), index= 0,placeholder ='Escolha uma opção')
    with col8:
        target_year = st.selectbox("Ano", ordens["Ano"].sort_values().unique(), index= 1 ,placeholder ='Escolha uma opção')
    
    new_df = ordens[ordens['estacao'] == estacao]
    
    df_filtrado_year = new_df[new_df['Datetime_ini'].dt.year == target_year]
    df_filtrado = df_filtrado_year[df_filtrado_year['Datetime_ini'].dt.month == target_month]
    hora_esperada_de_trabalho = 200
    num_entries = df_filtrado.shape[0]
    
    total_de_horas = round(df_filtrado['delta_time_hours'].sum(),1)
    
    percent_horas = round((total_de_horas/hora_esperada_de_trabalho) * 100, 1)
    
    media = np.divide(total_de_horas,num_entries,out=np.zeros_like(total_de_horas), where=num_entries!=0).round(1)
    
    delta_1 = round(percent_horas - 100, 1)
    
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Total de Horas da Máquina em {target_month}-{target_year}", f"{total_de_horas}H", f'{round(total_de_horas-200,1)}H')
    col2.metric('Eficiência (%)', f'{percent_horas}%', f'{delta_1}%')
    col3.metric("Média", f"{media}H")
   
    col11,col12 = st.columns([0.8,0.2])

    ordem_2 = df_filtrado_year.groupby(['estacao', df_filtrado_year['Datetime_ini'].dt.month])['delta_time_hours'].sum().reset_index().round(2)
    ordem_2.rename(columns = {'delta_time_hours':'Tempo de uso total (H)'}, inplace = True)
    ordem_2.rename(columns = {'Datetime_ini': 'Mês'}, inplace = True)

    fig2 = px.bar(ordem_2, x = 'Mês', y = round((ordem_2['Tempo de uso total (H)']/hora_esperada_de_trabalho)*100,2),color='Mês',title= 'Eficiência Mensal',text_auto='.3s', width=1300)
    fig2.update_traces(textfont_size=16, textangle=0, textposition="outside", cliponaxis=False)
    fig2.update_layout(yaxis_title = 'Eficiência (%)', title_x = 0.5, title_y = 0.95,title_xanchor = 'center')
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
    
    ordem.loc[ordem['estacao'].str.contains('SRC', na=False), 'estacao'] = 'Corte-Serra'
    ordem.loc[ordem['estacao'].str.contains('SFH', na=False), 'estacao'] = 'Corte-Serra'
    ordem.loc[ordem['estacao'].str.contains('TCNV', na=False), 'estacao'] = 'Torno convencional'
    ordem.loc[ordem['estacao'].str.contains('TCNC', na=False), 'estacao'] = 'Torno CNC'
    ordem.loc[ordem['estacao'].str.contains('FRZ', na=False), 'estacao'] = 'Fresadora convencional'
    ordem.loc[ordem['estacao'].str.contains('CNC', na=False), 'estacao'] = 'Fresadora CNC'
    ordem.loc[ordem['estacao'].str.contains('PLM', na=False), 'estacao'] = 'Corte-Plasma'
    ordem.loc[ordem['estacao'].str.contains('MCL', na=False), 'estacao'] = 'Corte-Laser'
    ordem.loc[ordem['estacao'].str.contains('GLT', na=False), 'estacao'] = 'Corte-Guilhotina'
    ordem.loc[ordem['estacao'].str.contains('DHCNC', na=False), 'estacao'] = 'Dobra'
    ordem.loc[ordem['estacao'].str.contains('MQS', na=False), 'estacao'] = 'Soldagem'
    ordem = ordem.dropna(subset=['estacao', 'delta_time_hours'])
   
    soma_por_estacao = ordem.groupby('estacao')['delta_time_hours'].sum().reset_index().round(2)
    soma_por_estacao.rename(columns={'delta_time_hours': 'Tempo de uso total (H:M)'}, inplace=True)
    soma_por_estacao.rename(columns={'estacao': 'Estação de Trabalho'}, inplace=True)
    total_de_horas_pedido = round(soma_por_estacao['Tempo de uso total (H:M)'].sum(),2)

    fig = px.pie(soma_por_estacao, values='Tempo de uso total (H:M)', names='Estação de Trabalho', title='Proporção de Tempo de Uso por Máquina em Cada Pedido', width=800, height=500)
    fig.update_layout(title_yref='container',title_xanchor = 'center',title_x = 0.43, title_y = 0.95, legend=dict(font=dict(size=18)),font=dict(size=20), title_font=dict(size=20))
    col4.plotly_chart(fig, use_container_width=True)
    nova_linha = {'Estação de Trabalho': 'Total', 'Tempo de uso total (H:M)': total_de_horas_pedido}
    soma_por_estacao = pd.concat([soma_por_estacao, pd.DataFrame([nova_linha])], ignore_index=True)
    
    orc_pv = orc[orc['PV'] == target_pv]
    if not orc_pv.empty:
        soma_por_estacao['Tempo esperado no Orçamento'] = pd.Series([np.nan] * len(soma_por_estacao))

        for index,row in orc_pv.iterrows():
            corte_plasma = convert_to_HM(row['Corte-Plasma'])
            corte_serra = convert_to_HM(row['Corte-Serra'])
            corte_laser = convert_to_HM(row['Corte-Laser'])
            corte_guilhotina = convert_to_HM(row['Corte-Guilhotina'])
            torno = convert_to_HM(row['Torno convencional'])
            torno_CNC = convert_to_HM(row['Torno CNC'])
            fresa = convert_to_HM(row['Fresadora convencional'])
            fresa_CNC = convert_to_HM(row['Fresadora CNC'])
            prensa = convert_to_HM(row['Prensa'])
            dobra = convert_to_HM(row['Dobra'])
            rosqueadeira = convert_to_HM(row['Rosqueadeira'])
            furadeira = convert_to_HM(row['Furadeira de bancada'])
            soldagem = convert_to_HM(row['Soldagem'])
            acabamento = convert_to_HM(row['Acabamento'])
            jato = convert_to_HM(row['Jateamento'])
            pintura = convert_to_HM(row['Pintura'])
            montagem = convert_to_HM(row['Montagem'])
            inspecao = convert_to_HM(row['inspecao'])
            expedicao = convert_to_HM(row['expedicao'])
            adm = convert_to_HM(row['ADM'])
            dgq = convert_to_HM(row['DGQ'])
            total = convert_to_HM(row['Total'])

        tempo_esperado = {'Corte-Plasma': corte_plasma,'Corte-Serra': corte_serra,'Corte-Laser': corte_laser,'Corte-Guilhotina': corte_guilhotina,'Torno convencional': torno,'Torno CNC': torno_CNC,'Fresadora convencional': fresa,'Fresadora CNC': fresa_CNC,'Prensa': prensa,'Dobra': dobra,'Rosqueadeira': rosqueadeira,'Furadeira de bancada': furadeira,'Soldagem': soldagem,'Acabamento': acabamento,'Jateamento': jato,'Pintura': pintura,'Montagem': montagem,'inspecao': inspecao,'expedicao': expedicao,'ADM': adm,'DGQ': dgq,'Total': total}
        for estacao, tempo in tempo_esperado.items():
            if (soma_por_estacao['Estação de Trabalho'] == estacao).any():
                soma_por_estacao.loc[soma_por_estacao['Estação de Trabalho'] == estacao, 'Tempo esperado no Orçamento'] = tempo
    
    col17,col18 = st.columns([0.9,0.1])
    # g = sns.lineplot(data=soma_por_estacao, x="Estação de Trabalho", y="Tempo de uso total (H:M)",hue = 'Estação de Trabalho', height=5, kind="strip",aspect=2)    
    # col17.pyplot(g)
    # x_rmin = (min(ordem["Datetime_ini"]))
    # x_rmax = (max(ordem["Datetime_fim"]))
    
    # x_range = [x_rmin, x_rmax]
    # timeline = px.timeline(ordem, x_start="Datetime_ini", x_end="Datetime_fim", y="estacao", range_x=x_range)

    # col17.plotly_chart(timeline, use_container_width=True)

    soma_por_estacao['Tempo de uso total (H:M)'] = soma_por_estacao['Tempo de uso total (H:M)'].apply(convert_to_HM)
    
    with col5:
        st.markdown(f"<h1 style='font-size: 20px;'>Tabela de Horas por Estação no PV {target_pv}/Número de peças é {quant}</h1>", unsafe_allow_html=True)
    col5.dataframe(soma_por_estacao, width= 500,hide_index=True)
    with col10:
        st.markdown(f"<h1 style='text-align: left;'>{descricao}</h1>", unsafe_allow_html=True)