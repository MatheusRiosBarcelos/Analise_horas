import streamlit as st
import pandas as pd
import plotly.express as px
from datetime  import datetime as dt , timedelta
import numpy as np
import plotly.graph_objects as go

def convert_to_HM(x):
    hours = float(x)
    h = int(hours)
    m = int((hours - h) * 60)
    return f"{h}:{m:02d}"

def count_weekend_days(start_date, end_date):
    if pd.isna(start_date) or pd.isna(end_date):
        return 0
    all_dates = pd.date_range(start=start_date, end=end_date)
    weekends = all_dates[(all_dates.weekday == 5) | (all_dates.weekday == 6)]
    return len(weekends)

def adjust_delta_time(row):
    if pd.notna(row['hora_ini']) and pd.notna(row['hora_fim']):
        if (row['delta_dia'] == 0 and row['hora_fim'] > pd.to_datetime('12:30:00').time() and row['hora_ini'] < pd.to_datetime('11:30:00').time()):
            row['delta_time_hours']  = row['delta_time_hours'] - 1
    return row

def adjust_delta_time_hours(row):
    if row['delta_dia'] != 0:
        row['delta_time_hours'] = row['delta_time_hours'] - ((row['delta_dia']-row['weekends_count']) * 14) - (row['weekends_count'] * 24)
    return row

def get_quantity(cliente_name):
    return np.where(
        df_combinado['cliente'].str.contains(cliente_name, na=False),
        df_combinado['Quantidade de Pedidos'],
        0
    ).sum()

def create_pie_chart(df, values_column, names_column, title):
    ordem_das_categorias = ['WEG', 'GE', 'TAVRIDA', 'HITACHI', 'SHAMAH', 'PRODUZ', 'PISOM', 'MAGVATECH', 'HVEX', 'ANTONIO EDUARDO']
    cores = {
        'WEG': 'blue',
        'GE': 'lightblue',
        'TAVRIDA': 'red',
        'HITACHI': 'orange',
        'SHAMAH': 'purple',
        'PRODUZ': 'green',
        'PISOM': 'yellow',
        'MAGVATECH': 'pink',
        'HVEX': 'grey',
        'ANTONIO EDUARDO': 'brown'
    }
    red_palette = ['#FF0000', '#fd3a3a', '#fa5252', '#FF8E8E', '#FF9E9E',
               '#FFAEAE', '#FFBEBE', '#FFCECE', '#FFE0E0', '#FFF0F0']  
    df = df.sort_values(by=names_column)
    
    fig = go.Figure(data=[go.Pie(
        labels=df[names_column],
        values=df[values_column],
        marker=dict(colors=[red_palette[i] for i in range(len(df[names_column]))]),
        text=[f"{percent:.2f}%" for percent in df[values_column]],
        textinfo='label+text',
        insidetextorientation='tangential',
        textfont=dict(size=18)
    )])

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        width=800,
        height=500,
        margin=dict(t=100, b=0, l=125, r=110),
        showlegend=False,
        font=dict(size=20),
        title_font=dict(size=18)
    )
    
    return fig


st.set_page_config(layout="wide")   
st.image('logo.png', width= 150)

ordens = pd.read_csv('ordens (4).csv', sep = ',')
pedidos = pd.read_csv('pedidos (1).csv', sep = ',')
orc = pd.read_csv('orcamento_csv.csv', sep=';')

ordens = ordens[ordens['estacao'] != 'Selecione...']

ordens['ordem'] = ordens['ordem'].fillna(0)
ordens['data_ini'] = ordens['data_ini'].fillna(0)
ordens['hora_ini'] = ordens['hora_ini'].fillna(0)
ordens['ordem'] = ordens['ordem'].astype(int)

ordens.loc[:, 'Datetime_ini'] = pd.to_datetime(ordens['data_ini'] + ' ' + ordens['hora_ini'], format = 'mixed', errors='coerce')
ordens.loc[:,'Datetime_fim'] = pd.to_datetime(ordens['data_fim'] + ' ' + ordens['hora_fim'], format = 'mixed', errors='coerce')
ordens.loc[:, 'delta_time_seconds'] = (ordens['Datetime_fim'] - ordens['Datetime_ini']).dt.total_seconds()
ordens.loc[:, 'delta_time_hours'] = ordens['delta_time_seconds'] / 3600
ordens.loc[:,'delta_time_min'] = ordens['delta_time_seconds']/60

ordens['estacao'] = ordens['estacao'].fillna('a')

mqs = (ordens['estacao'].str.contains('MQS'))
ordens.loc[mqs,'estacao'] = 'Soldagem'
ordens.loc[ordens['estacao'].str.contains('JPS'),'estacao'] = 'JATO'
ordens.loc[ordens['estacao'].str.contains('LASER'),'estacao'] = 'MCL 001'
ordens.loc[ordens['estacao'].str.contains('DGQ'),'estacao'] = 'QUALIDADE'
ordens.loc[ordens['estacao'] == 'FRZ 033', 'estacao'] = 'FRZ 003'
ordens.loc[ordens['estacao'] == 'FRZ003', 'estacao'] = 'FRZ 003'
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
ordens['hora_fim'] = pd.to_datetime(ordens['hora_fim'], format='%H:%M:%S').dt.time
ordens['hora_ini'] = pd.to_datetime(ordens['hora_ini'], format='%H:%M:%S').dt.time

midnight = ordens['Datetime_fim'].dt.normalize()
seven_am = midnight + pd.Timedelta(hours=7) 
condition = ((ordens['delta_dia'] == 1) & (ordens['Datetime_ini'] < midnight) & (ordens['Datetime_fim'] >= midnight) & (ordens['Datetime_fim'] <= seven_am))
if condition.any():
    ordens.loc[condition, 'delta_dia'] = 0

ordens['weekends_count'] = ordens.apply(lambda row: count_weekend_days(row['data_ini'], row['data_fim']), axis=1)

ordens = ordens.apply(adjust_delta_time_hours, axis=1)
ordens = ordens.apply(adjust_delta_time, axis=1)
ordens = ordens.loc[~(ordens[['delta_time_hours']] < 0).any(axis=1)]

tab1, tab2, tab3, tab4 = st.tabs(["ANÁLISE HORA DE TRABALHO MENSAL", "ANÁLISE HORA DE TRABALHO POR PV", "TEMPO MÉDIO PARA A FARBICAÇÃO DE PRODUTOS ", 'ANÁLISE MENSAL DE PEDIDOS'])
with tab1:
    col6, col7, col8 = st.columns(3)
    with col6:
        estacao = st.selectbox("Estação", ordens["estacao"].sort_values().unique(), index= 4,placeholder ='Escolha uma opção')
    
    ordens = ordens[~((ordens['id'] == 17854) | (ordens['id'] == 17856) | (ordens['id'] == 17858))]
    
    ordens=ordens.sort_values("data_ini")

    with col7:
        target_month = st.selectbox("Mês", ordens["Mes"].sort_values().unique(),key=1, index= 0,placeholder ='Escolha uma opção')
    with col8:
        target_year = st.selectbox("Ano", ordens["Ano"].sort_values().unique(),key=2 ,index= 1 ,placeholder ='Escolha uma opção')
    
    new_df = ordens[ordens['estacao'] == estacao]
    
    df_filtrado_year = new_df[new_df['Datetime_ini'].dt.year == target_year]
    df_filtrado = df_filtrado_year[df_filtrado_year['Datetime_ini'].dt.month == target_month]
    hora_esperada_de_trabalho = 200
    num_entries = df_filtrado.shape[0]
    
    total_de_horas = round(df_filtrado['delta_time_hours'].sum(),1)
    
    percent_horas = int((total_de_horas/hora_esperada_de_trabalho) * 100)
    
    media = np.divide(total_de_horas,num_entries,out=np.zeros_like(total_de_horas), where=num_entries!=0).round(1)
    
    delta_1 = round(percent_horas - 100, 1)
    
    col1, col2 = st.columns(2)

    col1.metric(f"Total de Horas da Máquina em {target_month}-{target_year}", f"{total_de_horas}H", f'{round(total_de_horas-200,1)}H')
    col2.metric('Eficiência (%)', f'{percent_horas}%', f'{delta_1}%')
    # col3.metric("Média", f"{media}H")
   
    col11,col12,col13 = st.columns(3)

    ordem_2 = df_filtrado_year.groupby(['estacao', df_filtrado_year['Datetime_ini'].dt.month])['delta_time_hours'].sum().reset_index().round(2)
    ordem_2.rename(columns = {'delta_time_hours':'Tempo de uso total (H)'}, inplace = True)
    ordem_2.rename(columns = {'Datetime_ini': 'Mês'}, inplace = True)

    fig2 = px.bar(ordem_2, x = 'Mês', y = (ordem_2['Tempo de uso total (H)']/hora_esperada_de_trabalho*100).astype(int),title= f'Eficiência Mensal {estacao} (%)',text_auto='.2s', width=400, height=500)
    fig2.update_traces(textfont_size=16, textangle=0, textposition="outside", cliponaxis=False, marker_color='#e53737')
    fig2.update_layout(yaxis_title = 'Eficiência (%)', title_x = 0.55, title_y = 0.95,title_xanchor = 'center')
    fig2.update_xaxes(tickvals=list(range(len(ordem_2)+1)))
    
    col11.plotly_chart(fig2)

    x = ordens[ordens['Datetime_ini'].dt.year == target_year]
    x.loc[x['estacao'].str.contains('SRC', na=False), 'estacao'] = 'Corte-Serra'
    x.loc[x['estacao'].str.contains('SFH', na=False), 'estacao'] = 'Corte-Serra'
    x.loc[x['estacao'].str.contains('TCNV', na=False), 'estacao'] = 'Torno convencional'
    x.loc[x['estacao'].str.contains('TCNC', na=False), 'estacao'] = 'Torno CNC'
    x.loc[x['estacao'].str.contains('FRZ', na=False), 'estacao'] = 'Fresadora convencional'
    x.loc[x['estacao'].str.contains('CNC 001', na=False), 'estacao'] = 'Fresadora CNC'
    x.loc[x['estacao'].str.contains('PLM', na=False), 'estacao'] = 'Corte-Plasma'
    x.loc[x['estacao'].str.contains('MCL', na=False), 'estacao'] = 'Corte-Laser'
    x.loc[x['estacao'].str.contains('GLT', na=False), 'estacao'] = 'Corte-Guilhotina'
    x.loc[x['estacao'].str.contains('DHCNC', na=False), 'estacao'] = 'Dobra'
    x.loc[x['estacao'].str.contains('DBE', na=False), 'estacao'] = 'Dobra'
    x.loc[x['estacao'].str.contains('MQS', na=False), 'estacao'] = 'Soldagem'
    x = x.groupby(['estacao', x['Datetime_ini'].dt.month])['delta_time_hours'].sum().reset_index().round(2)
    x = x[x['estacao'].isin(['Corte-Serra', 'Torno convencional', 'Torno CNC', 'Fresadora convencional', 'Fresadora CNC', 'Corte-Plasma', 'Corte-Laser', 'Corte-Guilhotina', 'Dobra', 'Soldagem'])]
    
    y = x.groupby('Datetime_ini')['delta_time_hours'].sum().reset_index().round(2)
    y['delta_time_hours'] = ((y['delta_time_hours'] / 5200)*100).round(2)
    
    fig21 = px.bar(y, x = 'Datetime_ini', y = 'delta_time_hours',title= f'Eficiência Mensal Total da Fábrica (%)',text_auto='.2s', width=400, height=500)
    fig21.update_traces(textfont_size=16, textangle=0, textposition="outside", cliponaxis=False, marker_color='#e53737')
    fig21.update_layout(yaxis_title = 'Eficiência (%)', xaxis_title = 'Mês', title_x = 0.55, title_y = 0.95,title_xanchor = 'center')
    fig21.update_xaxes(tickvals=list(range(len(y)+1)))

    col12.plotly_chart(fig21)


    fig20 = go.Figure(data=go.Heatmap(
            z=x['delta_time_hours'],
            x=x['Datetime_ini'],
            y=x['estacao'],
            colorscale='Reds'),
            )

    fig20.update_layout(title='Mapa de Calor Horas trabalhadas Mensalmente', width=500, height= 500,title_x = 0.55, title_y = 0.95,title_xanchor = 'center', xaxis_title = 'Mês')

    # col001,col002 = st.columns([0.5,0.5])

    col13.plotly_chart(fig20)

with tab2:
    col9,col10 = st.columns([0.2,0.8])
    with col9:
        target_pv = st.selectbox("Selecione o PV", pedidos["pedido"].sort_values().unique(),index=1200,placeholder ='Escolha uma opção')
    
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
    ordem.loc[ordem['estacao'].str.contains('CNC 001', na=False), 'estacao'] = 'Fresadora CNC'
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

    soma_por_estacao['Tempo de uso total (H:M)'] = soma_por_estacao['Tempo de uso total (H:M)'].apply(convert_to_HM)
    
    with col5:
        st.markdown(f"<h1 style='font-size: 20px;'>Tabela de Horas por Estação no PV {target_pv}/Número de peças é {quant}</h1>", unsafe_allow_html=True)
    col5.dataframe(soma_por_estacao, width= 500,hide_index=True)
    with col10:
        st.markdown(f"<h1 style='text-align: left;'>{descricao}</h1>", unsafe_allow_html=True)

with tab3:

    col20,col21 = st.columns([0.35,0.65])
    codprod_target = st.text_input("Código do Produto", value= '14303600')
    number_parts = st.number_input("Quantas peças são", value=int(1), placeholder="Type a number...")

    pedido_cod = pedidos[pedidos['codprod'].str.contains(codprod_target, na= False)]
    
    filtro_df_cod = pedido_cod['ordem']
    ordem_cod = ordens[ordens['ordem'].isin(filtro_df_cod)]
    merged_df = pd.merge(pedido_cod, ordem_cod, on='ordem', how='left')
    merged_df['delta_time_hours'] = merged_df['delta_time_hours'] / merged_df['quant_a_fat']
    merged_df.loc[merged_df['estacao'].str.contains('SRC', na=False), 'estacao'] = 'Corte-Serra'
    merged_df.loc[merged_df['estacao'].str.contains('SFH', na=False), 'estacao'] = 'Corte-Serra'
    merged_df.loc[merged_df['estacao'].str.contains('TCNV', na=False), 'estacao'] = 'Torno convencional'
    merged_df.loc[merged_df['estacao'].str.contains('TCNC', na=False), 'estacao'] = 'Torno CNC'
    merged_df.loc[merged_df['estacao'].str.contains('FRZ', na=False), 'estacao'] = 'Fresadora convencional'
    merged_df.loc[merged_df['estacao'].str.contains('CNC', na=False), 'estacao'] = 'Fresadora CNC'
    merged_df.loc[merged_df['estacao'].str.contains('PLM', na=False), 'estacao'] = 'Corte-Plasma'
    merged_df.loc[merged_df['estacao'].str.contains('MCL', na=False), 'estacao'] = 'Corte-Laser'
    merged_df.loc[merged_df['estacao'].str.contains('GLT', na=False), 'estacao'] = 'Corte-Guilhotina'
    merged_df.loc[merged_df['estacao'].str.contains('DHCNC', na=False), 'estacao'] = 'Dobra'
    merged_df.loc[merged_df['estacao'].str.contains('MQS', na=False), 'estacao'] = 'Soldagem'
    
    index_of_first_occurrence = merged_df[((merged_df['matriz'] == 'SIM') & (~merged_df['descricao'].isna()))].index[0]
    descricao_2 = merged_df.loc[index_of_first_occurrence, 'descricao']

    merged_df = merged_df.dropna(subset=['estacao', 'delta_time_hours'])
    merged_df = merged_df.groupby('estacao')['delta_time_hours'].mean().reset_index().round(2)
    merged_df['delta_time_hours'] = merged_df['delta_time_hours'] * number_parts
    tempo_total_medio = convert_to_HM(merged_df['delta_time_hours'].sum())

    if 'delta_time_hours' in merged_df.columns:
        if merged_df['delta_time_hours'].notnull().all():
            # Aplicando a função de conversão
            merged_df['delta_time_hours'] = merged_df['delta_time_hours'].apply(convert_to_HM)
    
    merged_df.rename(columns={'delta_time_hours': 'Tempo Médio de Uso (H:M)'}, inplace=True)
    merged_df.rename(columns={'estacao': 'Operação'}, inplace=True)

    operacoes_excluir = ['ADM', 'QUALIDADE', 'INSPEÇÃO DE QUANTIDA']
    merged_df = merged_df[~merged_df['Operação'].isin(operacoes_excluir)]
    if ('Corte-Plasma' in merged_df['Operação'].values) and ('Corte-Laser' in merged_df['Operação'].values):
        merged_df = merged_df[merged_df['Operação'] != 'Corte-Plasma']

    nova_linha_2 = {'Operação': 'Total', 'Tempo Médio de Uso (H:M)': tempo_total_medio}
    merged_df = pd.concat([merged_df, pd.DataFrame([nova_linha_2])], ignore_index=True)

    col20.dataframe(merged_df, width= 500,hide_index=True)

    with col21:
        st.markdown(f"<h1 style='text-align: left;'>{descricao_2}</h1>", unsafe_allow_html=True)

with tab4:
    
    col22,col23 = st.columns([0.5,0.5])
    with col22:
        target_month_2 = st.selectbox("Mês", ordens["Mes"].sort_values().unique(), key=3,index= 0,placeholder ='Escolha uma opção')
    with col23:
        target_year_2 = st.selectbox("Ano", ordens["Ano"].sort_values().unique(), key=4,index= 1 ,placeholder ='Escolha uma opção')

    pedidos = pedidos.drop_duplicates(subset=['pedido'], keep='first')

    pedidos["entrega"] = pd.to_datetime(pedidos["entrega"], format = 'mixed', errors='coerce')
    pedidos = pedidos[pedidos['entrega'].dt.month == target_month_2]
    pedidos = pedidos[pedidos['entrega'].dt.year == target_year_2]  

    pedidos.loc[pedidos['cliente'].str.contains('WEG', na=False), 'cliente'] = 'WEG'
    pedidos.loc[pedidos['cliente'].str.contains('GE', na=False), 'cliente'] = 'GE'

    pedidos_clientes = pedidos.groupby('cliente').size().reset_index(name='Quantidade de Pedidos')
    pedidos_clientes.sort_values(by='Quantidade de Pedidos', ascending=False, inplace=True)
    pedidos_clientes.reset_index(drop=True, inplace=True)

    total = pedidos_clientes['Quantidade de Pedidos'].sum()
    
    pedidos_clientes['Porcentagem (%)'] = ((pedidos_clientes['Quantidade de Pedidos'] / total) * 100).round(2)
    
    pedidos_pecas = pedidos.groupby('cliente')['quant_a_fat'].sum().reset_index()
    pedidos_pecas.sort_values(by='quant_a_fat', ascending=False, inplace=True)
    pedidos_pecas.reset_index(drop=True, inplace=True)
    df_combinado = pedidos_clientes.merge(pedidos_pecas[['cliente', 'quant_a_fat']], on='cliente', how='left')
    total_pecas = df_combinado['quant_a_fat'].sum()
    df_combinado['Porcentagem de Peças (%)'] = ((df_combinado['quant_a_fat'] / total_pecas) * 100).round(2)
    df_combinado.rename(columns={'quant_a_fat': 'Quantidade de peças por cliente'}, inplace=True)
    df_combinado.sort_values(by= 'cliente', ascending=False, inplace= True)
    col24,col25,col26,col27,col28,col29 = st.columns(6)
    col30,col31,col32 = st.columns(3)

    weg = get_quantity('WEG')
    ge = get_quantity('GE')
    tav = get_quantity('TAV')
    hita = get_quantity('HITA')
    sha = get_quantity('SHA')
    pis = get_quantity('PIS')
    prod = get_quantity('PROD')
    hv = get_quantity('HVEX')
    mg = get_quantity('MAGVATECH')

    total_de_pedidos = weg + ge + tav + hita + sha + pis + prod + hv + mg

    col24.metric(f"Pedidos para WEG", weg)
    col25.metric(f"Pedidos para GE", ge)
    col26.metric(f"Pedidos para TAVRIDA", tav)
    col27.metric(f"Pedidos para HITACHI", hita)
    col28.metric(f"Pedidos para SHAMAH", sha)
    col29.metric(f"Pedidos para PISOM", pis)
    col30.metric(f"Pedidos para PRODUZ", prod)
    col31.metric(f"Pedidos para HVEX", hv)
    col32.metric(f"Pedidos para MAGVATECH", mg)

    mes = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Desembro']

    st.markdown(f"<h1 style='text-align: center; color: black; font-size: {14}px; font-family: sans-serif; font-weight: normal;'>Total de Pedidos no mês de {mes[target_month_2-1]}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: black; font-size: {30}px; font-family: sans-serif; font-weight: normal;'>{total_de_pedidos}</h1>", unsafe_allow_html=True)



    # st.dataframe(df_combinado, use_container_width= True, hide_index=True)

    limite = 5


    fig3 = create_pie_chart(df_combinado, 'Porcentagem (%)', 'cliente', 'Proporção de Pedidos por Cliente no Mês')
    fig4 = create_pie_chart(df_combinado, 'Porcentagem de Peças (%)', 'cliente', 'Proporção de Peças por Cliente no Mês')


    col36,col37 = st.columns([0.5,0.5])

    col36.plotly_chart(fig3, use_container_width=True)
    col37.plotly_chart(fig4, use_container_width=True)




st.markdown("""
    <style>
    /* Centralizar o conteúdo dentro do label do st.metric */
    [data-testid="stMetricLabel"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* Centralizar o conteúdo interno do label */
    [data-testid="stMetricLabel"] div {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }

    /* Centralizar o valor do st.metric */
    [data-testid="stMetricValue"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* Centralizar o conteúdo interno do valor */
    [data-testid="stMetricValue"] div {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }
    /* Centralizar o valor do st.metric */
    [data-testid="stMetricDelta"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* Centralizar o conteúdo interno do valor */
    [data-testid="stMetricDelta"] div {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)