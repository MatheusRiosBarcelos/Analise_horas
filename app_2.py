import streamlit as st
import pandas as pd
import plotly.express as px
from datetime  import datetime as dt , timedelta
import numpy as np
import plotly.graph_objects as go
import requests
import pytz

def convert_to_HM(x):
    hours = float(x)
    h = int(hours)
    m = int((hours - h) * 60)
    return f"{h}:{m:02d}"

def convert_to_float(hm):
    h, m = map(int, hm.split(':'))
    return h + m / 60

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

def get_hours_expected(estacao):
    return {
        'SOLDAGEM': 1100,
        'FRESADORAS': 440,
        'TORNO CONVENCIONAL': 440,
        'ACABAMENTO': 440
    }.get(estacao, 220)

def inserir_hifen(valor):
    if valor in ['HVHV30716401-1', 'HVHV30716401-2']:
        prefixo, sufixo = valor.split('-')
        novo_prefixo = prefixo[:-2] + '-' + prefixo[-2:]
        return f"{novo_prefixo}-{sufixo}"
    return valor

def get_last_commit_date(repo_owner, repo_name):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits"
    response = requests.get(url)
    
    # Verificar o status da resposta
    if response.status_code != 200:
        raise ValueError(f"Failed to retrieve commits. Status code: {response.status_code}")
    
    commits = response.json()
    
    # Imprimir a resposta para diagnóstico
    print("Response JSON:", commits)
    
    if isinstance(commits, list) and len(commits) > 0:
        last_commit = commits[0]
        # Verificar a presença dos campos esperados
        if 'commit' in last_commit and 'committer' in last_commit['commit'] and 'date' in last_commit['commit']['committer']:
            commit_date = last_commit['commit']['committer']['date']
            return dt.strptime(commit_date, "%Y-%m-%dT%H:%M:%SZ")
        else:
            raise ValueError("Expected fields are missing in the commit data.")
    else:
        raise ValueError("No commits found or unexpected response format.")

def convert_to_brasilia_time(utc_datetime):
    utc_zone = pytz.utc
    brasilia_zone = pytz.timezone('America/Sao_Paulo')
    utc_datetime = utc_zone.localize(utc_datetime)
    brasilia_datetime = utc_datetime.astimezone(brasilia_zone)
    return brasilia_datetime

st.set_page_config(layout="wide") 
colA, colB = st.columns([0.8,0.2])

repo_owner = "MatheusRiosBarcelos"
repo_name = "Analise_horas"
try:
    last_commit_date = get_last_commit_date(repo_owner, repo_name)
    print(last_commit_date)
except ValueError as e:
    print(e)

with colA:
    st.image('logo.png', width= 150)

with colB:
    if last_commit_date:
        last_commit_date_brasilia = convert_to_brasilia_time(last_commit_date)
        st.write(f"Última atualização: {last_commit_date_brasilia.strftime('%d/%m/%Y %H:%M:%S')}")
    else:
        st.write("Não foi possível obter a data do último commit.")


ordens = pd.read_csv('ordens (4).csv', sep=',')
pedidos = pd.read_csv('pedidos (1).csv', sep=',')
orc = pd.read_excel('Processos_de_Fabricacao.xlsx')

pedidos['codprod'] = pedidos['codprod'].apply(inserir_hifen)
pedidos["entrega"] = pd.to_datetime(pedidos["entrega"], format = 'mixed', errors='coerce')


ordens = ordens[ordens['estacao'] != 'Selecione...']

ordens.fillna({'ordem': 0, 'data_ini': 0, 'hora_ini': 0}, inplace=True)
ordens['ordem'] = ordens['ordem'].astype(int)

ordens['Datetime_ini'] = pd.to_datetime(ordens['data_ini'] + ' ' + ordens['hora_ini'], errors='coerce')
ordens['Datetime_fim'] = pd.to_datetime(ordens['data_fim'] + ' ' + ordens['hora_fim'], errors='coerce')

ordens['delta_time_seconds'] = (ordens['Datetime_fim'] - ordens['Datetime_ini']).dt.total_seconds()
ordens['delta_time_hours'] = ordens['delta_time_seconds'] / 3600
ordens['delta_time_min'] = ordens['delta_time_seconds'] / 60

substituicoes = {'MQS': 'SOLDAGEM','JPS': 'JATO','PLM001': 'PLM 001','PLM 01': 'PLM 001','SFH': 'CORTE - SERRA','SRC': 'CORTE - SERRA','TCNV': 'TORNO CONVENCIONAL','TCNC': 'TORNO CNC','LASER': 'CORTE-LASER','MCL': 'CORTE-LASER','PLM': 'CORTE-PLASMA','GLT': 'CORTE-GUILHOTINA','DGQ': 'QUALIDADE','FRZ 033': 'FRZ 003','FRZ003': 'FRZ 003','CNC 001': 'CENTRO DE USINAGEM','CCNC 001': 'CENTRO DE USINAGEM','CCNC001': 'CENTRO DE USINAGEM','CCNC01': 'CENTRO DE USINAGEM','Bancada': 'ACABAMENTO','BANCADA': 'ACABAMENTO','AJT': 'ACABAMENTO','FRZ': 'FRESADORAS','Acabamento': 'ACABAMENTO','DHCNC': 'DOBRADEIRA','DHCN': 'DOBRADEIRA','DBEP': 'PRENSA (AMASSAMENTO)'}

for key, value in substituicoes.items():
    ordens.loc[ordens['estacao'].str.contains(key, na=False), 'estacao'] = value

ordens["data_ini"] = pd.to_datetime(ordens["data_ini"], errors='coerce')
ordens["data_fim"] = pd.to_datetime(ordens["data_fim"], errors='coerce')
ordens["Ano"] = ordens["data_ini"].dt.year.astype('Int64')
ordens["Mes"] = ordens["data_ini"].dt.month.astype('Int64')

ordens['delta_dia'] = (ordens['data_fim'] - ordens['data_ini']).dt.days
ordens['hora_fim'] = pd.to_datetime(ordens['hora_fim'], format='%H:%M:%S', errors='coerce').dt.time
ordens['hora_ini'] = pd.to_datetime(ordens['hora_ini'], format='%H:%M:%S', errors='coerce').dt.time

midnight = ordens['Datetime_fim'].dt.normalize()
seven_am = midnight + pd.Timedelta(hours=7)
condition = (ordens['delta_dia'] == 1) & (ordens['Datetime_ini'] < midnight) & (ordens['Datetime_fim'] <= seven_am)
ordens.loc[condition, 'delta_dia'] = 0

ordens['weekends_count'] = ordens.apply(lambda row: count_weekend_days(row['data_ini'], row['data_fim']), axis=1)
ordens = ordens.apply(adjust_delta_time_hours, axis=1)
ordens = ordens.apply(adjust_delta_time, axis=1)
ordens = ordens[ordens['delta_time_hours'] >= 0]
    
ordens = ordens.sort_values("data_ini")

tab1, tab2, tab3, tab4 = st.tabs(["ANÁLISE HORA DE TRABALHO MENSAL", "ANÁLISE HORA DE TRABALHO POR PV", "TEMPO MÉDIO PARA A FARBICAÇÃO DE PRODUTOS ", 'ANÁLISE MENSAL DE PEDIDOS'])

with tab1:
    with st.sidebar:
        date_now = dt.now()
        estacao = st.selectbox("Estação", ordens["estacao"].sort_values().unique(), placeholder='Escolha uma opção')
        target_month = st.selectbox("Mês", pedidos["entrega"].dt.month.dropna().astype(int).sort_values().unique(), key=1, index=date_now.month, placeholder='Escolha uma opção')
        target_year = st.selectbox("Ano", ordens["Ano"].sort_values().unique(), key=2, index=0, placeholder='Escolha uma opção')
    
    new_df = ordens[ordens['estacao'] == estacao]
    df_filtrado_year = new_df[new_df['Datetime_ini'].dt.year == target_year]
    df_filtrado = df_filtrado_year[df_filtrado_year['Datetime_ini'].dt.month == target_month]

    hora_esperada_de_trabalho = get_hours_expected(estacao)
    num_entries = df_filtrado.shape[0]
    total_de_horas = round(df_filtrado['delta_time_hours'].sum(), 1)
    percent_horas = int((total_de_horas / hora_esperada_de_trabalho) * 100)
    media = np.divide(total_de_horas, num_entries, out=np.zeros_like(total_de_horas), where=num_entries != 0).round(1)
    delta_1 = round(percent_horas - 100, 1)
    
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Total de Horas da Máquina {estacao} em {target_month}-{target_year}", f"{total_de_horas}H", f'{round(total_de_horas-hora_esperada_de_trabalho,1)}H')
    col2.metric(f'Eficiência (%) da Máquina {estacao}', f'{percent_horas}%', f'{delta_1}%')
    col3.metric(f"Média da Máquina {estacao}", f"{media}H")
    
    pedidos['codprod'] = pedidos['codprod'].astype(str)
    orc['CODIGO'] = orc['CODIGO'].astype(str)

    pedidos_orc = pedidos[(pedidos['entrega'].dt.month == target_month) & (pedidos['entrega'].dt.year == target_year)]
    pedidos_orc = pedidos_orc.merge(orc, left_on='codprod', right_on='CODIGO', how='left').dropna(subset=['CODIGO'])
    pedidos_orc['TOTAL'] = pedidos_orc['TOTAL'] * pedidos_orc['quant_a_fat']
    total_de_horas_orcadas = (pedidos_orc['TOTAL'].sum() / 60).round(0)

    ordens_orc = ordens[(ordens['data_ini'].dt.month == target_month) & (ordens['data_ini'].dt.year == target_year)]
    total_de_horas_trabalhadas = ordens_orc['delta_time_hours'].sum().round(0)

    colunas = ['ACABAMENTO', 'CORTE - SERRA', 'CORTE-PLASMA', 'CORTE-LASER', 'CENTRO DE USINAGEM','DOBRADEIRA','PRENSA (AMASSAMENTO)', 'FRESADORAS','TORNO CONVENCIONAL', 'TORNO CNC','MONTAGEM','SOLDAGEM']

    for index, row in pedidos_orc.iterrows():
        for coluna in colunas:
            if not pd.isna(row[coluna]):
                # Use .loc to access the specific cell and assign the value
                pedidos_orc.loc[index, coluna] = round((row[coluna]*row['quant_a_fat'])/60,0)
    
    mapa_maquinas = {'CORTE - SERRA': 'CORTE - SERRA','FRESADORAS': 'FRESADORAS','CORTE-PLASMA': 'CORTE-PLASMA','CORTE-LASER': 'CORTE-LASER','CORTE-GUILHOTINA': 'CORTE-GUILHOTINA','TORNO CONVENCIONAL': 'TORNO CONVENCIONAL','TORNO CNC': 'TORNO CNC','CENTRO DE USINAGEM': 'CENTRO DE USINAGEM','SOLDAGEM': 'SOLDAGEM','ACABAMENTO': 'ACABAMENTO','DOBRA': 'DOBRADEIRA','PRENSA (AMASSAMENTO)' : 'PRENSA (AMASSAMENTO)','JATO' : 'JATEAMENTO','MONTAGEM':'MONTAGEM'}
    maquina = next((valor for chave, valor in mapa_maquinas.items() if chave in estacao), 'DESCONHECIDA')
    total_de_horas_orcadas_maquina = pedidos_orc[maquina].sum()

    col53,col54, col55 = st.columns(3)

    disp_tempo_maquina = get_hours_expected(estacao)

    col53.metric(f"Total de horas orçadas em {target_month}-{target_year} para {maquina}", f"{total_de_horas_orcadas_maquina}H",f'{round((total_de_horas_orcadas_maquina-disp_tempo_maquina)*-1,1)}H')
    col54.metric(f"Total de horas Trabalhadas {target_month}-{target_year}", f"{total_de_horas_trabalhadas}H")
    col55.metric(f"Total de horas Orçadas {target_month}-{target_year}", f"{total_de_horas_orcadas}H", f'{round((total_de_horas_orcadas-3300)*-1,1)}H')

    col11,col12,col13 = st.columns([0.30,0.30,0.4])

    ordem_2 = df_filtrado_year.groupby(['estacao', df_filtrado_year['Datetime_ini'].dt.month])['delta_time_hours'].sum().reset_index().round(2)
    ordem_2.rename(columns={'delta_time_hours': 'Tempo de uso total (H)', 'Datetime_ini': 'Mês'}, inplace=True)
    ordem_2['Tempo de uso total (H)'] = (ordem_2['Tempo de uso total (H)']/hora_esperada_de_trabalho*100).astype(int)
    ordem_2['Tempo de uso total (H)_label'] = ordem_2['Tempo de uso total (H)'].apply(lambda x: f"{x:.0f}%")

    fig2 = px.bar(ordem_2, x = 'Mês', y ='Tempo de uso total (H)' ,title= f'Eficiência Mensal {estacao} (%)',text='Tempo de uso total (H)_label', width=350, height=600)
    fig2.update_traces(textfont_size=16, textangle=0, textposition="outside", cliponaxis=False, marker_color='#e53737')
    fig2.update_layout(yaxis_title = 'Eficiência (%)', title_x = 0.55, title_y = 0.95,title_xanchor = 'center',xaxis=dict(tickfont=dict(size=14)))
    fig2.update_xaxes(tickvals=list(range(len(ordem_2)+1)))
    col11.plotly_chart(fig2)

    x = ordens[ordens['Datetime_ini'].dt.year == target_year]
    x = x.groupby(['estacao', x['Datetime_ini'].dt.month])['delta_time_hours'].sum().reset_index().round(2)
    x = x[x['estacao'].isin(['CORTE - SERRA', 'TORNO CONVENCIONAL', 'TORNO CNC', 'FRESADORAS', 'CENTRO DE USINAGEM', 'CORTE-PLASMA', 'CORTE-LASER', 'CORTE-GUILHOTINA', 'DOBRA', 'SOLDAGEM', 'ACABAMENTO'])]
    y = x.groupby('Datetime_ini')['delta_time_hours'].sum().reset_index().round(2)
    y['delta_time_hours'] = ((y['delta_time_hours'] / 3300) * 100).round(2)
    y['delta_time_hours_label'] = y['delta_time_hours'].apply(lambda x: f"{x:.0f}%")

    fig21 = px.bar(y, x = 'Datetime_ini', y = 'delta_time_hours',title= f'Eficiência Mensal Total da Fábrica (%)',text='delta_time_hours_label', width=350, height=600)
    fig21.update_traces(textfont_size=16, textangle=0, textposition="outside", cliponaxis=False, marker_color='#e53737')
    fig21.update_layout(yaxis_title = 'Eficiência (%)', xaxis_title = 'Mês', title_x = 0.55, title_y = 0.95,title_xanchor = 'center',xaxis=dict(tickfont=dict(size=14)))
    fig21.update_xaxes(tickvals=list(range(len(y)+1)))
    col12.plotly_chart(fig21)

    fig20 = go.Figure(data=go.Heatmap(
            z=x['delta_time_hours'],
            x=x['Datetime_ini'],
            y=x['estacao'],
            colorscale='Reds'),
            )

    fig20.update_layout(title='Mapa de Calor Horas trabalhadas Mensalmente', width=500, height= 500,title_x = 0.55, title_y = 0.95,title_xanchor = 'center', xaxis_title = 'Mês',xaxis=dict(tickfont=dict(size=14)))
    fig20.update_xaxes(tickvals=list(range(len(x)+1)))

    somas = [pedidos_orc[coluna].sum() for coluna in colunas]

    limites = {'FRESADORAS': 440,'CORTE - SERRA': 440,'CORTE-PLASMA': 220,'CORTE-LASER': 220,'TORNO CONVENCIONAL': 440,'TORNO CNC': 220,'CENTRO DE USINAGEM': 220,'DOBRADEIRA': 220,'SOLDAGEM': 1100,'ACABAMENTO' : 440, 'MONTAGEM': 440, 'PRENSA (AMASSAMENTO)':220}
    
    df_somas = pd.DataFrame({'Estação': colunas, 'Horas Orçadas': somas})
    df_somas['Limite de Horas'] = df_somas['Estação'].map(limites)
    df_somas['Horas Restantes'] = df_somas['Limite de Horas'] - df_somas['Horas Orçadas']
    df_long = df_somas.melt(id_vars='Estação', value_vars=['Horas Orçadas', 'Horas Restantes'], var_name='Tipo', value_name='Horas')

    fig_71 = px.bar(df_long, x='Estação', y='Horas', color='Tipo', text_auto='.2s', color_discrete_sequence=['#e53737', '#FFCECE'])
    fig_71.update_layout(width=800, height=700, title_x=0.45, title_y=0.95, title_xanchor='center', xaxis=dict(tickfont=dict(size=14)), legend=dict(font=dict(size=14),orientation = 'h',yanchor='top',y=-0.25,xanchor='center',x=0.5), title=dict(text=f'Horas Orçadas e Restantes por Estação no mês {target_month}', font=dict(size=18)))
    fig_71.update_traces(textfont_size=18, textangle=0, textposition="outside", cliponaxis=False)
 
    col71,col72 = st.columns([0.9,0.1])

    col13.plotly_chart(fig_71)
    
    col71.plotly_chart(fig20)

with tab2:
    col9,col10 = st.columns([0.2,0.8])
    
    col4, col5 = st.columns(2)

    with col9:
        target_pv = st.selectbox("Selecione o PV", pedidos["pedido"].sort_values().unique(),index=1200,placeholder ='Escolha uma opção')

    pedido = pedidos[pedidos['pedido'] == target_pv]
    descricao = pedido.iloc[0, 14]

    with col10:
        st.markdown(f"<h1 style='text-align: left;'>{descricao}</h1>", unsafe_allow_html=True)

    quant = pedido['quant_a_fat'].iloc[0]
    filtro_df = pedido['ordem']
    ordem = ordens[ordens['ordem'].isin(filtro_df)]
    codprod = pedido['codprod'].iloc[0]
    ordem = ordem.dropna(subset=['estacao', 'delta_time_hours'])

    soma_por_estacao = ordem.groupby('estacao')['delta_time_hours'].sum().reset_index().round(2)
    soma_por_estacao.rename(columns={'delta_time_hours': 'Tempo de uso total (H:M)', 'estacao': 'Estação de Trabalho'}, inplace=True)

    fig = px.pie(soma_por_estacao, values='Tempo de uso total (H:M)', names='Estação de Trabalho', title='Proporção de Tempo de Uso por Máquina em Cada Pedido', width=800, height=500)
    fig.update_layout(title_yref='container',title_xanchor = 'center',title_x = 0.43, title_y = 0.95, legend=dict(font=dict(size=18)),font=dict(size=20), title_font=dict(size=20))

    total_de_horas_pedido = soma_por_estacao['Tempo de uso total (H:M)'].sum()

    nova_linha = {'Estação de Trabalho': 'TOTAL', 'Tempo de uso total (H:M)': total_de_horas_pedido}
    soma_por_estacao = pd.concat([soma_por_estacao, pd.DataFrame([nova_linha])], ignore_index=True)

    orc_codprod = orc[orc['CODIGO'] == codprod]

    tempo_esperado = {
        'CORTE-PLASMA': None, 'CORTE - SERRA': None, 'CORTE-LASER': None, 'CORTE-GUILHOTINA': None,
        'TORNO CONVENCIONAL': None, 'TORNO CNC': None, 'FRESADORAS': None, 'CENTRO DE USINAGEM': None,
        'PRENSA (AMASSAMENTO)': None, 'DOBRADEIRA': None, 'ROSQUEADEIRA': None, 'FURADEIRA DE BANCADA': None,
        'SOLDAGEM': None, 'ACABAMENTO': None, 'JATEAMENTO': None, 'PINTURA': None, 'MONTAGEM': None, 'CALANDRA': None,
        'TOTAL': None
    }

    if not orc_codprod.empty:
        for index, row in orc_codprod.iterrows():
            for key in tempo_esperado.keys():
                if not pd.isna(row[key]):
                    tempo_esperado[key] = convert_to_HM((row[key] / 60) * quant)

    estacoes_todas = pd.DataFrame(list(tempo_esperado.keys()), columns=['Estação de Trabalho'])

    soma_por_estacao = estacoes_todas.merge(soma_por_estacao, on='Estação de Trabalho', how='left')
    soma_por_estacao['Tempo de uso total (H:M)'].fillna(0, inplace=True)
    soma_por_estacao['Tempo de uso total (H:M)'] = soma_por_estacao['Tempo de uso total (H:M)'].apply(lambda x: convert_to_HM(x) if x != 0 else "Não Apontado")
    soma_por_estacao['Tempo esperado no Orçamento'] = soma_por_estacao['Estação de Trabalho'].map(tempo_esperado)
    soma_por_estacao = soma_por_estacao[~((soma_por_estacao['Tempo de uso total (H:M)'] == 'Não Apontado') & (soma_por_estacao['Tempo esperado no Orçamento'].isna()))]

       
    col17,col18 = st.columns([0.9,0.1])

    soma_por_estacao  = soma_por_estacao[soma_por_estacao['Estação de Trabalho'] != 'ADM']
    soma_por_estacao  = soma_por_estacao[soma_por_estacao['Estação de Trabalho'] != 'QUALIDADE']
    soma_por_estacao = soma_por_estacao.reset_index(drop=True)
    
    col4.plotly_chart(fig, use_container_width=True)
    with col5:
        st.markdown(f"<h1 style='font-size: 20px;'>Tabela de Horas por Estação no PV {target_pv}/Número de peças é {quant}</h1>", unsafe_allow_html=True)
    
    header_styles = {
    'selector': 'th.col_heading',
    'props': [('background-color', 'lightblue'), 
              ('color', 'black'),
              ('font-size', '14px'),
              ('font-weight', 'bold')]
}
    col5.table(soma_por_estacao.style.set_table_styles([header_styles]))
    
    col60,col61 = st.columns([0.9,0.1])
    colunas_selecionadas = ['ordem', 'estacao', 'nome_func', 'Datetime_ini', 'Datetime_fim']
    ordem['ordem'] = ordem['ordem'].astype(str)
    ordem_colunas_selecionas = ordem[colunas_selecionadas]
    ordem_colunas_selecionas = ordem_colunas_selecionas.reset_index(drop=True)
    ordem_colunas_selecionas.rename(columns={'ordem': 'Ordem', 'nome_func': 'Nome Colaborador', 'Datetime_ini': 'Data/Hora Inicial', 'Datetime_fim': 'Data/Hora Final'}, inplace=True)

    with col60:
        col60.table(ordem_colunas_selecionas.style.set_table_styles([header_styles]))
    
with tab3:
    codprod_target = st.text_input("Código do Produto", value= 'HVHV307164-01')
    number_parts = st.number_input("Quantas peças são", value=int(1), placeholder="Type a number...")

    pedido_cod = pedidos[pedidos['codprod'].str.contains(codprod_target, na=False)]
    filtro_df_cod = pedido_cod['ordem']
    ordem_cod = ordens[ordens['ordem'].isin(filtro_df_cod)]
    ordem_cod = ordem_cod[ordem_cod['data_ini'].dt.year == 2024]
    merged_df = pd.merge(pedido_cod, ordem_cod, on='ordem', how='left')
    merged_df['delta_time_hours'] = merged_df['delta_time_hours'] / merged_df['quant_a_fat']
    
    merged_df['delta_time_hours'] = merged_df['delta_time_hours'].replace([np.inf, -np.inf], np.nan)

    index_of_first_occurrence = merged_df[((~merged_df['descricao'].isna()))].index[0]
    descricao_2 = merged_df.loc[index_of_first_occurrence, 'descricao']

    merged_df = merged_df.dropna(subset=['estacao', 'delta_time_hours'])
    merged_df = merged_df.groupby('estacao')['delta_time_hours'].mean().reset_index().round(2)
    merged_df['delta_time_hours'] = merged_df['delta_time_hours'] * number_parts
    total_hours = merged_df['delta_time_hours'].sum(skipna=True)
    total_hours_rounded = round(total_hours, 2)
    tempo_total_medio = convert_to_HM(total_hours_rounded)

    if 'delta_time_hours' in merged_df.columns and merged_df['delta_time_hours'].notnull().all():
        merged_df['delta_time_hours'] = merged_df['delta_time_hours'].apply(convert_to_HM)
    
    merged_df.rename(columns={'delta_time_hours': 'Tempo Médio de Uso (H:M)', 'estacao': 'Operação'}, inplace=True)

    operacoes_excluir = ['ADM', 'QUALIDADE', 'INSPEÇÃO DE QUANTIDA']
    merged_df = merged_df[~merged_df['Operação'].isin(operacoes_excluir)]
    if ('Corte-Plasma' in merged_df['Operação'].values) and ('Corte-Laser' in merged_df['Operação'].values):
        merged_df = merged_df[merged_df['Operação'] != 'Corte-Plasma']

    nova_linha_2 = {'Operação': 'TOTAL', 'Tempo Médio de Uso (H:M)': tempo_total_medio}
    merged_df = pd.concat([merged_df, pd.DataFrame([nova_linha_2])], ignore_index=True)

    orc_codprod = orc[orc['CODIGO'] == codprod_target]
    
    if not orc_codprod.empty:
        merged_df['Tempo no Orçamento'] = pd.Series([np.nan] * len(merged_df))
        for index, row in orc_codprod.iterrows():
            for key in tempo_esperado.keys():
                if key in orc_codprod.columns and not pd.isna(row[key]):
                    tempo_esperado[key] = convert_to_HM((row[key] / 60) * number_parts)

    for estacao, tempo in tempo_esperado.items():
        if (merged_df['Operação'] == estacao).any():
            merged_df.loc[merged_df['Operação'] == estacao, 'Tempo no Orçamento'] = tempo

    st.markdown(f"<h1 style='text-align: left;'>{descricao_2}</h1>", unsafe_allow_html=True)

    col20,col21 = st.columns([0.9,0.1])

    col20.table(merged_df.style.set_table_styles([header_styles]))
    
with tab4:
    pedidos_1 = pedidos.drop_duplicates(subset=['pedido'], keep='first')
    pedidos_1["entrega"] = pd.to_datetime(pedidos_1["entrega"], format='mixed', errors='coerce')
    pedidos_1 = pedidos_1[pedidos_1['entrega'].dt.month == target_month]
    pedidos_1 = pedidos_1[pedidos_1['entrega'].dt.year == target_year]

    pedidos_1.loc[pedidos['cliente'].str.contains('WEG', na=False), 'cliente'] = 'WEG'
    pedidos_1.loc[pedidos['cliente'].str.contains('GE', na=False), 'cliente'] = 'GE'

    pedidos_clientes = pedidos_1.groupby('cliente').size().reset_index(name='Quantidade de Pedidos')
    pedidos_clientes.sort_values(by='Quantidade de Pedidos', ascending=False, inplace=True)
    pedidos_clientes.reset_index(drop=True, inplace=True)

    total = pedidos_clientes['Quantidade de Pedidos'].sum()
    pedidos_clientes['Porcentagem (%)'] = ((pedidos_clientes['Quantidade de Pedidos'] / total) * 100).round(2)
    
    pedidos_pecas = pedidos_1.groupby('cliente')['quant_a_fat'].sum().reset_index()
    pedidos_pecas.sort_values(by='quant_a_fat', ascending=False, inplace=True)
    pedidos_pecas.reset_index(drop=True, inplace=True)
    
    df_combinado = pedidos_clientes.merge(pedidos_pecas[['cliente', 'quant_a_fat']], on='cliente', how='left')
    total_pecas = df_combinado['quant_a_fat'].sum()
    df_combinado['Porcentagem de Peças (%)'] = ((df_combinado['quant_a_fat'] / total_pecas) * 100).round(2)
    df_combinado.rename(columns={'quant_a_fat': 'Quantidade de peças por cliente'}, inplace=True)
    df_combinado.sort_values(by='cliente', ascending=False, inplace=True)
    
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

    st.markdown(f"<h1 style='text-align: center; color: black; font-size: {14}px; font-family: sans-serif; font-weight: normal;'>Total de Pedidos no mês de {mes[target_month-1]}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: black; font-size: {30}px; font-family: sans-serif; font-weight: normal;'>{total_de_pedidos}</h1>", unsafe_allow_html=True)

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