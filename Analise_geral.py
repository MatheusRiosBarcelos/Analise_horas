# %% IMPORTANDO BIBLIOTECAS
import pandas as pd
from datetime import datetime , timedelta
import seaborn as sns
import matplotlib.pyplot as plt
# %% LENDO E PREPARANDO OS DATAFRAMES
ordens = pd.read_csv('ordens (4).csv', sep = ',')
pedidos = pd.read_csv('pedidos (1).csv', sep = ',')
ordens['ordem'] = ordens['ordem'].fillna(0)
ordens['ordem'] = ordens['ordem'].astype(int)
ordens.loc[:, 'Datetime_ini'] = pd.to_datetime(ordens['data_ini'] + ' ' + ordens['hora_ini'], format = 'mixed', errors='coerce')
ordens.loc[:,'Datetime_fim'] = pd.to_datetime(ordens['data_fim'] + ' ' + ordens['hora_fim'], format = 'mixed', errors='coerce')
ordens.loc[:, 'delta_time_seconds'] = (ordens['Datetime_fim'] - ordens['Datetime_ini']).dt.total_seconds()
ordens.loc[:, 'delta_time_hours'] = ordens['delta_time_seconds'] / 3600
# %% SELECIONAR A MÁQUINA E MÊS
maquina_pesquisada = 'ADM'
new_df = ordens[ordens['estacao'] == maquina_pesquisada]
target_month = 2
df_filtrado = new_df[new_df['Datetime_ini'].dt.month == target_month]
df_filtrado.head(20)
# %% CALCULANDO O TOTAL DE HORAS TRABALHADAS E A PORCENTAGEM DE HORAS TRABALHADAS
hora_esperada_de_trabalho = 160
num_entries = df_filtrado.shape[0]
total_de_horas = round(df_filtrado['delta_time_hours'].sum(), 1)
percent_horas = round((total_de_horas/hora_esperada_de_trabalho) * 100, 1)
print(num_entries)
# %% PRINTANDO O RESULTADO
print(f'Análise horas trabalhadas:{maquina_pesquisada}')
print(f'Total de Horas no mês:{total_de_horas}H')
print(f'Porcentagem de horas trabalhadas:{percent_horas}%')
print(f'Média:{round(total_de_horas / num_entries,2)}H')
# %% SELECIONAR O PEDIDO (PV) E FILTRANDO DA TABELEBA ORDENS
target_pedido = 2024
pedido = pedidos[pedidos['pedido'] == target_pedido]
filtro_df = pedido['ordem']
ordem = ordens[ordens['ordem'].isin(filtro_df)]
# %% CALCULANDO O TOTAL DE HORAS TRABALHADAS POR PEDIDO
total_de_horas_pedido = round(ordem['delta_time_hours'].sum(),2)
print(f'Total de horas gasta no PV {target_pedido} : {total_de_horas_pedido}H')
# %% SEPARANDO POR MÁQUINA QUANTO HORAS FOI GASTA EM CADA UMA
soma_por_estacao = ordem.groupby('estacao')['delta_time_hours'].sum().reset_index().round(2)
soma_por_estacao.rename(columns={'delta_time_hours': 'Tempo de uso (h)'}, inplace=True)
soma_por_estacao.rename(columns={'estacao': 'Estação de Trabalho'}, inplace=True)
nova_linha = {'Estação de Trabalho': 'Total', 'Tempo de uso (h)': total_de_horas_pedido}
soma_por_estacao = pd.concat([soma_por_estacao, pd.DataFrame([nova_linha])], ignore_index=True)
print(soma_por_estacao)
# %% REPRESENTAÇÃO GRÁFICA DA TABELA FEITA LOGO ACIMA
df_plot = soma_por_estacao[soma_por_estacao['Estação de Trabalho'] != 'Total']
sns.set_style("whitegrid")
plt.figure(figsize=(6,6))
plt.pie(df_plot['Tempo de uso (h)'], labels=df_plot['Estação de Trabalho'], autopct='%1.1f%%')
plt.title('Distribuição de Horas por Máquina')
plt.show()