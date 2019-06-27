#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:36:22 2019

@author: roduran
"""

import pandas as pd
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override()
import seaborn as sns; sns.set()

#start date
yearS=2016
monthS=6
dayS=1

#end date
yearE=2019
monthE=1
dayE=20

#ticker de las acciones
tickers = pd.read_csv('ipsa_tickers.csv')
tickers = tickers.drop(tickers.index[[11,16]])
tickers = tickers.reset_index(drop=True)
#tickers y nombre de las acciones
ticker = tickers.Symbol
names = tickers['Company Name']

stocks2 = []

# Descarga las 28 acciones del IPSA y las combina en un solo data frame con los precios de cierre
for i in range(0,len(ticker)):
    stocks = pdr.get_data_yahoo(ticker[i], 
                          start=datetime.datetime(yearS,monthS,dayS), 
                          end=datetime.datetime(yearE, monthE, dayE))
    stocks = stocks[['Adj Close']]
    stocks2.append(stocks)
    stocks2[i].rename(columns={'Adj Close':'%s' % (names[i])}, inplace=True)

stocksadjusted2 = pd.concat(stocks2,axis=1)
stocksadjusted = stocksadjusted2.dropna()


###### Correlación ######

df = stocksadjusted2[['Banco Santander-Chile', 'Inversiones La Construcción S.A.', 'Parque Arauco S.A.',
                      'Enel Chile S.A.', 'Cencosud S.A.', 'Banco de Crédito e Inversiones', 
                      'Colbún S.A.', 'Empresas CMPC S.A.', 'Viña Concha y Toro S.A.', 'AES Gener S.A.']]

df_corr = df.corr()
       

# Plot de correlaciones entre las acciones de la lista.
def visualize_data():
    df_corr = df.corr()
    print(df_corr.head())
    #takes only data (numbers) inside the corr matrix
    data =  df_corr.values
    fig = plt.figure()
    #1 by 1 in plot number one
    ax = fig.add_subplot(1,1,1)
    #customizing the plot showing the correlation in color
    #if theres negative correlation if stock goes up other goes down
    heatmap = ax.pcolor(data,cmap = plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0])+0.5,minor = False)
    ax.set_yticks(np.arange(data.shape[1])+0.5,minor = False)
    #errases any gaps from the graph
    ax.invert_yaxis()
    #moves ticks from xaxis to the top
    ax.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    #limit of the color limit of the heatmap of correlationmatrix
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()

visualize_data()



###### Función Adm Activa ######


def admactiva(N):
    boughtstock = stocksadjusted[names[N]]
#    boughtMA50 = moving50final[names[N]]
#    boughtMA200 = moving200final[names[N]]

    stock = boughtstock
    
    ## Signals (Configurar la ventana corta y larga)
    short_window = 30
    long_window = 75

    # Creamos un DataFrame con el nombre de la accion y una columa llamada signal
    signals = pd.DataFrame(index=stock.index)
    signals['signal'] = 0.0

    # Creamos una media movil sobre la la ventana corta
    signals['short_mavg'] = stock.rolling(window=short_window, min_periods=1, center=False).mean()
    # Creamos una media movil sobre la ventana larga
    signals['long_mavg'] = stock.rolling(window=long_window, min_periods=1, center=False).mean()
    # Creamos las señales
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 
                                                > signals['long_mavg'][short_window:], 1.0, 0.0)   
    # Generamos las ordenes de trading
    signals['positions'] = signals['signal'].diff()
    
    #Incluimos los  precios Adj. Close
    signals['Adj Close'] = boughtstock
    
    # Retorno
    compra = []
    venta = []

    for i in range(0,len(boughtstock)):
        if signals['positions'].iloc[i] == 1:
            compra.append(signals['Adj Close'][i]) 
    
    for i in range(0,len(boughtstock)):
        if signals['positions'].iloc[i] == -1:
            venta.append(signals['Adj Close'][i])
        
        retorno = []
    for i in range(0,len(venta)):
        retorno.append(math.log(venta[i]/compra[i]))
        
    r_total = []
    r_total = np.sum(retorno)
    
    # Plot 
    fig = plt.figure(4)
    # Label
    ax1 = fig.add_subplot(111,  ylabel='Precio en CLP')
    # Plot sobre el precio de cierre
    stock.plot(ax=ax1, color='r', lw=2.)
    # Plot sobre las medias moviles
    signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)
    # Plot de las señales de compra
    ax1.plot(signals.loc[signals.positions == 1.0].index, 
             signals.short_mavg[signals.positions == 1.0],
             '^', markersize=10, color='purple')        
    # Plot sobre las señales de venta
    ax1.plot(signals.loc[signals.positions == -1.0].index, 
             signals.short_mavg[signals.positions == -1.0],
             'v', markersize=10, color='black')         
    # Mostrar Plot
    #plt.show()
    
    return r_total


###### Señales ######
# Santander = N = 2
# ILC = N = 4
# Parauco = N = 10
# ENEL.CHILE = N = 12
# Cenco = N = 15
# BCI = N = 18
# Colbun = N = 19
# AES = N = 20
# CMPC = N = 21
# Concha = N = 22

# Retorno Santander
santanderactiva = admactiva(2)
print (santanderactiva)

# Retorno ILC
ilcactiva = admactiva(4)
print(ilcactiva)

# Retorno Parauco
paraucoactiva = admactiva(10)
print(paraucoactiva)

# Retorno Enel Chile
enelactiva = admactiva(12)
print(enelactiva)

# Retorno Cencosud
cencoactiva = admactiva(15)
print(cencoactiva)

# Retorno BCI
bciactiva = admactiva(18)
print(bciactiva)

# Retorno Colbun
colbunactiva = admactiva(19)
print(colbunactiva)

# Retorno AES
aesactiva = admactiva(20)
print(aesactiva)

# Retorno CMPC
cmpcactiva = admactiva(21)
print(cmpcactiva)

# Retorno Concha y Toro
conchaactiva = admactiva(22)
print(conchaactiva)


portfolio = santanderactiva + ilcactiva + paraucoactiva + enelactiva 
+ cencoactiva + bciactiva + colbunactiva + aesactiva + cmpcactiva + conchaactiva
print(portfolio)


