import datetime
import os
import random

import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import pmdarima as pm
import requests
import streamlit as st
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title = "QRT Solutions")
st.title("Kompro by QRT Solutions")




api_site = "https://f5ijgk.deta.dev/historical/all"
data = requests.get(api_site).json()

df = pd.DataFrame(data["result"])

df["fecha"] = pd.to_datetime(df["fecha"])
df.index = df["fecha"]
df = df.drop(["fecha"],axis = 1)
df = df.sort_index(ascending=True)
resample_dict = {"close":"last"}
data_1d = df.resample("1D").apply(resample_dict)
data_1d = data_1d.dropna()
data_daily =  data_1d.asfreq("1d").ffill()
last_date = df.index[-1]
last_close = df.iloc[-1]["close"]
st.text(f"Ultima fecha registrada {last_date} con un cierre de {last_close}")




data_daily["r"] = data_daily["close"].pct_change()
indicadores_finales = []
n_lags = 15
for i in range(1,n_lags):
    name = "lag"+str(i)
    data_daily[f"lag{i}"] =  data_daily["r"].shift(i)
    indicadores_finales.append(name)

# st.dataframe(data_daily)

st.header(f"Gráfico de los ultimos {n_lags} días")

last_data_fig = go.Figure(data=[
                        go.Line(x = data_daily.index[-n_lags:],y = data_daily["close"].iloc[-n_lags:],name = "close price")])
        #                 go.Line(x = forecast_df.index,y = forecast_df["predictions"],name = "prediccion"),
        #                 go.Line(x = forecast_df.index,y = forecast_df["up"],name = "IC sup 95%"),
        #                 go.Line(x = forecast_df.index,y = forecast_df["down"],name = "IC inf 95%")])
st.plotly_chart(last_data_fig)
X = data_daily[indicadores_finales].dropna().values
n_vecinos = 4

neigh = NearestNeighbors(n_neighbors=n_vecinos)
neigh.fit(X)
sample = X[20]
cercanos = neigh.kneighbors([sample], return_distance=False)
st.text(cercanos)
nearest = X[cercanos,:n_lags]
for i in range(nearest.shape[0]):
    dia = nearest[i]


dias_similares = []
for i in range(n_vecinos):
    prueba = nearest[0,i,:]
    dia_muestra = data_daily[data_daily["lag1"]== prueba[0]].index[0]
    dias_similares.append(dia_muestra)
st.header("Ver días similares y lo que ocurrió")
st.write(f"""Nuestros modelos de machine learning han determinado que a través de un modelo de 
[KNN](https://es.wikipedia.org/wiki/K_vecinos_m%C3%A1s_pr%C3%B3ximos) los {n_vecinos} días más parecidos al de hoy son :""")
# st.text(dias_similares)
fecha_seleccionada = st.selectbox("Selecciona una fecha para graficar",dias_similares)


antes = data_daily["r"].loc[:fecha_seleccionada].iloc[-n_lags:]

despues = data_daily["r"].loc[fecha_seleccionada:].iloc[:n_lags]
final = pd.concat([antes,despues]).reset_index(drop=True)
final  =( 1+final).cumprod()
data_fig = go.Figure(data=[
                        go.Line(x = final.index[:n_lags],y = final[:n_lags],name = "Similitud"),
                        go.Line(x = final.index[-n_lags:],y = final[-n_lags:],name = "Posterior")
                        ])
data_fig.add_vline(x=n_lags, line_width=3, line_dash="dash", line_color="green")
st.plotly_chart(data_fig)

