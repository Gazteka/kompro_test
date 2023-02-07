import datetime
import random

import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import pmdarima as pm
import requests
import streamlit as st
import os


if 'stage' not in st.session_state:
    st.session_state.stage = 0

def set_stage(stage,number,target,loss):
    st.session_state.stage = stage
    st.session_state.n_days = number
    st.session_state.target = target
    st.session_state.loss = loss

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
# ruta_ordenes = os.path.join("Users","HP","Desktop","QRT","Kompro","ordenes.csv")
# ruta = os.path.join("C:\\Users","HP","Desktop","QRT","Kompro","ordenes.csv")
ruta = "ordenes.csv"

data_ordenes = pd.read_csv(ruta)
data_ordenes.index =data_ordenes["id"]
data_ordenes = data_ordenes.drop(["id"],axis = 1)
data_ordenes["created_at"] = pd.to_datetime(data_ordenes["created_at"],utc= True)
data_ordenes["deadline"] = pd.to_datetime(data_ordenes["deadline"],utc = True)
orden_seleccionada = st.selectbox("Selecciona una orden para el seguimiento",data_ordenes.index)
orden_seleccionada = data_ordenes.loc[orden_seleccionada]
fecha_inicio = orden_seleccionada["created_at"]
fecha_final = orden_seleccionada["deadline"]
ultima_guardada = data_daily.index[-1]
st.text(f"Fecha de pago : {fecha_final}")
data_daily = data_daily.iloc[-70:]
if fecha_final > ultima_guardada:
    n_days = (fecha_final - fecha_inicio).days
    df_kompro = data_daily.loc[fecha_inicio:fecha_final]
    df_kompro["target_price"] = int(orden_seleccionada["target_price"])
    df_kompro["loss_price"] = int(orden_seleccionada["loss_price"])
    arima_fit = pm.auto_arima(data_daily["close"].loc[:str(fecha_inicio)])
    arima_fcast = arima_fit.predict(n_periods = n_days,return_conf_int = True,alpha = 0.05)
    forecast_df = pd.DataFrame()
    forecast_df["predictions"] = arima_fcast[0]
    forecast_df["up"] = arima_fcast[1][:,1]
    forecast_df["down"] = arima_fcast[1][:,0]
    fig = go.Figure(data=[
                        go.Line(x = data_daily.index,y = data_daily["close"],name = "close price"),
                        # go.Line(x = forecast_df.index,y = forecast_df["predictions"],name = "prediccion"),
                        go.Line(x = forecast_df.index,y = forecast_df["up"],name = "IC sup 95%"),
                        go.Line(x = forecast_df.index,y = forecast_df["down"],name = "IC inf 95%"),
                        go.Line(x = data_daily.index,y = data_daily["close"],name = "close price"),
                        go.Line(x = df_kompro.index,y = df_kompro["target_price"],name = "target"),
                        go.Line(x = df_kompro.index,y = df_kompro["loss_price"],name = "loss")])


    fig.add_vline(x=str(fecha_inicio), line_width=3, line_dash="dash", line_color="green")
    st.plotly_chart(fig)
    st.header("Distancia al objetivo")
    df_kompro["win_factor"] = (df_kompro["close"]/df_kompro["target_price"])**-1
    fig_win = go.Figure(data=[ go.Line(x =list(range(df_kompro.shape[0])),y = df_kompro["win_factor"],name = "win_factor"),
                    go.Line(x =list(range(df_kompro.shape[0])),y = np.ones(df_kompro.shape[0]),name = "Best price")   
                                     ])

    st.plotly_chart(fig_win)
    last_factor = df_kompro.iloc[-1]["win_factor"]
    if last_factor < 1.01:
        st.text(f"""Ultimo valor de distancia: {last_factor}.
Estas cerca de tu objetivo, atento a tus oportunidades de compra""")
    else : 
        st.text(f"""Ultimo valor de distancia: {last_factor}.
Te estas alejando de tu objetivo no pierdas de vista tu riesgo""")
    

else:
    st.text((fecha_final > ultima_guardada))