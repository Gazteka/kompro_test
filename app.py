import datetime
import random

import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import pmdarima as pm
import requests
import streamlit as st

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
# df = df.drop(["fecha","up","down","volumen"],axis = 1)
# df.columns = ["open","high","low","close"]
df = df.sort_index(ascending=True)
# resample_dict = {"open":"first",
#                     "high":"max","low":"min",
#                     "close":"last"}
resample_dict = {"close":"last"}
data_1d = df.resample("1D").apply(resample_dict)
data_1d = data_1d.dropna()
data_daily =  data_1d.asfreq("1d").ffill()
last_date = df.index[-1]
last_close = df.iloc[-1]["close"]
st.text(f"Ultima fecha registrada {last_date} con un cierre de {last_close}")

st.header("Evalua el periodo para tus facturas : ")
if st.button("Reiniciar",on_click= set_stage,args = (0,0,0,0,)):
    pass
number = st.number_input('Ingresa la cantidad de dias para pagar la factura',min_value = 1,max_value = 100,step = 1)
if st.session_state.stage ==0:
    if st.button('Evaluar periodo',on_click=set_stage , args=(1,number,0,0,)):
        pass

elif st.session_state.stage == 1:
            
    arima_fit = pm.auto_arima(data_daily["close"])
    arima_fcast = arima_fit.predict(n_periods = number,return_conf_int = True,alpha = 0.05)
    forecast_df = pd.DataFrame()
    forecast_df["predictions"] = arima_fcast[0]
    forecast_df["up"] = arima_fcast[1][:,1]
    forecast_df["down"] = arima_fcast[1][:,0]
    arima_fig = go.Figure(data=[
                        go.Line(x = data_daily.index,y = data_daily["close"],name = "close price"),
                        go.Line(x = forecast_df.index,y = forecast_df["predictions"],name = "prediccion"),
                        go.Line(x = forecast_df.index,y = forecast_df["up"],name = "IC sup 95%"),
                        go.Line(x = forecast_df.index,y = forecast_df["down"],name = "IC inf 95%")])

    ic_up_last = forecast_df.iloc[-1]["up"]
    ic_down_last = forecast_df.iloc[-1]["down"]
    mean_last = forecast_df.iloc[-1]["predictions"]

    st.text(F"""
    Nuestros modelos predicen con un 95% de confianza que para los siguientes {number} dias 
    tu rango de precios se encuentra entre {ic_up_last} y {ic_down_last}
                """)
    st.plotly_chart(arima_fig)
    st.header("Evalua tu riesgo ")

    target = st.number_input('Ingresa el precio deseado',value = int(mean_last),min_value = int(ic_down_last),max_value = int(ic_up_last),step = 1)
    loss = st.number_input('Ingresa el precio de pérdida',value = int(mean_last),min_value = int(ic_down_last),max_value = int(ic_up_last),step = 1)
    if st.button('Evaluar riesgo',on_click = set_stage,args =(2,number,target,loss,)): 
        pass
elif st.session_state.stage == 2:
        
    arima_fit = pm.auto_arima(data_daily["close"])
    arima_fcast = arima_fit.predict(n_periods = number,return_conf_int = True,alpha = 0.05)
    forecast_df = pd.DataFrame()
    forecast_df["predictions"] = arima_fcast[0]
    forecast_df["up"] = arima_fcast[1][:,1]
    forecast_df["down"] = arima_fcast[1][:,0]
    arima_fig = go.Figure(data=[
                        go.Line(x = data_daily.index,y = data_daily["close"],name = "close price"),
                        go.Line(x = forecast_df.index,y = forecast_df["predictions"],name = "prediccion"),
                        go.Line(x = forecast_df.index,y = forecast_df["up"],name = "IC sup 95%"),
                        go.Line(x = forecast_df.index,y = forecast_df["down"],name = "IC inf 95%")])

    ic_up_last = forecast_df.iloc[-1]["up"]
    ic_down_last = forecast_df.iloc[-1]["down"]
    mean_last = forecast_df.iloc[-1]["predictions"]

    st.text(F"""
    Nuestros modelos predicen con un 95% de confianza que para los siguientes {number} dias 
    tu rango de precios se encuentra entre {ic_up_last} y {ic_down_last}
                """)
    st.plotly_chart(arima_fig)
    st.header("Evalua tu riesgo ")
    st.text(f"Has seleccionado tu riesgo entre {st.session_state.target} y {st.session_state.loss}")
    T = st.session_state.n_days
    count = 0
    price_list = []
    last_price = data_daily['close'][-1]
    returns = data_daily['close'].pct_change()
    daily_vol = returns.std()


    np.random.seed(22)
    NUM_SIMULATIONS =  100
    df_montecarlo = pd.DataFrame()
    last_price_list = []
    for x in range(NUM_SIMULATIONS):
        count = 0
        price_list = []
        price = last_price * (1 + np.random.normal(0, daily_vol))
        price_list.append(price)
        
        for y in range(T):
            if count == T-1:
                break
            price = price_list[count]* (1 + np.random.normal(0, daily_vol))
            price_list.append(price)
            count += 1
            
        df_montecarlo[x] = price_list
        last_price_list.append(price_list[-1])

    base = data_daily.index[-1]
    numdays = T+1
    date_list = [base + datetime.timedelta(days=x) for x in range(1,numdays)]
    df_montecarlo.index = date_list

    data = [go.Line(x = data_daily.index,y = data_daily["close"],name = "close price")]

    for i in range(df_montecarlo.shape[1]):
        data.append(go.Line(x = df_montecarlo.index, y =df_montecarlo[i],name = f"sim_{i}"))
    data_daily["target"] = st.session_state.target
    data_daily["loss"] = st.session_state.loss
    data.append(go.Line(x = data_daily.index, y =data_daily["target"],name = f"target"))
    data.append(go.Line(x = data_daily.index, y =data_daily["loss"],name = f"loss"))
    # st.text(df_montecarlo.index)
    montecarlo_fig = go.Figure(data = data)
    st.plotly_chart(montecarlo_fig)

    ultimos_valores = list(df_montecarlo.iloc[-1])
    
    beneficio = [valor for valor in ultimos_valores if valor > st.session_state.target]
    prob_exito = len(beneficio)
    perdida = [valor for valor in ultimos_valores if valor < st.session_state.loss]
    prob_perdida = (len(perdida))

    st.text(f"""Nuestros modelos señalan que en un {prob_exito}% de los casos en los siguientes {T} dias el 
            precio   estará por sobre tu target y que en un {prob_perdida}% 
            de los casos por debajo de tu perdida""")