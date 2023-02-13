import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


#@st.cache_data
#@st.cache_resource
st.set_page_config(page_title="Upload CSV File", page_icon=":folder:", layout="wide")
st.title("Sales Forecast")


template_df = pd.DataFrame({'date':[
    '01/11/2020','01/12/2020','01/01/2021','01/02/2021'],
'sales':[100,200,300,400]
})

# Create a download link for the template file
if st.button("Download Template"):
    st.write("Downloading template...")
    template_df.to_csv("template.csv", index=False)
    with open("template.csv", "rb") as f:
        b = f.read()
    st.write("Done!")
    st.markdown("**Click the following link to download the template:**")
    st.markdown("[Download Template](/files/template.csv?download=True)")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)



st.write("Preview of the sales data:")
st.write(data)


# convert the date column to datetime
data['date'] = pd.to_datetime(data['date'])

# extract additional features
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['day_of_week'] = data['date'].dt.dayofweek

# define the features and target
features = ['year', 'month', 'day', 'day_of_week']
target = 'sales'

#fix this
#train data, those without na
train = data[~data.isna().any(axis=1)]
test = data[data.isna().any(axis=1)]

# train the random forest model
model = RandomForestRegressor()
model.fit(train[features], train['sales'])

forecast_periods = st.slider("Select the number of forecast periods:", min_value=1, max_value=24, value=10, step=1)

#fix this
#predict
forecast = model.predict(test[features].tail(forecast_periods))

#fix this
#create df to show
dates_forecast = data.iloc[-10:].date
dates_forecast.reset_index(drop=True,inplace=True)
forecast = pd.DataFrame(forecast, columns=['sales'])
forecast['date'] = dates_forecast

st.markdown("<p style='font-size:50px'>Forecast results</p>", unsafe_allow_html=True)

st.write(forecast)


#compare actual sales vs forecasting and raise alerts

actual = pd.read_csv('actual_sales.csv')
actual['date'] = pd.to_datetime(actual['date'])
actual = actual[actual['date'].isin(forecast.date)]
actual.reset_index(drop=True,inplace=True)
comparison = actual.copy()
comparison['forecast_sales'] = forecast.sales
comparison['difference'] = comparison.sales - comparison.forecast_sales
comparison['alert'] = comparison['difference'].apply(lambda x: 'yes' if x > 100 else 'no')

st.markdown("<p style='font-size:35px'>Forecast vs Actual Sales</p>", unsafe_allow_html=True)

st.write(comparison)
st.line_chart(comparison, x='date', y=['sales','forecast_sales'])


#alerts

alert = comparison[comparison['alert']=='yes']
st.warning("Alerts for difference between forecasted and actual sales")

st.write("""
    <div style="font-size:20px; color:red;">
    WARNING: Difference between forecasted and actual sales!
    </div>
    """, unsafe_allow_html=True)
st.write(alert)

st.markdown("<p style='font-size:50px'>Stock Alerts</p>", unsafe_allow_html=True)



# when will we run out of stock
stock = pd.read_csv('stock.csv')
stock_comparison = forecast.copy()
stock = stock.dropna()
last_stock = stock.iloc[-1].stock
stock_updated = []
for i in stock_comparison.sales:
    last_stock -= i 
    stock_updated.append(last_stock)
stock_comparison['stock_updated'] = stock_updated
stock_comparison['stock_alert'] = stock_comparison['stock_updated'].apply(lambda x: 'yes' if x < 0 else 'no')
stock_comparison = stock_comparison.rename(columns={'sales':'sales_forecasted'})


alert_stock = stock_comparison[stock_comparison['stock_alert']=='yes']
st.warning("Alerts for running out of stock. When are we running out of stock? (when we are not going to be able to fill the demand, input of stock refill still needed)")
st.write(alert_stock)
st.write("""
    <div style="font-size:20px; color:red;">
    WARNING: Stock is running low!
    </div>
    """, unsafe_allow_html=True)

st.line_chart(stock_comparison,x=['date'], y=['stock_updated','sales_forecasted'])

