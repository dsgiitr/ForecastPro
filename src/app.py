import streamlit as st
from run_forecast import FetchStocks, LSTNetModel, LSTNetIteratedModel
import matplotlib.pyplot as plt
import os
curr_path = os.path.dirname(os.path.abspath(__file__))
# Define functions for fetching, training, and getting forecasts
def fetch_stocks():
    fetcher = FetchStocks()
    fetcher.csv_path = curr_path+"/" +fetcher.csv_path
    return fetcher.csv_path 

def train_lstnet_iterated(csv_path):
    lstnet_iterated = LSTNetIteratedModel(n_series=48, forecast_steps=15, csv_path=csv_path)
    lstnet_iterated.train_model()
    return lstnet_iterated

def get_lstnet_forecast(model):
    return model.get_forecast()

def plot_lstnet_forecast(model, series_index):
    fig, ax = plt.subplots()
    model.plot_lstnet_forecast(series_index, ax=ax)
    return fig

# Streamlit app
def main():
    st.title("ForecastPro")
    saved_path =None
    if st.button("Fetch Data"):
        saved_csv_path = fetch_stocks()
        st.success("Stocks fetched successfully!")
        st.write(f"out put saved at {saved_csv_path}")
        saved_path= saved_csv_path
    if st.button("Train LSTNet Iterated Model"):
        lstnet_model = train_lstnet_iterated(saved_path)
        st.success("LSTNet Iterated Model trained successfully!")
    if st.button("Get Forecast"):
        forecast = get_lstnet_forecast(lstnet_model)
        st.success("Forecast generated successfully!")

    if st.button("Plot Forecast"):
        series_index = st.number_input("Enter series index:", min_value=0, max_value=47, value=12)
        fig = plot_lstnet_forecast(lstnet_model, series_index)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
