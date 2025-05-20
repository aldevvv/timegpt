import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import io
from datetime import datetime, timedelta
from nixtla import NixtlaClient
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title="Yahoo Finance Historical Data Stock Forecasting Dashboard - Version 1.0")

if 'df' not in st.session_state:
    st.session_state.df = None
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None
if 'residuals' not in st.session_state:
    st.session_state.residuals = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'forecast_horizon' not in st.session_state:
    st.session_state.forecast_horizon = 7
if 'has_forecast' not in st.session_state:
    st.session_state.has_forecast = False


def load_data(file):
    """Load Data From Uploaded File"""
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

    return df


def prepare_data_for_nixtla(df):
    """
    Prepare Data for Nixtla Forecasting by Ensuring Proper Rormat
    """

    prepared_df = df.copy()

    prepared_df = prepared_df.rename(columns={'Date': 'ds', 'Close': 'y'})

    prepared_df = prepared_df.sort_values('ds')

    final_df = prepared_df[['ds', 'y']]

    return final_df


def data_preview_tab():
    """Content for Tab 1: Data & Preview"""
    st.header("Data Upload and Preview")

    uploaded_file = st.file_uploader("Upload Historical Stock Data (Excel or CSV)",
                                   type=['xlsx', 'xls', 'csv'])

    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.session_state.df = df

            st.subheader("Data Preview")
            st.dataframe(df.head(10))

            st.subheader("Summary Statistics")
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            st.dataframe(df[numeric_cols].describe())

            st.subheader("Historical Close Price")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=df['Date'],
                          y=df['Close'],
                          mode='lines',
                          name='Close Price'))
            fig.update_layout(xaxis_title="Date",
                            yaxis_title="Price",
                            height=400,
                            margin=dict(l=40, r=40, t=40, b=40))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Trading Volume")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume'))
            fig.update_layout(xaxis_title="Date",
                            yaxis_title="Volume",
                            height=400,
                            margin=dict(l=40, r=40, t=40, b=40))
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error Loading Data : {str(e)}")
    else:
        st.info("Please Upload an Excel or CSV File Containing Historical Stock Data.")


def forecast_tab():
    """Content for Tab 2: Forecasting"""
    st.header("Forecasting")

    if st.session_state.df is None:
        st.warning("Please Upload Data In The 'Data & Preview' Tab First.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        horizon = st.selectbox("Select Forecast Horizon (Periods)",
                             options=[7, 14, 30],
                             index=0,
                             key='forecast_horizon_select')
    with col2:
        freq_choice = st.selectbox("Select Frequency",
                                 options=["Daily", "Weekly", "Monthly"],
                                 index=0,
                                 key='forecast_freq_select')
        if freq_choice == "Daily":
            resample_rule = "B"
            nixtla_freq = "B"
        elif freq_choice == "Weekly":
            resample_rule = "W-FRI"
            nixtla_freq = "W"
        else:
            resample_rule = "M"
            nixtla_freq = "M"
    with col3:
        forecast_btn = st.button("Run Forecast")

    if forecast_btn:
        with st.spinner("Generating Forecast..."):
            try:
                df = st.session_state.df.copy()
                nixtla_df = df.rename(columns={
                    "Date": "ds",
                    "Close": "y"
                })[["ds", "y"]]
                nixtla_df = nixtla_df.dropna(subset=["ds", "y"])

                nixtla_df = nixtla_df.drop_duplicates(subset="ds")
                nixtla_df["ds"] = pd.to_datetime(nixtla_df["ds"])

                nixtla_df = (nixtla_df.set_index("ds")
                            .resample(resample_rule)
                            .last()
                            .ffill()
                            .reset_index())

                nixtla_client = NixtlaClient(api_key="nixak-lNIIndERzx2khunJsElmx2CVFgZf6VAjtl5aMwSF64YfG9lktyS2aqS3YR9C6NJ2pDD6jO3Yrk8a3GF6")

                forecast_results = nixtla_client.forecast(
                    df=nixtla_df,
                    h=horizon,
                    freq=nixtla_freq,
                    level=[80, 95],
                    add_history=True,
                )

                st.session_state.forecast_results = forecast_results
                st.session_state.forecast_horizon = horizon
                st.session_state.has_forecast = True

                forecast_df = forecast_results.rename(columns={
                    "TimeGPT":       "y_hat",
                    "TimeGPT-lo-80": "y_hat_lo_80",
                    "TimeGPT-hi-80": "y_hat_hi_80",
                    "TimeGPT-lo-95": "y_hat_lo_95",
                    "TimeGPT-hi-95": "y_hat_hi_95",
                })
                forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

                in_sample = forecast_df[forecast_df["ds"] <= df["Date"].max()]

                merged_data = pd.merge(
                    df[["Date", "Close"]]
                      .rename(columns={"Date": "ds", "Close": "y"}),
                    in_sample[["ds", "y_hat"]]
                      .rename(columns={"y_hat": "Predicted"}),
                    on="ds"
                )
                merged_data["Residual"] = merged_data["y"] - merged_data["Predicted"]

                st.session_state.residuals = merged_data

                res = merged_data["Residual"]
                st.session_state.metrics = {
                    "MAPE": np.mean(np.abs(res / merged_data["y"])) * 100,
                    "RMSE": np.sqrt(np.mean(res**2)),
                    "MAE": np.mean(np.abs(res)),
                }

            except Exception as e:
                st.error(f"Error During Forecasting : {e}")
                st.session_state.has_forecast = False



    if st.session_state.has_forecast and st.session_state.forecast_results is not None:
        st.subheader("Forecast Results")

        try:
            df = st.session_state.df
            forecast_results = st.session_state.forecast_results

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(x=df['Date'],
                          y=df['Close'],
                          mode='lines',
                          name='Historical Close'))

            forecast_df = forecast_results.rename(
                columns={
                    "TimeGPT": "y_hat",
                    "TimeGPT-lo-80": "y_hat_lo_80",
                    "TimeGPT-hi-80": "y_hat_hi_80",
                    "TimeGPT-lo-95": "y_hat_lo_95",
                    "TimeGPT-hi-95": "y_hat_hi_95",
                })

            forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
            pred_col = "y_hat"
            lower_95_col = "y_hat_lo_95"
            upper_95_col = "y_hat_hi_95"

            st.line_chart(forecast_df.set_index("ds")[[pred_col]])

            future_forecast = forecast_df[forecast_df['ds'] > df['Date'].max()]

            fig.add_trace(
                go.Scatter(x=future_forecast['ds'],
                          y=future_forecast[pred_col],
                          mode='lines',
                          name='Forecast',
                          line=dict(color='red')))

            if lower_95_col in future_forecast.columns and upper_95_col in future_forecast.columns:
                fig.add_trace(
                    go.Scatter(x=future_forecast['ds'],
                              y=future_forecast[upper_95_col],
                              mode='lines',
                              line=dict(width=0),
                              showlegend=False))
                fig.add_trace(
                    go.Scatter(x=future_forecast['ds'],
                              y=future_forecast[lower_95_col],
                              mode='lines',
                              line=dict(width=0),
                              fill='tonexty',
                              fillcolor='rgba(255, 0, 0, 0.2)',
                              name='95% Confidence Interval'))

            fig.update_layout(title='Stock Price Forecast',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            height=500,
                            margin=dict(l=40, r=40, t=60, b=40),
                            legend=dict(orientation="h",
                                      yanchor="bottom",
                                      y=1.02,
                                      xanchor="right",
                                      x=1))
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Forecast Start",
                         future_forecast['ds'].min().strftime('%Y-%m-%d'))
            with col2:
                st.metric("Forecast End",
                         future_forecast['ds'].max().strftime('%Y-%m-%d'))
            with col3:
                st.metric("Forecast Horizon",
                         f"{st.session_state.forecast_horizon} days")

            if st.session_state.metrics:
                st.subheader("Forecast Performance Metrics")
                metrics = st.session_state.metrics

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                with col2:
                    st.metric("RMSE", f"{metrics['RMSE']:.2f}")
                with col3:
                    st.metric("MAE", f"{metrics['MAE']:.2f}")

        except Exception as e:
            st.error(f"Error Displaying Forecast Results : {str(e)}")
            st.write("Forecast DataFrame Columns :",
                    ", ".join(forecast_results.columns))
        p0 = future_forecast['y_hat'].iloc[0]
        pN = future_forecast['y_hat'].iloc[-1]
        change = ((pN-p0)/p0)*100
        trend = 'naik' if change>0 else 'turun'
        vol = st.session_state.residuals['Residual'].std()

        st.markdown("---")
        st.subheader("Deep Analysis")
        st.markdown(f"""
**1. Model Performance**
- **MAPE ({metrics['MAPE']:.2f}%)** : Menunjukkan Bahwa Rata-Rata Prediksi Menyimpang Sekitar {metrics['MAPE']:.2f}% Dari Nilai Aktual.
- **RMSE ({metrics['RMSE']:.2f})** : Mengukur Magnitudo Kesalahan Persebaran Kuadrat, Penting Untuk Menilai Pelanggaran Besar.
- **MAE ({metrics['MAE']:.2f})** : Rata-Rata Absolut Selisih Prediksi dan Aktual, Lebih Tahan Terhadap Outlier.

**2. Stock Price Projection**
- **Periode Prediksi** : {future_forecast['ds'].min().strftime('%Y-%m-%d')} s.d. {future_forecast['ds'].max().strftime('%Y-%m-%d')} ({len(future_forecast)} Hari).
- **Prediksi Awal** : {p0:.2f}, **Prediksi Akhir**: {pN:.2f} → Perubahan **{change:.2f}%** ({trend}).
- **95% Confidence Interval** :
  - Batas Bawah : ${future_forecast['y_hat_lo_95'].min():.2f}
  - Batas Atas : ${future_forecast['y_hat_hi_95'].max():.2f}
- Setiap Titik Prediksi Menampilkan Estimasi Interval, Contohnya Pada Tanggal {future_forecast['ds'].iloc[int(len(future_forecast)/2)].strftime('%Y-%m-%d')} Diproyeksikan ${future_forecast['y_hat'].iloc[int(len(future_forecast)/2)]:.2f} (Confidence Interval ${future_forecast['y_hat_lo_95'].iloc[int(len(future_forecast)/2)]:.2f} - ${future_forecast['y_hat_hi_95'].iloc[int(len(future_forecast)/2)]:.2f}]).

**3. Volatility & Trend**
- **Arah Tren** : Berdasarkan Kenaikan/Penurunan Sebesar {change:.2f}%, Tren Cenderung **{trend}**.
- **Volatilitas Residual** : Std Dev Error = {vol:.2f}, Mengindikasikan Fluktuasi Rata-Rata Selisih Prediksi.
- Residual Terdistribusi Mendekati Normal Dengan Rata-Rata ~0, Periksa Plot Distribusi Untuk Verifikasi.

**4. Tactical Strategy & Recommendation**
- **Buy Strategy** : Jika Tren Naik, Beli Saat Pullback Sekitar Support Historis, Posisikan Take-Profit di Level Resistance Berikut.
- **Sell Strategy** : Jika Tren Turun, Short Sell Atau Gunakan Trailing Stop-Loss di Atas Resistance Terdekat.
- **Manajemen Risiko** : Tentukan Stop-loss Maksimal 1-2 × Volatilitas Residual ({vol:.2f} × Faktor Risiko).
- **Monitoring** : Amati Rilis Data Ekonomi, Laporan Earning & Sentimen Pasar Untuk Potensi GAP.

**5. Limitation & Asumption**
- Model TimeGPT Mengasumsikan Konsistensi Pola Historis & Faktor Eksternal (Peristiwa Tak Terduga) Bisa Membuat Hasil Prediksi Kurang Akurat.
- Horizon Lebih Panjang Dari ({len(future_forecast)} Hari) → Tingkat Ketidakpastian Lebih Tinggi.
- Kombinasikan Dengan Analisis Fundamental, Sentimen & Indikator Teknikal (RSI, MACD) Untuk Validasi.

**6. Next Step**
- Uji Coba Parameter Frekuensi dan Horizon Berbeda Untuk Robustnes.
- Pantau Kinerja Model Tiap Periode dan Update Data Historis Secara Berkala.
""")




def residual_analysis_tab():
    """Content for Tab 3: Residual & Analysis"""
    st.header("Residual Analysis")

    if not st.session_state.has_forecast:
        st.warning("Please Run A Forecast In The 'Forecasting' Tab First.")
        return

    if st.session_state.residuals is not None:
        residuals = st.session_state.residuals

        st.subheader("Residuals Over Time")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=residuals['ds'],
                      y=residuals['Residual'],
                      mode='lines',
                      name='Residuals'))
        fig.add_shape(type="line",
                     x0=residuals['ds'].min(),
                     y0=0,
                     x1=residuals['ds'].max(),
                     y1=0,
                     line=dict(color="black", width=1, dash="dash"))
        fig.update_layout(xaxis_title='Date',
                         yaxis_title='Residual (Actual - Predicted)',
                         height=400,
                         margin=dict(l=40, r=40, t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Autocorrelation Analysis")

        if len(residuals) > 5:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            plot_acf(residuals['Residual'].values,
                    ax=ax1,
                    lags=min(20, len(residuals) - 1))
            ax1.set_title('Autocorrelation Function (ACF)')

            plot_pacf(residuals['Residual'].values,
                     ax=ax2,
                     lags=min(20, len(residuals) - 1))
            ax2.set_title('Partial Autocorrelation Function (PACF)')

            st.pyplot(fig)
        else:
            st.warning("Not Enough Data Points For ACF/PACF Analysis.")

        st.subheader("Residual Distribution")
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(x=residuals['Residual'],
                        nbinsx=30,
                        marker_color='lightblue',
                        opacity=0.7))
        fig.update_layout(xaxis_title='Residual Value',
                         yaxis_title='Frequency',
                         height=400,
                         margin=dict(l=40, r=40, t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("No Residual Data Available. - Please Run a Forecast First.")


def create_report_pdf():
    """Create a PDF Report With Forecast Results & Analysis"""
    buffer = io.BytesIO()

    with PdfPages(buffer) as pdf:
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5,
                 0.9,
                 'Stock Price Forecast Report',
                 ha='center',
                 fontsize=24,
                 fontweight='bold')
        plt.text(
            0.5,
            0.85,
            f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            ha='center',
            fontsize=12)
        plt.text(0.5,
                 0.8,
                 f"Forecast Horizon : {st.session_state.forecast_horizon} Days",
                 ha='center',
                 fontsize=14)

        metrics = st.session_state.metrics
        plt.text(0.5, 0.7, "Forecast Performance Metrics :",
                 ha='center', fontsize=14, fontweight='bold')

        plt.text(0.5, 0.66, f"MAPE : {metrics['MAPE']:.2f}%",
                 ha='center', fontsize=12)
        plt.text(0.5, 0.62, f"RMSE : {metrics['RMSE']:.2f}",
                 ha='center', fontsize=12)
        plt.text(0.5, 0.58, f"MAE : {metrics['MAE']:.2f}",
                 ha='center', fontsize=12)

        pdf.savefig()
        plt.close()

        if st.session_state.df is not None and st.session_state.forecast_results is not None:
            plt.figure(figsize=(10, 6))

            plt.plot(st.session_state.df['Date'],
                    st.session_state.df['Close'],
                    label='Historical Close Price',
                    color='blue')

            df = st.session_state.df
            forecast = st.session_state.forecast_results

            forecast = st.session_state.forecast_results.copy()
            forecast = forecast.rename(columns={
                "TimeGPT":       "y_hat",
                "TimeGPT-lo-80": "y_hat_lo_80",
                "TimeGPT-hi-80": "y_hat_hi_80",
                "TimeGPT-lo-95": "y_hat_lo_95",
                "TimeGPT-hi-95": "y_hat_hi_95",
            })

            pred_col = 'y_pred' if 'y_pred' in forecast.columns else 'y_hat'
            lower_95_col = 'y_pred_lower_95' if 'y_pred_lower_95' in forecast.columns else 'y_hat_lower_95'
            upper_95_col = 'y_pred_upper_95' if 'y_pred_upper_95' in forecast.columns else 'y_hat_upper_95'

            forecast['ds'] = pd.to_datetime(forecast['ds'])
            future_forecast = forecast[forecast['ds'] > df['Date'].max()]

            plt.plot(future_forecast['ds'],
                    future_forecast[pred_col],
                    label='Forecasted Close Price',
                    color='red')

            if lower_95_col in future_forecast.columns and upper_95_col in future_forecast.columns:
                plt.fill_between(future_forecast['ds'],
                               future_forecast[lower_95_col],
                               future_forecast[upper_95_col],
                               color='red',
                               alpha=0.2,
                               label='95% Confidence Interval')

            plt.title('Stock Price Historical Data and Forecast')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

        if st.session_state.residuals is not None:
            plt.figure(figsize=(10, 6))

            residuals = st.session_state.residuals
            plt.plot(residuals['ds'], residuals['Residual'], color='green')
            plt.axhline(y=0, color='r', linestyle='--')

            plt.title('Forecast Residuals (Actual - Predicted)')
            plt.xlabel('Date')
            plt.ylabel('Residual Value')
            plt.grid(True, alpha=0.3)

            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

            if len(residuals) > 5:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

                plot_acf(residuals['Residual'].values,
                        ax=ax1,
                        lags=min(20, len(residuals) - 1))
                ax1.set_title('Autocorrelation Function (ACF)')

                plot_pacf(residuals['Residual'].values,
                         ax=ax2,
                         lags=min(20, len(residuals) - 1))
                ax2.set_title('Partial Autocorrelation Function (PACF)')

                plt.tight_layout()
                pdf.savefig()
                plt.close()

            plt.figure(figsize=(8, 6))
            plt.hist(residuals['Residual'],
                    bins=30,
                    alpha=0.7,
                    color='lightblue')
            plt.title('Distribution of Residuals')
            plt.xlabel('Residual Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    buffer.seek(0)
    return buffer


def download_tab():
    """Content For Tab 4: Download & Export"""
    st.header("Download & Export Results")

    if not st.session_state.has_forecast:
        st.warning("Please Run a Forecast In The 'Forecasting' Tab First.")
        return

    st.write(
        "Generate A Comprehensive PDF Report With Forecast Results & Analysis.")

    if st.button("Generate PDF Report"):
        with st.spinner("Creating PDF Report..."):
            try:
                pdf_buffer = create_report_pdf()

                date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"StockForecastReport_{date_str}.pdf"

                st.success("PDF Report Generated Successfully!")
                st.download_button(label="Download PDF Report",
                                 data=pdf_buffer,
                                 file_name=filename,
                                 mime="application/pdf")
            except Exception as e:
                st.error(f"Error Generating PDF : {str(e)}")


def main():
    """Main Function To Run The Streamlit app"""
    st.title("Stock Price Forecasting Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Data & Preview", "Forecasting", "Residual & Analysis",
         "Download & Export"])

    with tab1:
        data_preview_tab()

    with tab2:
        forecast_tab()

    with tab3:
        residual_analysis_tab()

    with tab4:
        download_tab()


if __name__ == "__main__":
    main()

    footer_html = """
    <style>
      .footer {
          position: fixed;
          left: 0;
          bottom: 0;
          width: 100%;
          background-color: #000000;
          color: #ffffff;
          text-align: center;
          font-size: 0.85rem;
          padding: 0.5rem 0;
          z-index: 1000;
      }
      .footer a {
          color: #1E90FF;
          text-decoration: underline;
      }
      .footer a:hover {
          color: #ffffff;
      }
    </style>
    <div class="footer">
        © 2025 Makassar State University - Developed with ❤️ by
        <a href="https://aldev.web.id" target="_blank">Team 8</a>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)
