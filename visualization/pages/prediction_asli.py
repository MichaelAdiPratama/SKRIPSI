# import dash
# from dash import html, dcc, Input, Output, State, register_page, dash_table, callback
# import dash_bootstrap_components as dbc
# import pandas as pd
# import plotly.graph_objects as go
# from joblib import load  # untuk model RF & XGBoost
# from sklearn.preprocessing import StandardScaler
# from lib import attribute as att
# import tensorflow as tf  # untuk model LSTM
# import plotly.express as px
# from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
# import numpy as np
# import joblib

# register_page(__name__, path='/prediction_asli')
# # Dictionary untuk display model name di dropdown:
# model_display_names = {
#     "Linear Regression (Machine Learning)": "Linear Regression", #1
#     "Random Forest (Machine Learning)": "Random Forest", #2
#     "XGBoost (Machine Learning)": "XGBoost", #3
#     "SVR (Machine Learning)": "SVR", #4
#     "KNN (Machine Learning)": "KNN", #5
#     "LightGBM (Machine Learning)": "LightGBM", #6
#     "CatBoost (Machine Learning)": "CatBoost", #7
#     "Extra Trees (Machine Learning)": "Extra Trees", #8
#     "LSTM (Deep Learning)": "LSTM", #9
#     "Transformer (Deep Learning)": "Transformer", #10
#     "GRU (Deep Learning)": "GRU", #11
#     "CNN-1D (Deep Learning)": "CNN-1D", #12
#     "MLP (Deep Learning)": "MLP" #13
# }
# # Load dataset parquet
# df = att.data_all

# # Layout utama sesuai template
# layout = dbc.Container([
#     html.Br(),
#     html.H4("Prediction and Visualization", className="text-center border border-dark p-2 fw-bold"),
#     html.Br(),
#     html.Div([
#             dbc.Row([
#                 html.Div(
#                     id='year-slider-warning',
#                     children="⚠️ To display the matrix evaluation model (RMSE, MAE, and R²), choose the year slider for 2021 to 2023 only!",
#                     style={'color': 'red', 'fontWeight': 'bold', 'display': 'none'}
#                 ),

#                 dbc.Col(dcc.RangeSlider(2021, 2030, 1, value=[2021, 2030],
#                                         marks={i: str(i) for i in range(2021, 2031)},
#                                         id='year-slider'), width=14)
#             ]),
#             html.Br(),
#             dbc.Row([
#                 dbc.Col(dcc.Dropdown(id='province-dropdown',
#                                     options=[{'label': i, 'value': i} for i in df['Province'].unique()],
#                                     placeholder='Choose Province',
#                                     style={"textAlign": "center"}
#                                     ), width=3),
                                    
#                 dbc.Col(dcc.Dropdown(id='frequency-dropdown',
#                                     options=[
#                                         {'label': 'Daily', 'value': 'daily'},
#                                         {'label': 'Weekly', 'value': 'weekly'},
#                                         {'label': 'Monthly', 'value': 'monthly'}
#                                     ],
#                                     placeholder='Choose Frequency',
#                                     style={"textAlign": "center"}
#                                     ), width=3),
#                 dbc.Col(dcc.Dropdown(id='model-dropdown',
#                                     options=[{'label': m, 'value': m} for m in model_display_names.keys()],
#                                     placeholder='Choose Model',
#                                     style={"textAlign": "center"}
#                                     ), width=5),
#                 dbc.Col(html.Button("Predict", id='predict-btn', className="btn btn-success", style={"textAlign": "center"}), width=1),
#             ], justify='center'),
#     ]),

#     html.Br(),
#     # Map by Selected Province:
#     dbc.Row([dbc.Col(dbc.Card(dcc.Graph(id="map-graph-predict", figure=att.get_indonesia_map(), style={"width": "100%", "height": "350px"}), body=True, style={"opacity": "0.5"}), width=12)]),
#     html.Br(),
#     # Line Chart Prediction Demand Energy:
#     dbc.Row([
#         dbc.Col(dcc.Graph(id='prediction-line-chart'), width=12)
#     ]),
#     html.Br(),
#     # Matrix Evaluation Table
#     # dbc.Table(id="eval-metrics", bordered=True, striped=True, hover=True),
#     dbc.Table(
#         children=[
#             dbc.Row(
#                 dbc.Col(
#                     html.H6("Matrix Evaluation Table", className="fw-bold"), width=12), className="text-center mt-4"
#             ),
#             dbc.Row(
#                 dbc.Col(
#                     dbc.Table(
#                         id="eval-metrics", bordered=True, striped=True, hover=True,
#                     ), width=12
#                 )
#             )
#         ]
#     ),
#     html.Br(),
#     # Card Demografi dan PDRB:
#     dbc.Row([
#         dbc.Col(
#         dbc.Card([
#             dbc.CardHeader("Total Population"),
#             dbc.CardBody([
#                 html.H3(id='card-pred-jumlah-penduduk', className='p-3'),
#             ]),
#             ], className='text-center',color='light'),
#             width=4,
#         ),
#         dbc.Col(
#             dbc.Card([
#                 dbc.CardHeader("Total Poor People"),
#                 dbc.CardBody([
#                     html.H3(id='card-pred-jumlah-penduduk-miskin', className='p-3'),
#                 ]),
#             ], className='text-center', color='light'),
#             width=4,
#         ),
#         dbc.Col(
#             dbc.Card([
#                 dbc.CardHeader("Gross Regional Domestic Product (PDRB)"),
#                 dbc.CardBody([
#                     html.H3(id='card-pred-pdrb', className='p-3'),
#                 ]),
#             ], className='text-center', color='light'),width=4,
#         ),
#     ]),
#     html.Br(),
# ])
# @callback(
#     [
#         Output('map-graph-predict', 'figure'),
#         Output('prediction-line-chart', 'figure'),
#         Output('eval-metrics', 'children'),
#         Output('card-pred-jumlah-penduduk', 'children'),
#         Output('card-pred-jumlah-penduduk-miskin', 'children'),
#         Output('card-pred-pdrb', 'children'),
#     ],
#     Input("predict-btn", "n_clicks"),
#     [
#         State("province-dropdown", "value"),
#         State("frequency-dropdown", "value"),
#         State("model-dropdown", "value"),
#         State("year-slider", "value")
#     ]
# )
# def update_prediction(n_clicks,selected_province, selected_freq, selected_model, year_range):
#     # Pastikan tombol prediksi sudah diklik dan semua input sudah dipilih
#     if not n_clicks:
#         return att.get_indonesia_map(), go.Figure(), "", "", "", ""
#     if not selected_province or not selected_model or not year_range:
#         return att.get_indonesia_map(), go.Figure(), "Please fill in and complete all input dropdowns and year sliders!", "", "", ""
    
#     df_filtered = df[(df["Province"] == selected_province)]
#     if df_filtered.empty:
#         return att.get_indonesia_map(), go.Figure(), "Data is not available for that filter, please try again!", "", "", ""

#     selected_display_name = selected_model  # misal dari Dash callback
#     actual_model_name = model_display_names.get(selected_display_name)
#     # buat list model untuk predict dan train evaluate model:
#     model = att.load_model_dynamic(selected_province, actual_model_name)
#     # Lakukan prediksi jika tombol klik valid
#     df_filtered = df_filtered.sort_values(by='Year')

#     # Batas data historis
#     train_df = df_filtered[df_filtered['Year'] <= 2023].copy()
#     test_df = df_filtered[df_filtered['Year'] >= year_range[0]]

#     feature_cols = [
#         'Temperature', 'Jumlah_Penduduk', 'Jumlah_Penduduk_Miskin', 'PDRB',
#         'Jumlah_Pelanggan_Listrik', 'Listrik_Terjual', 'Daya_Terpasang', 'Produksi_Listrik',
#         'Persentase_Penduduk_Miskin', 'Latitude', 'Longitude'
#     ]
#     scaler = StandardScaler()
#     X_train = train_df[feature_cols]
#     y_train = train_df['Demand']
#     scaler.fit(X_train)

#     # Tentukan target_years di luar blok if-else agar bisa diakses oleh seluruh bagian kode
#     target_years = list(range(year_range[0], year_range[1] + 1))

#     # Jika tahun prediksi > 2023 → Buat data masa depan
#     if year_range[0] > 2023:
#         location_coords = df_filtered[["Province", "Regency", "Latitude", "Longitude"]].drop_duplicates()
#         selected_year = year_range[-1]
#         # map_fig = att.get_province_point(selected_province, selected_year, top_n=20)
#         # Pastikan kolom Date dalam format datetime
#         if train_df["Date"].dtype == "O":  # objek/string
#             train_df["Date"] = pd.to_datetime(train_df["Date"], errors='coerce')

#         # Filter data historis dari tahun 2021 hingga 2023
#         train_df_filtered = train_df[(train_df["Date"].dt.year >= 2021) & (train_df["Date"].dt.year <= 2023)]

#         # Fitur dan target
#         X_train = train_df_filtered[feature_cols]
#         y_train = train_df_filtered["Demand"]

#         # Fit scaler dengan data historis
#         scaler.fit(X_train)
#         X_train_scaled = scaler.transform(X_train)

#         if actual_model_name in ["Linear Regression", "Random Forest", "XGBoost", "KNN","LightGBM","CatBoost","Extra Trees","SVR"]:
#             model.fit(X_train_scaled, y_train)
#         elif actual_model_name in ["LSTM", "CNN-1D", "GRU", "MLP"]:
#             model.fit(X_train_scaled, y_train)
#         else:
#             model.fit(X_train_scaled, y_train)
        
#         # --- Proyeksi masa depan (future_df) tetap seperti yang Anda tulis ---
#         # Hitung nilai rata-rata historis untuk digunakan dalam future_df
#         mean_hourly_values = train_df_filtered[feature_cols].mean()
#         y_pred = None 
#         # latest_hist = train_df[train_df['Year'] == 2023].iloc[-1]
#         base_year = 2023
#         n_years = year_range[1] - year_range[0] + 1

#         # Tentukan jumlah periode berdasarkan frekuensi
#         if selected_freq == "monthly":
#             n_periods = n_years * 12
#             freq = "MS"
#         elif selected_freq == "weekly":
#             n_periods = n_years * 52
#             freq = "W-SUN"
#         elif selected_freq == "daily":
#             n_periods = n_years * 365
#             freq = "D"
#         else:
#             raise ValueError("Invalid frequency selected")

#         # Buat date range berdasarkan frekuensi
#         date_range = pd.date_range(start=f"{year_range[0]}-01-01", periods=n_periods, freq=freq)
#         # Bangun future_df dengan baris sebanyak n_periods
#         future_df = pd.DataFrame({
#             "Date": date_range
#         })

#         # Assign kolom statis atau dengan pertumbuhan tahunan
#         future_df["Year"] = future_df["Date"].dt.year
#         future_df["Province"] = selected_province
#         np.random.seed(42)
#         temp_base = mean_hourly_values["Temperature"]
#         future_df["Temperature"] = temp_base + np.random.normal(loc=0, scale=1.5, size=n_periods)
#         # future_df["Temperature"] = latest_hist["Temperature"]
#         future_df["Jumlah_Penduduk"] = [mean_hourly_values["Jumlah_Penduduk"] * ((1 + 0.05) ** (y - base_year)) for y in future_df["Year"]]
#         future_df["Jumlah_Penduduk_Miskin"] = [mean_hourly_values["Jumlah_Penduduk_Miskin"] * ((1 - 0.05) ** (y - base_year)) for y in future_df["Year"]]
#         future_df["PDRB"] = [mean_hourly_values["PDRB"] * ((1 + 0.05) ** (y - base_year)) for y in future_df["Year"]]
#         future_df["Jumlah_Pelanggan_Listrik"] = [mean_hourly_values["Jumlah_Pelanggan_Listrik"] * ((1 + 0.05) ** (y - base_year)) for y in future_df["Year"]]
#         # Listrik Terjual
#         start_value = mean_hourly_values["Listrik_Terjual"] * ((1 + 0.05) ** (year_range[0] - base_year))
#         weekly_growth = 0.001  # contoh pertumbuhan mingguan
#         future_df["Listrik_Terjual"] = [start_value * ((1 + weekly_growth) ** i) for i in range(n_periods)]
#         # future_df["Listrik_Terjual"] = [latest_hist["Listrik_Terjual"] * ((1 + 0.05) ** (y - base_year)) for y in future_df["Year"]]

#         # Daya Terpasang
#         start_val = mean_hourly_values["Daya_Terpasang"] * ((1 + 0.05) ** (year_range[0] - base_year))
#         week_growth = 0.001
#         future_df["Daya_Terpasang"] = [start_val * ((1 + week_growth) ** i) for i in range(n_periods)]
#         # future_df["Daya_Terpasang"] = [latest_hist["Daya_Terpasang"] * ((1 + 0.05) ** (y - base_year)) for y in future_df["Year"]]

#         # Produksi Listrik
#         start_values = mean_hourly_values["Produksi_Listrik"] * ((1 + 0.05) ** (year_range[0] - base_year))
#         weekly_growths = 0.001
#         future_df["Produksi_Listrik"] = [start_values * ((1 + weekly_growths) ** i) for i in range(n_periods)]
#         # future_df["Produksi_Listrik"] = [latest_hist["Produksi_Listrik"] * ((1 + 0.05) ** (y - base_year)) for y in future_df["Year"]]
#         future_df["Persentase_Penduduk_Miskin"] = mean_hourly_values["Persentase_Penduduk_Miskin"]

#         future_df["Month"] = future_df["Date"].dt.month
#         future_df["Week"] = future_df["Date"].dt.isocalendar().week.astype(int)
#         future_df["DayOfYear"] = future_df["Date"].dt.dayofyear

#         # Menyamakan nilai-nilai setiap parameter sesuai selected frequency:
#         if selected_freq == "monthly":
#             seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * future_df["Month"] / 12)
#         elif selected_freq == "weekly":
#             seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * future_df["Week"] / 52)
#         elif selected_freq == "daily":
#             seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * future_df["DayOfYear"] / 365)

#         # Terapkan fluktuasi ke kolom-kolom energi (misalnya Produksi, Terjual, Daya Terpasang)
#         for col in ["Listrik_Terjual", "Daya_Terpasang", "Produksi_Listrik"]:
#             future_df[col] *= seasonal_factor

#         future_df = pd.concat([future_df.assign(Regency=reg) for reg in location_coords["Regency"].unique()], ignore_index=True)
#         future_df = future_df.merge(location_coords, on=["Province", "Regency"], how="left")

#         # Prediksi
#         X_future = scaler.transform(future_df[feature_cols])

#         if actual_model_name in ["Linear Regression", "Random Forest", "XGBoost", "KNN","LightGBM","CatBoost","Extra Trees","SVR"]:
#             # model.fit(X_train_scaled, y_train)
#             y_pred = model.predict(X_future)
#         elif actual_model_name in ["LSTM", "CNN-1D", "GRU", "MLP"]:
#             # model.fit(X_train_scaled, y_train)
#             X_future_reshaped = X_future.reshape((X_future.shape[0], X_future.shape[1], 1))
#             y_pred = model.predict(X_future_reshaped)
#         else:
#             # model.fit(X_train_scaled, y_train)
#             y_pred = model.predict(X_future).flatten() # NO expand_dims

#         future_df["Prediction"] = y_pred

#         # Agregasi rata-rata prediksi per lokasi (misal per Regency)
#         top_n_df = future_df.groupby(["Regency", "Latitude", "Longitude"], as_index=False)["Prediction"].sum()
#         print(f"Data size before top_n: {top_n_df.shape}")
#         print(f"Unique coordinate points: {top_n_df[['Latitude', 'Longitude']].drop_duplicates().shape}")
#         top_n_df = top_n_df.sort_values("Prediction", ascending=False).head(20)  # top 10 demand tertinggi
#         print(top_n_df.head(20))

#         # Output the prediction table (Evaluation)
#         eval_metrics = ""

#         # Output kartu indikator (ambil dari akhir tahun terakhir)
#         pred_population = future_df[future_df["Year"] == year_range[1]]['Jumlah_Penduduk'].iloc[-1]
#         pred_poverty = future_df[future_df["Year"] == year_range[1]]['Jumlah_Penduduk_Miskin'].iloc[-1]
#         pred_pdrb = future_df[future_df["Year"] == year_range[1]]['PDRB'].iloc[-1]

#         # Visualisasi
#         fig_map = att.predict_map_figure(top_n_df, selected_province, year_range, top_n=20) # untuk map predict lokasi koordinat di peta map

#         # Agregasi prediksi berdasarkan tanggal (rata-rata atau total)
#         agg_pred_df = future_df.groupby("Date", as_index=False)["Prediction"].mean()
#         fig_line = go.Figure()
#         fig_line.add_trace(go.Scatter(x=agg_pred_df["Date"], y=agg_pred_df["Prediction"], mode='lines', name="Predicted"))
#         fig_line.update_layout(
#             title='Modelled Electricity Demand (in KWh)', title_x=0.5,  # center judul
#             height=500, xaxis_title='Date',  # label sumbu X
#             yaxis_title='Electricity Demand (KWh)',  # label sumbu Y
#             yaxis=dict(range=[0, 800]),
#             legend=dict(
#                 x=1, y=1, xanchor='right', yanchor='top', bgcolor='rgba(255,255,255,0.5)'  # opsional: latar belakang legend transparan agar lebih rapi
#             ), 
#             margin=dict(l=40, r=40, t=80, b=40), # opsional: margin yang seimbang
#         )

#         return fig_map, fig_line, eval_metrics, f"{int(pred_population):,}", f"{int(pred_poverty):,}", f"{int(pred_pdrb):,}"

#     # Kalau tahun <= 2023 → Pakai data test (ground truth tersedia)
#     else:
#         selected_year = year_range[-1]
#         map_fig = att.get_province_point(selected_province, selected_year, top_n=20)
#         X_test = test_df[feature_cols]
#         y_test = test_df['Demand']
#         X_train_scaled = scaler.transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
#         # Prediksi berdasarkan model yang dipilih
#         if actual_model_name in ["Linear Regression", "Random Forest", "XGBoost", "KNN","LightGBM","CatBoost","Extra Trees","SVR"]:
#             model.fit(X_train_scaled, y_train)
#             y_pred = model.predict(X_test_scaled)
#         elif actual_model_name in ["LSTM", "CNN-1D", "GRU", "MLP"]:
#             model.fit(X_train_scaled, y_train)
#             X_test_seq = np.expand_dims(X_test_scaled, axis=-1)  # Ini penting! untuk model DL selain Transformer, karena TransformerBlock nya customize bukan bawaan library.
#             y_pred = model.predict(X_test_seq).flatten()
#         elif actual_model_name == "Transformer":
#             model.fit(X_train_scaled, y_train)
#             # Transformer custom_model:
#             y_pred = model.predict(X_test_scaled).flatten() # NO expand_dims
#         else:
#             # Handle case when selected model is invalid or missing
#             return map_fig, go.Figure(), "The selected model is invalid or unavailable. Please choose another model!", "", "", ""
#         # Evaluasi model
#         rmse = root_mean_squared_error(y_test, y_pred)
#         mae = mean_absolute_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)
#         eval_table = html.Tbody([
#             html.Tr([html.Td("Root Mean Squared Error (RMSE)"), html.Td("=",  className="text-center"), html.Td(f"{rmse:.2f}", className="text-center")]),
#             html.Tr([html.Td("Mean Absolute Error (MAE)"), html.Td("=",  className="text-center"),  html.Td(f"{mae:.2f}", className="text-center")]),
#             html.Tr([html.Td("R² Score"), html.Td("=",  className="text-center"), html.Td(f"{r2:.2f}", className="text-center")]),
#         ])
        
#         # Menghitung prediksi berdasarkan model
#         pred_df = test_df.copy()
#         pred_df['Predicted_Demand'] = y_pred

#         # Ambil 10 data sampel secara acak
#         random_sample = pred_df[['Date', 'Predicted_Demand']].sample(n=10, random_state=42)
#         print(random_sample)
#         # print(pred_df[['Date', 'Predicted_Demand']].head(10))  # Periksa beberapa nilai pertama

#         # --- RESAMPLING BERDASARKAN FREQUENCY YANG DIPILIH ---
#         pred_df.set_index('Date', inplace=True)
#         pred_df = pred_df[pred_df.index.year.isin(target_years)] 
#         # Pilih hanya kolom numerik untuk resampling
#         numeric_columns = pred_df.select_dtypes(include=['number']).columns
#         num_periods = min(len(pred_df), len(target_years) * 52)  # Sesuaikan jumlah periode dengan jumlah baris

#         if selected_freq == 'weekly':
#             pred_df_resampled =  pred_df[numeric_columns].resample('W').mean()  # Weekly resampling
#         elif selected_freq == 'monthly':
#             pred_df_resampled = pred_df[numeric_columns].resample('M').mean() # Monthly resampling (start of each month) 
#         elif selected_freq == 'daily':
#             pred_df_resampled = pred_df[numeric_columns].resample('D').mean() # Daily resampling (start of each year)
#         else:  # Daily (no resampling needed)
#             pred_df_resampled = pred_df.copy()

#         pred_df_resampled.reset_index(inplace=True)
#         # --- PLOTTING --- Grafik Prediksi vs Real Demand
#         demand_fig = go.Figure()
#         # Grafik actual electricity demand
#         demand_fig.add_trace(go.Scatter(x=pred_df_resampled['Date'], y=pred_df_resampled['Demand'], mode='lines+markers', name='Actual Electricity Demand'))
#         # Grafik predicted electricity demand
#         demand_fig.add_trace(go.Scatter(x=pred_df_resampled['Date'], y=pred_df_resampled['Predicted_Demand'], mode='lines+markers', name='Modelled Electricity Demand'))
#         # Layout pengaturan
#         demand_fig.update_layout(
#             title='Actual vs Modelled Electricity Demand (in KWh)', title_x=0.5,  # center judul
#             height=500, xaxis_title='Date',  # label sumbu X
#             yaxis_title='Electricity Demand (KWh)',  # label sumbu Y
#             yaxis=dict(range=[0, 800]),
#             legend=dict(
#                 x=1, y=0.2, xanchor='right', yanchor='top', bgcolor='white'  # opsional: latar belakang legend transparan agar lebih rapi
#             ), margin=dict(l=40, r=40, t=80, b=40)  # opsional: margin yang seimbang
#         )
#         # --- CARD PERHITUNGAN --- Menampilkan data statistik demografi
#         selected_year = max(year_range)
#         filtered_card_df = test_df[test_df['Date'].dt.year == selected_year]
#         avg_population = filtered_card_df['Jumlah_Penduduk'].max()
#         avg_poor_population = filtered_card_df['Jumlah_Penduduk_Miskin'].min()
#         avg_pdrb = filtered_card_df['PDRB'].max()

#         # --- UPDATE DATE COLUMN BERDASARKAN FREQUENCY --- (Untuk Tahun Setelah 2023)
#         update_pred_df = pred_df_resampled.copy()
#         # Tentukan tanggal awal dan akhir eksplisit dari slider
#         start_year = year_range[0]
#         end_year = year_range[1]
#         start_date = pd.Timestamp(f"{start_year}-01-01")
#         final_end_date = pd.Timestamp(f"{end_year}-12-31")

#         # Pastikan final_end_date tidak melampaui tanggal terakhir dalam data
#         if final_end_date > update_pred_df['Date'].max():
#             final_end_date = update_pred_df['Date'].max()

#         if selected_freq == "monthly":
#             date_range = pd.date_range(start=start_date, end=final_end_date, freq="M")
#         elif selected_freq == "weekly":
#             # Generate all weeks
#             all_weeks = pd.date_range(start=start_date, end=final_end_date + pd.Timedelta(days=7), freq="W")
#             # Sekarang filter supaya hanya tanggal <= 31 Dec end_year
#             all_weeks = all_weeks[all_weeks <= final_end_date]
#             date_range = all_weeks
#         elif selected_freq == "daily":
#             date_range = pd.date_range(start=start_date, end=final_end_date, freq="D")  # fallback harian
#         else:
#             date_range = pd.date_range(start=start_date, end=final_end_date, freq="YS") # fallback Tahunan
#         # Pastikan panjang sesuai dengan prediksi
#         if len(date_range) > len(update_pred_df):
#             date_range = date_range[:len(update_pred_df)]
#         elif len(date_range) < len(update_pred_df):
#             update_pred_df = update_pred_df.iloc[:len(date_range)]
#         # Assign tanggal
#         update_pred_df["Date"] = date_range

#         return ( map_fig, demand_fig, eval_table, f"{int(avg_population):,}", f"{int(avg_poor_population):,}", f"{int(avg_pdrb):,}")

# @callback(
#     Output('year-slider-warning', 'style'),
#     Input('province-dropdown', 'value'),
#     Input('frequency-dropdown', 'value'),
#     Input('model-dropdown', 'value'),
#     prevent_initial_call=True
# )

# def year_slider_warning(selected_province, selected_frequency, selected_model):
#     if selected_province is None or selected_frequency is None or selected_model is None:
#         return {'color': 'red', 'fontWeight': 'bold', 'display': 'block'}
#     return {'display': 'none'}