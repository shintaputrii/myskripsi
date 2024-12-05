import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from numpy import array
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


st.set_page_config(
    page_title="Prediksi Kualitas Udara DKI Jakarta",
    page_icon="https://raw.githubusercontent.com/shintaputrii/skripsi/main/house_1152964.png",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.extremelycoolapp.com/help",
        "Report a bug": "https://www.extremelycoolapp.com/bug",
        "About": "# This is a header. This is an *extremely* cool app!",
    },
)

st.write(
    """<h1 style="font-size: 40px;">Prediksi Kualitas Udara di DKI Jakarta</h1>""",
    unsafe_allow_html=True,
)

with st.container():
    with st.sidebar:
        selected = option_menu(
            st.write(
                """<h2 style = "text-align: center;"><img src="https://raw.githubusercontent.com/shintaputrii/skripsi/main/house_1152964.png" width="130" height="130"><br></h2>""",
                unsafe_allow_html=True,
            ),
            [
                "Home",
                "Data",
                "Missing Value & Normalisasi",
                "Hasil MAPE",
                "Next Day",

            ],
            icons=[
                "house",
                "file-earmark-font",
                "bar-chart",
                "gear",
                "arrow-down-square",
                "person",
            ],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#87CEEB"},
                "icon": {"color": "white", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "color": "white",
                },
                "nav-link-selected": {"background-color": "#005980"},
            },
        )

    if selected == "Home":
        st.write(
            """<h3 style = "text-align: center;">
        <img src="https://raw.githubusercontent.com/shintaputrii/skripsi/main/udara.jpeg" width="500" height="300">
        </h3>""",
            unsafe_allow_html=True,
        )

        st.subheader("""Deskripsi Aplikasi""")
        st.write(
            """
         Aplikasi Prediksi kualitas Udara di DKI Jakarta merupakan aplikasi yang digunakan untuk meramalkan 6 konsentrasi polutan udara di DKI Jakarta yang meliputi PM10, PM25, SO2, CO, NO2, dan O3 serta menentukan kategori untuk hari berikutnya..
        """
        )

    elif selected == "Data":

        st.subheader("""Deskripsi Data""")
        st.write(
            """
        Data yang digunakan dalam aplikasi ini yaitu data ISPU DKI Jakarta periode 1 Desember 2022 sampai 30 November 2023. Data yang ditampilkan adalah data ispu yang diperoleh per harinya. 
        """
        )

        st.subheader("""Sumber Dataset""")
        st.write(
            """
        Sumber data didapatkan dari website "Satu Data DKI Jakarta". Berikut merupakan link untuk mengakses sumber dataset.
        <a href="https://satudata.jakarta.go.id/search?q=data%20ispu&halaman=all&kategori=all&topik=all&organisasi=all&status=all&sort=desc&page_no=1&show_filter=true&lang=id">Klik disini</a>""",
            unsafe_allow_html=True,
        )

        st.subheader("""Dataset""")
        # Menggunakan file Excel dari GitHub
        df = pd.read_excel(
            "https://raw.githubusercontent.com/shintaputrii/skripsi/main/kualitasudara.xlsx"
        )
        st.dataframe(df, width=600)
        
        st.subheader("Penghapusan kolom")
        # Membaca dataset dari file Excel
        data = pd.read_excel(
            "https://raw.githubusercontent.com/shintaputrii/skripsi/main/kualitasudara.xlsx"
        )
        
        # Menghapus kolom yang tidak diinginkan
        data = data.drop(['periode_data', 'stasiun', 'parameter_pencemar_kritis', 'max', 'kategori'], axis=1)
        data.tanggal = pd.to_datetime(data.tanggal)
        
        # Menampilkan dataframe setelah penghapusan kolom
        st.dataframe(data, width=600)
        
    elif selected == "Missing Value & Normalisasi":
        # MEAN IMPUTATION
        st.subheader("""Mean Imputation""")
        # Membaca dataset dari file Excel
        data = pd.read_excel(
            "https://raw.githubusercontent.com/shintaputrii/skripsi/main/kualitasudara.xlsx"
        )
        
        # Menghapus kolom yang tidak diinginkan
        data = data.drop(['periode_data', 'stasiun', 'parameter_pencemar_kritis', 'max', 'kategori'], axis=1)
        
        # Mengganti nilai '-' dengan NaN
        data.replace(r'-+', np.nan, regex=True, inplace=True)
        
        # Menampilkan jumlah missing value per kolom
        missing_values = data.isnull().sum()
        st.write("Jumlah Missing Value per Kolom:")
        st.dataframe(missing_values[missing_values > 0].reset_index(name='missing_values'))
        
        # Mengidentifikasi kolom numerik
        numeric_cols = data.select_dtypes(include=np.number).columns
        
        # Imputasi mean untuk kolom numerik
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        # Konversi kolom yang disebutkan ke tipe data integer
        data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']] = data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']].astype(int)
        # Menampilkan data yang telah diproses
        st.dataframe(data, width=600)
        
        # Mengelompokkan data berdasarkan 'tanggal' dan menghitung rata-rata untuk kolom numerik
        data_grouped = data.groupby('tanggal')[numeric_cols].mean().reset_index()
    
        # Tampilkan hasil setelah pengelompokan dan perhitungan rata-rata
        st.write("Data Setelah Pengelompokan Berdasarkan Tanggal dan Perhitungan Rata-Rata:")
        st.dataframe(data_grouped, width=600)

        # ---- Mulai Menambahkan Kode untuk Supervised Learning ----
        from numpy import array
            
        # Fungsi untuk membagi urutan menjadi sampel
        def split_sequence(sequence, n_steps):
            X, y = list(), list()
            for i in range(len(sequence)):
                # Tentukan akhir pola
                end_ix = i + n_steps
                # Periksa apakah kita sudah melampaui urutan
                if end_ix > len(sequence)-1:
                    break
                # Ambil bagian input dan output dari pola
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
                X.append(seq_x)
                y.append(seq_y)
            return array(X), array(y)  # ubah menjadi masalah supervised learning
            
        # Tentukan panjang langkah (n_steps)
        kolom = 4
    
        # List kolom polutan yang ingin diproses
        polutan_cols = ['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']
    
        # Loop untuk setiap polutan dan buat data supervised learning
        for polutan in polutan_cols:
            # Ambil urutan data untuk polutan tersebut
            sequence = data_grouped[polutan].tolist()
            X, y = split_sequence(sequence, kolom)
            
            # Konversi data fitur (X) ke DataFrame
            dataX = pd.DataFrame(X, columns=[f'Step_{i+1}' for i in range(kolom)])
            
            # Konversi target (y) ke DataFrame
            datay = pd.DataFrame(y, columns=["Xt"])
            
            # Gabungkan DataFrame fitur dan target
            data_supervised = pd.concat((dataX, datay), axis=1)
            
            # Tampilkan data yang telah dikonversi ke masalah supervised learning untuk setiap polutan
            st.write(f"Data Supervised Learning untuk {polutan}:")
            st.dataframe(data_supervised)

        # PLOTING DATA
        # ---- Plotting Data untuk Setiap Polutan ----
        import matplotlib.pyplot as plt
        
        # Menentukan ukuran figure untuk subplot
        plt.figure(figsize=(12, 18))  # Ukuran figure untuk 6 subplot
        
        # Loop untuk setiap polutan dan plot datanya
        for idx, polutan in enumerate(polutan_cols):
            plt.subplot(6, 1, idx+1)  # 6 baris, 1 kolom, subplot ke-idx+1
            plt.plot(data_grouped['tanggal'], data_grouped[polutan], label=polutan, color='tab:red' if polutan == 'pm_sepuluh' else 'tab:blue')
            plt.title(f'Konsentrasi {polutan}')
            plt.xlabel('Tanggal')
            plt.ylabel('Konsentrasi (µg/m³)')
            plt.grid(True)
            plt.legend()
    
        # Tampilkan plot 
        plt.tight_layout()
        st.pyplot(plt)
        
        # Normalisasi Data
        st.subheader("Normalisasi Data")
        all_supervised_data_normalized = pd.DataFrame()
        
        # Loop untuk setiap polutan dan buat data supervised learning
        for polutan in polutan_cols:
            # Ambil urutan data untuk polutan tersebut
            sequence = data_grouped[polutan].tolist()
            X, y = split_sequence(sequence, kolom)
            
            # Konversi data fitur (X) ke DataFrame
            dataX = pd.DataFrame(X, columns=[f'Step_{i+1}' for i in range(kolom)])
            
            # Konversi target (y) ke DataFrame
            datay = pd.DataFrame(y, columns=["Xt"])
            
            # Gabungkan DataFrame fitur dan target
            data_supervised = pd.concat((dataX, datay), axis=1)
            
            # Normalisasi Data Supervised Learning untuk polutan ini, termasuk target
            scaler = MinMaxScaler()
            data_supervised.iloc[:, :-1] = scaler.fit_transform(data_supervised.iloc[:, :-1])  # Normalisasi fitur
            data_supervised['Xt'] = scaler.fit_transform(data_supervised[['Xt']])  # Normalisasi target
            
            # Tampilkan hasil setelah normalisasi untuk setiap polutan
            st.write(f"Data Supervised Learning Setelah Normalisasi Min-Max untuk {polutan}:")
            st.dataframe(data_supervised)

    elif selected == "Hasil MAPE":
        st.subheader("Hasil MAPE untuk Setiap Polutan")
        # Membaca dataset dari file Excel
        data = pd.read_excel(
            "https://raw.githubusercontent.com/shintaputrii/skripsi/main/kualitasudara.xlsx"
        )
        
        # Menghapus kolom yang tidak diinginkan
        data = data.drop(['periode_data', 'stasiun', 'parameter_pencemar_kritis', 'max', 'kategori'], axis=1)
        
        # Mengganti nilai '-' dengan NaN
        data.replace(r'-+', np.nan, regex=True, inplace=True)
        
        # Menampilkan jumlah missing value per kolom
        missing_values = data.isnull().sum()
        
        # Mengidentifikasi kolom numerik
        numeric_cols = data.select_dtypes(include=np.number).columns
        
        # Imputasi mean untuk kolom numerik
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        # Konversi kolom yang disebutkan ke tipe data integer
        data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']] = data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']].astype(int)
        # Menampilkan data yang telah diproses
        
        # Mengelompokkan data berdasarkan 'tanggal' dan menghitung rata-rata untuk kolom numerik
        data_grouped = data.groupby('tanggal')[numeric_cols].mean().reset_index()

        # ---- Mulai Menambahkan Kode untuk Supervised Learning ----
        from numpy import array
            
        # Fungsi untuk membagi urutan menjadi sampel
        def split_sequence(sequence, n_steps):
            X, y = list(), list()
            for i in range(len(sequence)):
                # Tentukan akhir pola
                end_ix = i + n_steps
                # Periksa apakah kita sudah melampaui urutan
                if end_ix > len(sequence)-1:
                    break
                # Ambil bagian input dan output dari pola
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
                X.append(seq_x)
                y.append(seq_y)
            return array(X), array(y)  # ubah menjadi masalah supervised learning
            
        # Tentukan panjang langkah (n_steps)
        kolom = 4
    
        # List kolom polutan yang ingin diproses
        polutan_cols = ['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']
    
        # Loop untuk setiap polutan dan buat data supervised learning
        for polutan in polutan_cols:
            # Ambil urutan data untuk polutan tersebut
            sequence = data_grouped[polutan].tolist()
            X, y = split_sequence(sequence, kolom)
            
            # Konversi data fitur (X) ke DataFrame
            dataX = pd.DataFrame(X, columns=[f'Step_{i+1}' for i in range(kolom)])
            
            # Konversi target (y) ke DataFrame
            datay = pd.DataFrame(y, columns=["Xt"])
            
            # Gabungkan DataFrame fitur dan target
            data_supervised = pd.concat((dataX, datay), axis=1)
        
        # Loop untuk setiap polutan dan buat data supervised learning
        for polutan in polutan_cols:
            # Ambil urutan data untuk polutan tersebut
            sequence = data_grouped[polutan].tolist()
            X, y = split_sequence(sequence, kolom)
            
            # Konversi data fitur (X) ke DataFrame
            dataX = pd.DataFrame(X, columns=[f'Step_{i+1}' for i in range(kolom)])
            
            # Konversi target (y) ke DataFrame
            datay = pd.DataFrame(y, columns=["Xt"])
            
            # Gabungkan DataFrame fitur dan target
            data_supervised = pd.concat((dataX, datay), axis=1)
            
            # Normalisasi Data Supervised Learning untuk polutan ini, termasuk target
            scaler = MinMaxScaler()
            data_supervised.iloc[:, :-1] = scaler.fit_transform(data_supervised.iloc[:, :-1])  # Normalisasi fitur
            data_supervised['Xt'] = scaler.fit_transform(data_supervised[['Xt']])  # Normalisasi target
        import pandas as pd
        import numpy as np
        import streamlit as st
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.neighbors import KNeighborsRegressor
        
        # Fungsi untuk menghitung keanggotaan fuzzy (inverse distance)
        def calculate_membership_inverse(distances, epsilon=1e-10):
            memberships = 1 / (distances + epsilon)
            return memberships
        
        # Fungsi Fuzzy KNN
        def fuzzy_knn_predict(X_train, y_train, X_test, k=3):
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
            distances, indices = knn.kneighbors(X_test, n_neighbors=k, return_distance=True)
        
            y_pred = np.zeros(len(X_test))
        
            for i in range(len(X_test)):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train.iloc[neighbor_indices].values
        
                memberships = calculate_membership_inverse(neighbor_distances)
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            return y_pred
        
        # Membaca dataset dari file Excel
        data = pd.read_excel(
            "https://raw.githubusercontent.com/shintaputrii/skripsi/main/kualitasudara.xlsx"
        )
        
        # Menghapus kolom yang tidak diinginkan
        data = data.drop(['periode_data', 'stasiun', 'parameter_pencemar_kritis', 'max', 'kategori'], axis=1)
        
        # Mengganti nilai '-' dengan NaN
        data.replace(r'-+', np.nan, regex=True, inplace=True)
        
        # Imputasi mean untuk kolom numerik
        numeric_cols = data.select_dtypes(include=np.number).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        # Konversi kolom ke tipe data integer
        data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']] = data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']].astype(int)
        
        # Mengelompokkan data berdasarkan 'tanggal' dan menghitung rata-rata
        data_grouped = data.groupby('tanggal')[numeric_cols].mean().reset_index()
        
        # Fungsi untuk membagi urutan menjadi sampel
        def split_sequence(sequence, n_steps):
            X, y = list(), list()
            for i in range(len(sequence)):
                end_ix = i + n_steps
                if end_ix > len(sequence)-1:
                    break
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
                X.append(seq_x)
                y.append(seq_y)
            return np.array(X), np.array(y)
        
        # Parameter untuk split sequence
        kolom = 4
        polutan_cols = ['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']
        
        # Loop untuk setiap polutan
        for polutan in polutan_cols:
            sequence = data_grouped[polutan].tolist()
            X, y = split_sequence(sequence, kolom)
            
            dataX = pd.DataFrame(X, columns=[f'Step_{i+1}' for i in range(kolom)])
            datay = pd.DataFrame(y, columns=["Xt"])
            data_supervised = pd.concat((dataX, datay), axis=1)
        
            # Normalisasi Data
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
        
            data_supervised.iloc[:, :-1] = scaler_X.fit_transform(data_supervised.iloc[:, :-1])  # Normalisasi fitur
            data_supervised['Xt'] = scaler_y.fit_transform(data_supervised[['Xt']])  # Normalisasi target
        
            # Split data menjadi train dan test dengan rasio yang berbeda
            for train_size in [0.7, 0.8, 0.9]:
                X_train, X_test, y_train, y_test = train_test_split(
                    data_supervised.iloc[:, :-1],
                    data_supervised['Xt'],
                    train_size=train_size,
                    random_state=42
                )
        
                # Prediksi pada data uji
                y_test_pred_scaled = fuzzy_knn_predict(X_train, y_train, X_test, k=3)
        
                # Denormalisasi hasil prediksi dan target aktual
                y_test_actual = scaler_y.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
                y_test_pred_actual = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
        
                # Menghitung MAPE pada data uji dalam skala asli
                mape_test = np.mean(np.abs((y_test_actual - y_test_pred_actual) / y_test_actual)) * 100
        
                # Tampilkan hasil di Streamlit
                train_ratio = int(train_size * 100)
                test_ratio = 100 - train_ratio
                st.write(f"Rasio {train_ratio}:{test_ratio} - MAPE untuk {polutan}: {mape_test:.2f}%")
                
                # Menampilkan hasil prediksi dan nilai aktual pada data uji di Streamlit
                test_results = pd.DataFrame({
                    "Tanggal": data_grouped['tanggal'].iloc[-len(y_test):].values,
                    "Actual": y_test_actual,
                    "Predicted": y_test_pred_actual
                })
                st.write("Hasil Prediksi:")
                st.dataframe(test_results)

    elif selected == "Next Day":   
        st.subheader("Masukkan Konsentrasi SO2 untuk 3 Hari Terakhir")
        input_data = st.text_input("Masukkan nilai SO2 untuk 3 hari terakhir, dipisahkan dengan koma (contoh: 90,80,80):")
    
        if input_data:
            # Mengubah input string menjadi list float
            input_values = list(map(float, input_data.split(',')))
    
            if len(input_values) != kolom:
                st.error(f"Harap masukkan {kolom} nilai untuk 3 hari terakhir.")
            else:
                # Konversi input data ke DataFrame
                input_df = pd.DataFrame([input_values], columns=[f'Step_{i+1}' for i in range(kolom)])
    
                # Normalisasi data input
                input_df_normalized = scaler.transform(input_df)
    
                # Prediksi menggunakan Fuzzy KNN
                predictions_scaled = fuzzy_knn_predict(X_train, y_train, input_df_normalized, k=3)
    
                # Denormalisasi hasil prediksi untuk menampilkan nilai asli
                predictions_actual = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    
                # Menyusun hasil prediksi dengan tanggal
                last_date = data_grouped['tanggal'].iloc[-1]  # Tanggal terakhir di data training
                predicted_dates = pd.date_range(start=last_date, periods=4, freq='D')[1:]  # Tanggal berikutnya
    
                predicted_results = pd.DataFrame({
                    "tanggal": predicted_dates,
                    "predicted_so2": predictions_actual
                })
    
                # Menampilkan hasil prediksi di Streamlit
                st.write("Hasil Prediksi SO2 untuk 4 Hari ke Depan:")
                st.dataframe(predicted_results)


    # Menampilkan penanda
    st.markdown("---")  # Menambahkan garis pemisah
    st.write("Shinta Alya Imani Putri-200411100005 (Teknik Informatika)")
