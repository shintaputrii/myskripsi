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
                st.write(f"Rasio {int(train_size*100)}:{int((1-train_size)*100)} - MAPE untuk {polutan}: {mape_test:.2f}%")
                
                # Menampilkan hasil prediksi dan nilai aktual pada data uji di Streamlit
                test_results = pd.DataFrame({
                    "Tanggal": data_grouped['tanggal'].iloc[-len(y_test):].values,
                    "Actual": y_test_actual,
                    "Predicted": y_test_pred_actual
                })
                st.write("Hasil Prediksi:")
                st.dataframe(test_results)

    elif selected == "Next Day":   
        st.subheader("PM10")       
        # Fungsi untuk normalisasi data
        def normalize_data(data):
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
            return normalized_data, scaler
        
        # Fungsi untuk menghitung keanggotaan fuzzy (inverse distance)
        def calculate_membership_inverse(distances, epsilon=1e-10):
            memberships = 1 / (distances + epsilon)
            return memberships
        
        # Fungsi Fuzzy KNN untuk prediksi
        def fuzzy_knn_predict(data, pollutant, user_input, k=3):
            # Normalisasi data
            imports = data[pollutant].values.reshape(-1, 1)
            data[f'{pollutant}_normalized'], scaler = normalize_data(imports)
        
            # Ekstrak fitur dan target
            X = data[f'{pollutant}_normalized'].values[:-1].reshape(-1, 1)
            y = data[f'{pollutant}_normalized'].values[1:]
        
            # Bagi data menjadi train dan test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
        
            # Normalisasi input dari pengguna
            user_input_scaled = scaler.transform(np.array([[user_input]]))
        
            # Inisialisasi model KNN
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
        
            # Mendapatkan tetangga dan jaraknya
            distances, indices = knn.kneighbors(user_input_scaled, n_neighbors=k, return_distance=True)
        
            # Inisialisasi array untuk menyimpan prediksi
            y_pred = np.zeros(1)
        
            # Loop untuk menghitung prediksi berdasarkan membership
            for i in range(1):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train[neighbor_indices]
        
                # Hitung membership
                memberships = calculate_membership_inverse(neighbor_distances)
        
                # Hitung prediksi sebagai weighted average
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            # Mengembalikan nilai prediksi ke skala awal
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
        
            return y_pred_original[0][0]
        
        # Muat dan bersihkan data dari file
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
        
        # Input dari pengguna
        user_input = st.number_input("Masukkan konsentrasi PM 10:", min_value=0.0)
        
        # Prediksi berdasarkan input pengguna
        if st.button("Prediksi10"):
            prediction = fuzzy_knn_predict(data, "pm_sepuluh", user_input, k=3)
            st.write(f"Prediksi konsentrasi PM 10 esok hari: {prediction:.2f}")

        st.subheader("PM25")  
        # Fungsi untuk normalisasi data
        def normalize_data(data):
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
            return normalized_data, scaler
        
        # Fungsi untuk menghitung keanggotaan fuzzy (inverse distance)
        def calculate_membership_inverse(distances, epsilon=1e-10):
            memberships = 1 / (distances + epsilon)
            return memberships
        
        # Fungsi Fuzzy KNN untuk prediksi
        def fuzzy_knn_predict(data, pollutant, user_input, k=3):
            # Normalisasi data
            imports = data[pollutant].values.reshape(-1, 1)
            data[f'{pollutant}_normalized'], scaler = normalize_data(imports)
        
            # Ekstrak fitur dan target
            X = data[f'{pollutant}_normalized'].values[:-1].reshape(-1, 1)
            y = data[f'{pollutant}_normalized'].values[1:]
        
            # Bagi data menjadi train dan test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
        
            # Normalisasi input dari pengguna
            user_input_scaled = scaler.transform(np.array([[user_input]]))
        
            # Inisialisasi model KNN
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
        
            # Mendapatkan tetangga dan jaraknya
            distances, indices = knn.kneighbors(user_input_scaled, n_neighbors=k, return_distance=True)
        
            # Inisialisasi array untuk menyimpan prediksi
            y_pred = np.zeros(1)
        
            # Loop untuk menghitung prediksi berdasarkan membership
            for i in range(1):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train[neighbor_indices]
        
                # Hitung membership
                memberships = calculate_membership_inverse(neighbor_distances)
        
                # Hitung prediksi sebagai weighted average
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            # Mengembalikan nilai prediksi ke skala awal
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
        
            return y_pred_original[0][0]
        
        # Muat dan bersihkan data dari file
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
        
        # Konversi kolom yang disebutkan ke tipe data integer
        data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']] = data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']].astype(int)
        
        # Input dari pengguna untuk PM2.5
        user_input = st.number_input("Masukkan konsentrasi PM 2.5:", min_value=0.0)
        
        # Prediksi berdasarkan input pengguna
        if st.button("Prediksi 25"):
            prediction = fuzzy_knn_predict(data, "pm_duakomalima", user_input, k=3)
            st.write(f"Prediksi konsentrasi PM 2.5 esok hari: {prediction:.2f}")
        
        st.subheader("SO2")  
        # Fungsi untuk normalisasi data
        def normalize_data(data):
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
            return normalized_data, scaler
        
        # Fungsi untuk menghitung keanggotaan fuzzy (inverse distance)
        def calculate_membership_inverse(distances, epsilon=1e-10):
            memberships = 1 / (distances + epsilon)
            return memberships
        
        # Fungsi Fuzzy KNN untuk prediksi
        def fuzzy_knn_predict(data, pollutant, user_input, k=3):
            # Normalisasi data
            imports = data[pollutant].values.reshape(-1, 1)
            data[f'{pollutant}_normalized'], scaler = normalize_data(imports)
        
            # Ekstrak fitur dan target
            X = data[f'{pollutant}_normalized'].values[:-1].reshape(-1, 1)
            y = data[f'{pollutant}_normalized'].values[1:]
        
            # Bagi data menjadi train dan test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
        
            # Normalisasi input dari pengguna
            user_input_scaled = scaler.transform(np.array([[user_input]]))
        
            # Inisialisasi model KNN
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
        
            # Mendapatkan tetangga dan jaraknya
            distances, indices = knn.kneighbors(user_input_scaled, n_neighbors=k, return_distance=True)
        
            # Inisialisasi array untuk menyimpan prediksi
            y_pred = np.zeros(1)
        
            # Loop untuk menghitung prediksi berdasarkan membership
            for i in range(1):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train[neighbor_indices]
        
                # Hitung membership
                memberships = calculate_membership_inverse(neighbor_distances)
        
                # Hitung prediksi sebagai weighted average
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            # Mengembalikan nilai prediksi ke skala awal
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
        
            return y_pred_original[0][0]
        
        # Muat dan bersihkan data dari file
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
        
        # Konversi kolom yang disebutkan ke tipe data integer
        data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']] = data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']].astype(int)
        
        # Input dari pengguna untuk Sulfur Dioksida
        user_input = st.number_input("Masukkan konsentrasi Sulfur Dioksida:", min_value=0.0)
        
        # Prediksi berdasarkan input pengguna
        if st.button("Prediksi SO2"):
            prediction = fuzzy_knn_predict(data, "sulfur_dioksida", user_input, k=3)
            st.write(f"Prediksi konsentrasi Sulfur Dioksida esok hari: {prediction:.2f}")

        st.subheader("CO")  
        # Fungsi untuk normalisasi data
        def normalize_data(data):
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
            return normalized_data, scaler
        
        # Fungsi untuk menghitung keanggotaan fuzzy (inverse distance)
        def calculate_membership_inverse(distances, epsilon=1e-10):
            memberships = 1 / (distances + epsilon)
            return memberships
        
        # Fungsi Fuzzy KNN untuk prediksi
        def fuzzy_knn_predict(data, pollutant, user_input, k=3):
            # Normalisasi data
            imports = data[pollutant].values.reshape(-1, 1)
            data[f'{pollutant}_normalized'], scaler = normalize_data(imports)
        
            # Ekstrak fitur dan target
            X = data[f'{pollutant}_normalized'].values[:-1].reshape(-1, 1)
            y = data[f'{pollutant}_normalized'].values[1:]
        
            # Bagi data menjadi train dan test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
        
            # Normalisasi input dari pengguna
            user_input_scaled = scaler.transform(np.array([[user_input]]))
        
            # Inisialisasi model KNN
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
        
            # Mendapatkan tetangga dan jaraknya
            distances, indices = knn.kneighbors(user_input_scaled, n_neighbors=k, return_distance=True)
        
            # Inisialisasi array untuk menyimpan prediksi
            y_pred = np.zeros(1)
        
            # Loop untuk menghitung prediksi berdasarkan membership
            for i in range(1):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train[neighbor_indices]
        
                # Hitung membership
                memberships = calculate_membership_inverse(neighbor_distances)
        
                # Hitung prediksi sebagai weighted average
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            # Mengembalikan nilai prediksi ke skala awal
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
        
            return y_pred_original[0][0]
        
        # Muat dan bersihkan data dari file
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
        
        # Konversi kolom yang disebutkan ke tipe data integer
        data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']] = data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']].astype(int)
        
        # Input dari pengguna untuk Karbon Monoksida
        user_input = st.number_input("Masukkan konsentrasi Karbon Monoksida:", min_value=0.0)
        
        # Prediksi berdasarkan input pengguna
        if st.button("Prediksi CO"):
            prediction = fuzzy_knn_predict(data, "karbon_monoksida", user_input, k=3)
            st.write(f"Prediksi konsentrasi Karbon Monoksida esok hari: {prediction:.2f}")
        
        st.subheader("O3")  
        # Fungsi untuk normalisasi data
        def normalize_data(data):
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
            return normalized_data, scaler
        
        # Fungsi untuk menghitung keanggotaan fuzzy (inverse distance)
        def calculate_membership_inverse(distances, epsilon=1e-10):
            memberships = 1 / (distances + epsilon)
            return memberships
        
        # Fungsi Fuzzy KNN untuk prediksi
        def fuzzy_knn_predict(data, pollutant, user_input, k=3):
            # Normalisasi data
            imports = data[pollutant].values.reshape(-1, 1)
            data[f'{pollutant}_normalized'], scaler = normalize_data(imports)
        
            # Ekstrak fitur dan target
            X = data[f'{pollutant}_normalized'].values[:-1].reshape(-1, 1)
            y = data[f'{pollutant}_normalized'].values[1:]
        
            # Bagi data menjadi train dan test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
        
            # Normalisasi input dari pengguna
            user_input_scaled = scaler.transform(np.array([[user_input]]))
        
            # Inisialisasi model KNN
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
        
            # Mendapatkan tetangga dan jaraknya
            distances, indices = knn.kneighbors(user_input_scaled, n_neighbors=k, return_distance=True)
        
            # Inisialisasi array untuk menyimpan prediksi
            y_pred = np.zeros(1)
        
            # Loop untuk menghitung prediksi berdasarkan membership
            for i in range(1):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train[neighbor_indices]
        
                # Hitung membership
                memberships = calculate_membership_inverse(neighbor_distances)
        
                # Hitung prediksi sebagai weighted average
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            # Mengembalikan nilai prediksi ke skala awal
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
        
            return y_pred_original[0][0]
        
        # Muat dan bersihkan data dari file
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
        
        # Konversi kolom yang disebutkan ke tipe data integer
        data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']] = data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']].astype(int)
        
        # Input dari pengguna untuk Ozon
        user_input = st.number_input("Masukkan konsentrasi Ozon:", min_value=0.0)
        
        # Prediksi berdasarkan input pengguna
        if st.button("Prediksi O3"):
            prediction = fuzzy_knn_predict(data, "ozon", user_input, k=3)
            st.write(f"Prediksi konsentrasi Ozon esok hari: {prediction:.2f}")

        st.subheader("NO2")
        # Fungsi untuk normalisasi data
        def normalize_data(data):
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
            return normalized_data, scaler
        
        # Fungsi untuk menghitung keanggotaan fuzzy (inverse distance)
        def calculate_membership_inverse(distances, epsilon=1e-10):
            memberships = 1 / (distances + epsilon)
            return memberships
        
        # Fungsi Fuzzy KNN untuk prediksi
        def fuzzy_knn_predict(data, pollutant, user_input, k=3):
            # Normalisasi data
            imports = data[pollutant].values.reshape(-1, 1)
            data[f'{pollutant}_normalized'], scaler = normalize_data(imports)
        
            # Ekstrak fitur dan target
            X = data[f'{pollutant}_normalized'].values[:-1].reshape(-1, 1)
            y = data[f'{pollutant}_normalized'].values[1:]
        
            # Bagi data menjadi train dan test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
        
            # Normalisasi input dari pengguna
            user_input_scaled = scaler.transform(np.array([[user_input]]))
        
            # Inisialisasi model KNN
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
        
            # Mendapatkan tetangga dan jaraknya
            distances, indices = knn.kneighbors(user_input_scaled, n_neighbors=k, return_distance=True)
        
            # Inisialisasi array untuk menyimpan prediksi
            y_pred = np.zeros(1)
        
            # Loop untuk menghitung prediksi berdasarkan membership
            for i in range(1):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train[neighbor_indices]
        
                # Hitung membership
                memberships = calculate_membership_inverse(neighbor_distances)
        
                # Hitung prediksi sebagai weighted average
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            # Mengembalikan nilai prediksi ke skala awal
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
        
            return y_pred_original[0][0]
        
        # Muat dan bersihkan data dari file
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
        
        # Konversi kolom yang disebutkan ke tipe data integer
        data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']] = data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']].astype(int)
        
        # Input dari pengguna untuk Nitrogen Dioksida
        user_input = st.number_input("Masukkan konsentrasi Nitrogen Dioksida:", min_value=0.0)
        
        # Prediksi berdasarkan input pengguna
        if st.button("Prediksi NO2"):
            prediction = fuzzy_knn_predict(data, "nitrogen_dioksida", user_input, k=3)
            st.write(f"Prediksi konsentrasi Nitrogen Dioksida esok hari: {prediction:.2f}")

        # After all predictions, gather maximum values
        max_values = {
            "Polutan": ["PM10", "PM2.5", "SO2", "CO", "O3", "NO2"],
            "Nilai Maksimum": [
                data['pm_sepuluh'].max(),
                data['pm_duakomalima'].max(),
                data['sulfur_dioksida'].max(),
                data['karbon_monoksida'].max(),
                data['ozon'].max(),
                data['nitrogen_dioksida'].max()
            ]
        }
        
        # Convert to DataFrame
        max_values_data = pd.DataFrame(max_values)
        
        # Display the table
        st.subheader("Nilai Maksimum Per Polutan")
        st.table(max_values_data)

        st.subheader("Prediksi Kualitas Udara")
        # Fungsi untuk normalisasi data
        def normalize_data(data):
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
            return normalized_data, scaler
        
        # Fungsi untuk menghitung keanggotaan fuzzy (inverse distance)
        def calculate_membership_inverse(distances, epsilon=1e-10):
            memberships = 1 / (distances + epsilon)
            return memberships
        
        # Fungsi Fuzzy KNN untuk prediksi
        def fuzzy_knn_predict(data, pollutant, user_input, k=3):
            imports = data[pollutant].values.reshape(-1, 1)
            data[f'{pollutant}_normalized'], scaler = normalize_data(imports)
        
            X = data[f'{pollutant}_normalized'].values[:-1].reshape(-1, 1)
            y = data[f'{pollutant}_normalized'].values[1:]
        
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
        
            user_input_scaled = scaler.transform(np.array([[user_input]]))
        
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
        
            distances, indices = knn.kneighbors(user_input_scaled, n_neighbors=k, return_distance=True)
        
            y_pred = np.zeros(1)
        
            for i in range(1):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train[neighbor_indices]
        
                memberships = calculate_membership_inverse(neighbor_distances)
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
        
            return y_pred_original[0][0]
        
        # Muat dan bersihkan data dari file
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
        
        # Konversi kolom yang disebutkan ke tipe data integer
        data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']] = data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']].astype(int)
        
        # Input dari pengguna untuk semua polutan
        user_inputs = {}
        pollutants = ["pm_sepuluh", "pm_duakomalima", "sulfur_dioksida", "karbon_monoksida", "ozon", "nitrogen_dioksida"]
        
        for pollutant in pollutants:
            user_inputs[pollutant] = st.number_input(f"Masukkan konsentrasi {pollutant.replace('_', ' ').title()}:", min_value=0.0, key=pollutant)
        
        # Prediksi berdasarkan input pengguna
        if st.button("Prediksi Semua Polutan"):
            predictions = {}
            for pollutant in pollutants:
                prediction = fuzzy_knn_predict(data, pollutant, user_inputs[pollutant], k=3)
                predictions[pollutant] = prediction
            
            # Tampilkan semua prediksi dalam format tabel
            predictions_data = pd.DataFrame(predictions, index=[0])
            st.write(predictions_data)
            # Menentukan polutan tertinggi dan kategorinya
            max_pollutant = max(predictions, key=predictions.get)
            max_value = predictions[max_pollutant]
            
            # Menentukan kategori berdasarkan nilai
            if max_value <= 50:
                category = "Baik"
            elif max_value <= 100:
                category = "Sedang"
            elif max_value <= 150:
                category = "Tidak Sehat"
            else:
                category = "Berbahaya"
            
            # Buat DataFrame untuk polutan tertinggi
            highest_pollutant_data = pd.DataFrame({
                "Polutan_Tertinggi": [max_pollutant],
                "Nilai_Tertinggi": [max_value],
                "Kategori": [category]
            })
            
            # Tampilkan tabel polutan tertinggi
            st.write(highest_pollutant_data)

            # Membuat grafik untuk input dan hasil prediksi
            fig = go.Figure()
        
            # Menambahkan trace untuk input
            fig.add_trace(go.Bar(
                x=pollutants,
                y=list(user_inputs.values()),
                name='Input',
                marker_color='blue'
            ))
        
            # Menambahkan trace untuk prediksi
            fig.add_trace(go.Bar(
                x=pollutants,
                y=list(predictions.values()),
                name='Prediksi',
                marker_color='orange'
            ))
        
            # Menambahkan layout
            fig.update_layout(
                title='Input dan Hasil Prediksi Kualitas Udara',
                xaxis_title='Polutan',
                yaxis_title='Konsentrasi',
                barmode='group'
            )
        
            # Menampilkan grafik di Streamlit
            st.plotly_chart(fig)
        
            # Buat grafik keriting
            plt.figure(figsize=(10, 6))
            plt.plot(predictions.keys(), predictions.values(), marker='x', label='Hasil Prediksi', color='blue')
            
            # Tambahkan input pengguna ke grafik
            plt.plot(predictions.keys(), user_inputs.values(), marker='x', label='Input Pengguna', color='red')
            
            plt.title('Grafik Prediksi Kualitas Udara')
            plt.xlabel('Polutan')
            plt.ylabel('Konsentrasi (µg/m³)')
            plt.legend()
            plt.grid()
            st.pyplot(plt)

    # Menampilkan penanda
    st.markdown("---")  # Menambahkan garis pemisah
    st.write("Shinta Alya Imani Putri-200411100005 (Teknik Informatika)")
