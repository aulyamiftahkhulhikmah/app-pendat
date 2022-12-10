import pandas as pd
import streamlit as st
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

st.write(""" 
# Breast Cancer Coimbra
Aulya Miftahkhul Hikmah (200411100050)
""")

st.write("=========================================================================")

tab1, tab2, tab3, tab4, tab5= st.tabs(["Description", "Dataset", "Preprocessing", "Modeling", "Implementation"])

with tab1:
    st.subheader("""Pengertian""")
    st.write("""
    Dataset ini digunakan untuk menentukan apakah pasien menderita penyakit kanker payudara atau tidak.
    """)
    st.markdown(
        """
        Dataset ini memiliki beberapa fitur yaitu :
        - Age = umur (years)
        - BMI = Body Mass Index (BMI) atau Indeks Massa Tubuh (IMT)(kg/m2)
        - Glucose = kadar gula dalam tubuh(mg/dL)
        - Insulin = mengontrol kadar gula dalam darah(µU/mL)
        - HOMA = Homeostatic model assessment (HOMA) adalah metode untuk menilai fungsi sel pankreas
        - Leptin = hormon yang dibuat oleh sel lemak. Tugasnya mengendalikan nafsu makan serta rasa lapar (ng/mL)
        - Adiponectin = suatu protein yang spesifik disekresikan oleh adiposit dengan peran pada homeostasis glukosa dan lemak(µg/mL)
        - Resistin =merupakan hormon yang disebut-sebut menyebabkan obesitas(ng/mL)
        - MCP-1(pg/dL)
        - Classification(1/2)
        """
        )

with tab2:
    st.subheader("""Breast Cancer Coimbra Dataset""")
    df = pd.read_csv('https://raw.githubusercontent.com/aulyamiftahkhulhikmah/Dataset/main/dataR2.csv')
    st.dataframe(df) 
with tab3:
    st.subheader("""Rumus Normalisasi Data""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Keterangan :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    #Mendefinisikan Varible X dan Y
    X = df.drop(columns=['Classification'])
    y = df['Classification'].values
    df_min = X.min()
    df_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.Classification).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        'Positive' : [dumies[1]],
        'Negative' : [dumies[0]]
    })

    st.write(labels)
with tab4:
    st.subheader("""Metode Yang Digunakan""")
    #Split Data 
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.write("Pilih Metode yang digunakan : ")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-NN')
        destree = st.checkbox('Decission Tree')
        submitted = st.form_submit_button("Submit")

        #Gaussian Naive Bayes
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        probas = gaussian.predict_proba(test)
        probas = probas[:,1]
        probas = probas.round()

        gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

        #KNN
        K=10
        knn=KNeighborsClassifier(n_neighbors=K)
        knn.fit(training,training_label)
        knn_predict=knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

        #Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        #Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        if submitted :
            if naive :
                st.write('Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if k_nn :
                st.write("K-NN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree :
                st.write("Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
        
        grafik = st.form_submit_button("Grafik Akurasi")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)
with tab5:
        st.subheader("Form Implementasi")
        with st.form("my_form"):
            Age = st.slider('Usia pasien', 24, 89)
            BMI = st.slider('Body Mass Index (BMI) atau Indeks Massa Tubuh (IMT)', 18.4, 38.6)
            Glucose = st.slider('kadar gula dalam tubuh', 60, 201)
            Insulin = st.slider('mengontrol kadar gula dalam darah', 2.43, 58.5)
            HOMA = st.slider('Homeostatic model assessment (HOMA) untuk menilai fungsi sel pankreas', 0.47, 25.1)
            Leptin = st.slider('hormon yang dibuat oleh sel lemak untuk mengendalikan nafsu makan serta rasa lapar', 4.31, 90.3)
            Adiponectin = st.slider('suatu protein yang spesifik disekresikan oleh adiposit dengan peran pada homeostasis glukosa dan lemak', 1.66, 38.00)
            Resestin = st.slider('hormon yang disebut-sebut menyebabkan obesitas', 3.21, 82.1)
            MCP= st.slider('MCP-1(pg/dL)', 45.8, 17000.00)
            model = st.selectbox('Model untuk prediksi',
                    ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

            prediksi = st.form_submit_button("Submit")
            if prediksi:
                inputs = np.array([
                    Age,
                    BMI,
                    Glucose,
                    Insulin,
                    HOMA,
                    Leptin,
                    Adiponectin,
                    Resestin,
                    MCP,
                ])
                
                df_min = X.min()
                df_max = X.max()
                input_norm = ((inputs - df_min) / (df_max - df_min))
                input_norm = np.array(input_norm).reshape(1, -1)

                if model == 'Gaussian Naive Bayes':
                    mod = gaussian
                if model == 'K-NN':
                    mod = knn 
                if model == 'Decision Tree':
                    mod = dt

                input_pred = mod.predict(input_norm)

                st.subheader('Hasil Prediksi')
                st.write('Menggunakan Pemodelan :', model)

                if input_pred == 1:
                    st.error('Positive')
                else:
                    st.success('Negative')




