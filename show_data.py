import pandas as panda
import psycopg2
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt

conn = psycopg2.connect(
    dbname="latihandata", user="postgres",
    password="1234", host="localhost"
)

query = """
SELECT tipe_kontrak, 
    CASE
        WHEN tenure_bulan <= 12 THEN '0-12 Bulan'
        WHEN tenure_bulan <= 24 THEN '13-24 Bulan'
        WHEN tenure_bulan <= 36 THEN '25-36 Bulan'
        WHEN tenure_bulan <= 48 THEN '37-48 Bulan'
        ELSE '49+ Bulan'
    END AS kelompok_tenure, 
    COUNT(*) as jumlah,
    AVG(churn) as churn
FROM Pelanggan
GROUP BY tipe_kontrak, kelompok_tenure
ORDER BY tipe_kontrak, kelompok_tenure
"""
hasil = panda.read_sql(query, conn)
panda.set_option('display.max_rows', None)
dataframe = panda.DataFrame(hasil)

# Preprocessing
x = panda.get_dummies(dataframe[["tipe_kontrak", "kelompok_tenure"]]).astype(np.float32)
y = dataframe["churn"].values.astype(np.float32)

# Simpan urutan nama kolom untuk mapping
features_name = list(x.columns)
print("Daftar Input: ", features_name)

# Buat Model Keras
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(x.shape[1],), name="input_features"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid", name="output_churn"),
])

model.compile(optimizer="adam", loss="mse")
model.fit(x, y, epochs=200, verbose=0)

model.export("save_churn_model")

# Visualisasikan Data di matplot
#for tipe, group in hasil.groupby('tipe_kontrak'):
#    plt.plot(group['kelompok_tenure'], group['churn'], label=tipe, marker='o')

#plt.title('Tren Churn Rate')
#plt.xlabel('Kelompok Tenure')
#plt.ylabel('Churn Rate')
#plt.grid(True, linestyle='--', alpha=0.6)
#plt.tight_layout()
#plt.show()