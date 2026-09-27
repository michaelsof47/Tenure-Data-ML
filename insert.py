import pandas as panda
import psycopg2

conn = psycopg2.connect(
    dbname="latihandata", user="postgres",
    password="1234", host="localhost" 
)
cur = conn.cursor()

customer = panda.read_csv("data.csv")

for _, row in customer.iterrows():
    cur.execute(
        """
        INSERT INTO pelanggan
        (customer_id, tenure_bulan, biaya_bulanan, jumlah_komplain, tipe_kontrak, pakai_layanan_tambahan, churn)
        VALUES(%s, %s, %s, %s, %s, %s, %s)
        """,
        (row["customer_id"], row["tenure_bulan"], row["biaya_bulanan"], row["jumlah_komplain"], row["tipe_kontrak"], row["pakai_layanan_tambahan"], row["churn"])
    )

conn.commit()
cur.close()
conn.close()