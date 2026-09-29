import pandas as panda
import psycopg2
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

import sys
import os
import json


# Konfigurasi Database
NUM = ["tenure_bulan", "biaya_bulanan", "jumlah_komplain"]
CAT = ["tipe_kontrak", "pakai_layanan_tambahan"]
TARGET = "churn"

# Akses Database

def run_query(sql, params=None) -> panda.DataFrame:
    conn = psycopg2.connect(
        dbname="latihandata", user="postgres",
        password="1234", host="localhost"
    )

    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description] # type: ignore
            return panda.DataFrame(cur.fetchall(), columns=cols)
    finally:
        conn.close()

def extract() -> panda.DataFrame:
    df = run_query("""
    SELECT customer_id, tenure_bulan, biaya_bulanan, jumlah_komplain, tipe_kontrak, pakai_layanan_tambahan, churn
    FROM "pelanggan"
    """)

    # Postgres numeric/avg dikembalikan ke Decimal -> ubah angka biasa
    for c in NUM + [TARGET]:
        df[c] = panda.to_numeric(df[c])
    print(f"extract: {len(df)} baris")
    return df

# Cek Dataset apakah ada kolom yang kosong ? kalo kosong diisi dengan median

def fillData(dataframe: panda.DataFrame) -> panda.DataFrame:
    dataframe["biaya_bulanan"] = dataframe["biaya_bulanan"].fillna(dataframe["biaya_bulanan"].median())
    return dataframe

# SQL Analisis

def analisis():
    query_tenure = f"""
    SELECT tipe_kontrak,
        CASE WHEN tenure_bulan <= 12 THEN '0-12'
            WHEN tenure_bulan <= 24 THEN '13-24'
            WHEN tenure_bulan <= 36 THEN '25-36'
            WHEN tenure_bulan <= 48 THEN '37-48'
        ELSE '49+' END AS kelompok_tenure,
    COUNT(*) AS jumlah,
    ROUND(AVG(churn) * 100, 1) AS churn_pct
    FROM "pelanggan"
    GROUP BY 1, 2 ORDER BY 1, 2
    """

    query_komplain = f"""
    SELECT jumlah_komplain, COUNT(*) AS jumlah,
        ROUND(AVG(churn) * 100, 1) AS churn_pct
        FROM "pelanggan" GROUP BY 1 ORDER BY 1
    """

    query_layanan = f"""
    SELECT tipe_kontrak, pakai_layanan_tambahan, COUNT(*) AS jumlah,
        ROUND(AVG(churn) * 100, 1) AS churn_pct
        FROM "pelanggan" GROUP BY 1, 2 ORDER BY 1, 2
    """

    hasil_tenure = run_query(query_tenure)

    urutan = ['0-12', '13-24', '25-36', '37-48', '49+']
    hasil_tenure['kelompok_tenure'] = panda.Categorical(
        hasil_tenure['kelompok_tenure'], categories=urutan, ordered=True
    )
    hasil_tenure = hasil_tenure.sort_values(['tipe_kontrak', 'kelompok_tenure'])

    for tipe, group in hasil_tenure.groupby('tipe_kontrak'):
        plt.plot(group['kelompok_tenure'], group['churn_pct'], label=tipe, marker='o')

    plt.title('Tren Churn Rate per Kelompok Tenure')
    plt.xlabel('Kelompok Tenure')
    plt.ylabel('Churn Rate (%)')
    plt.legend(title='Tipe Kontrak')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# Data Quality Check

def check(dataframe: panda.DataFrame):
    assert len(dataframe) > 0, "data kosong"
    assert dataframe["customer_id"].is_unique, "customer_id duplikat"
    assert dataframe[NUM + CAT + [TARGET]].notna().all().all(), "ada nilai kosong"
    assert set(dataframe[TARGET].unique()) <= {0, 1}, "churn bukan 0/1"
    assert dataframe[TARGET].nunique() == 2, "churn hanya punya satu kelas"
    assert (dataframe["tenure_bulan"] >= 0).all(), "ada tenure negatif"
    assert (dataframe["biaya_bulanan"] >= 0).all(), "ada tenure negatif"
    print("check: lolos")

# Featuring Engineering

def make_features(dataframe: panda.DataFrame, meta=None):
    x = dataframe[NUM].astype(float).copy()

    if meta is None:
        std = x.std().replace(0, 1.0)
        meta = {"mean": x.mean().to_dict(), "std": std.to_dict()}

    for c in NUM:
        x[c] = (x[c] - meta["mean"][c]) / meta["std"][c]
    
    dummies = panda.get_dummies(dataframe[CAT]).astype(float)
    x = panda.concat([x, dummies], axis=1)

    if "columns" not in meta:
        meta["columns"] = list(x.columns) # type: ignore
    x = x.reindex(columns=meta["columns"], fill_value=0.0) # type: ignore
    return x.values.astype(np.float32), meta

# Training

def train(x_tr: panda.DataFrame,y_tr: panda.DataFrame, x_val: panda.DataFrame, y_val: panda.DataFrame):
    model = keras.Sequential([
        keras.layers.Input(shape=(x_tr.shape[1],), name="input_features"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid", name="output_churn")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    stop = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    model.fit(x_tr, y_tr, validation_data=(x_val, y_val), epochs= 200, batch_size=32, callbacks=[stop], verbose=0) # type: ignore
    return model

# Pipeline Utama

def main():
    keras.utils.set_random_seed(42)
    
    dataframe = extract()
    dataframe = fillData(dataframe)
    check(dataframe)

    tr: panda.DataFrame
    val: panda.DataFrame
    te: panda.DataFrame
    tr, te = train_test_split(dataframe, test_size=0.2, stratify=dataframe[TARGET],
        random_state=42)
    tr, val = train_test_split(tr, test_size=0.2, stratify=tr[TARGET],
        random_state=42)

    x_tr, meta = make_features(tr)
    x_val, _ = make_features(val, meta)
    x_te, _ = make_features(te, meta)
    y_tr = tr[TARGET].values.astype(np.float32)
    y_val = val[TARGET].values.astype(np.float32)
    y_te = te[TARGET].values.astype(np.float32)

    model = train(x_tr, y_tr, x_val, y_val) # type: ignore

    # evaluasi + pembanding sederhana
    auc_nn = roc_auc_score(y_te, model.predict(x_te, verbose=0).ravel()) # type: ignore
    lr = LogisticRegression(max_iter=1000).fit(x_tr, y_tr) # type: ignore
    auc_lr = roc_auc_score(y_te, lr.predict_proba(x_te)[:, 1])
    print(f"AUC test - Keras: {auc_nn:.3f} | LogisticRegression: {auc_lr:.3f}  (0.5 = tebakan acak)")

    # gerbang kualitas: model yang tidak lebih baik dari tebakan acak tidak di-export
    assert auc_nn > 0.5, "model tidak lebih baik dari tebakan acak, tidak di-export"

    model.export("save_churn_model")
    with open(os.path.join("save_churn_model", "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"model + meta.json tersimpan di save_churn_model/'")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analisis":
        analisis()
    else:
        main()

