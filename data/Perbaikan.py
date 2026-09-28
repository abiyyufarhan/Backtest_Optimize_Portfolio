import pandas as pd

# 1. Membaca file Excel
file_name = "BI-7Day-RR.xlsx"
df = pd.read_excel(file_name)

# 2. Pemetaan nama bulan Indonesia ke Bahasa Inggris agar bisa dibaca oleh datetime
bulan_map = {
    "Januari": "January",
    "Februari": "February",
    "Maret": "March",
    "April": "April",
    "Mei": "May",
    "Juni": "June",
    "Juli": "July",
    "Agustus": "August",
    "September": "September",
    "Oktober": "October",
    "November": "November",
    "Desember": "December",
}

# 3. Mengubah nama bulan ke Bahasa Inggris (jika ada)
df["Tanggal_EN"] = df["Tanggal"].astype(str)
for id_m, en_m in bulan_map.items():
    df["Tanggal_EN"] = df["Tanggal_EN"].str.replace(id_m, en_m)

# 4. Mengubah ke tipe data datetime
df["Tanggal_dt"] = pd.to_datetime(df["Tanggal_EN"], format="mixed")

# 5. Sort/Urutkan dari tanggal terlama ke terbaru (ascending)
df_sorted = df.sort_values(by="Tanggal_dt", ascending=True).reset_index(drop=True)

# 6. Ubah tampilan format tanggal ke YYYY-MM-DD
df_sorted["Tanggal"] = df_sorted["Tanggal_dt"].dt.strftime("%Y-%m-%d")

# Pilih kolom akhir yang diinginkan
df_final = df_sorted[["Tanggal", "BI-7Day-RR"]]

# Simpan ke file Excel baru
df_final.to_excel("BI-7Day-RR_Sorted.xlsx", index=False)