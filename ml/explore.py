import pandas as pd

# %%

df = pd.read_parquet("/home/wael/Code/admo/data/processed/experiment_100k.parquet")
cf = pd.read_parquet("/home/wael/Code/admo/data/processed/clean_100k.parquet")

# %%
df_filter = df[df["anomaly_type"] == "combination"]
df_filter

# %%
# energy_mix anomaly
clean = cf.loc[121]
clean
anomaly = df.loc[121]
anomaly
# %%
# contextual anomaly
clean = cf.loc[147]
clean
anomaly = df.loc[147]
anomaly
# %%
# correlational anomaly
clean = cf.loc[54]
clean
anomaly = df.loc[54]
anomaly
# %%
# structural anomaly
clean = cf.loc[50]
clean
anomaly = df.loc[50]
anomaly
# %%
# combination anomaly
clean = cf.loc[528]
clean
anomaly = df.loc[528]
anomaly

# %%
changes = pd.DataFrame({"clean": clean, "anomaly": anomaly})

changes = changes[changes["clean"] != changes["anomaly"]]

print(changes)
