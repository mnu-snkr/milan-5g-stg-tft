import pandas as pd
import torch
from pytorch_forecasting.data import TimeSeriesDataSet, MultiNormalizer, GroupNormalizer

def build_dataloaders(file_path: str, batch_size: int = 128):
    df = pd.read_parquet(file_path)
    target_grids = [5060, 4259, 4456]
    df = df[df["GridID"].isin(target_grids)].copy()

    df["timestamp"] = pd.to_datetime(df["TimeInterval"], unit='ms')
    df["hour_of_day"] = df["timestamp"].dt.hour.astype(str)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(str)
    
    df["time_idx"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // 600
    df["time_idx"] = df["time_idx"].astype(int)

    max_time_idx = df["time_idx"].max()
    training_cutoff = max_time_idx - 288
    validation_cutoff = max_time_idx - 144

    train_df = df[df["time_idx"] <= training_cutoff].copy()
    pivot_corr = train_df.pivot(index="time_idx", columns="GridID", values="internet")
    adjacency_matrix = torch.tensor(pivot_corr.corr(method="pearson").values, dtype=torch.float32)

    pivot_df = df.pivot(index="time_idx", columns="GridID", values="internet").reset_index()
    pivot_df.columns = ["time_idx", "internet_4259", "internet_4456", "internet_5060"]
    temporal_meta = df[["time_idx", "hour_of_day", "day_of_week", "timestamp"]].drop_duplicates(subset=["time_idx"])
    wide_df = pd.merge(pivot_df, temporal_meta, on="time_idx", how="left")
    wide_df["city_group"] = "Milan"

    target_columns = ["internet_5060", "internet_4259", "internet_4456"]
    training_dataset = TimeSeriesDataSet(
        wide_df[wide_df["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target=target_columns,
        group_ids=["city_group"],
        min_encoder_length=864, max_encoder_length=864,
        min_prediction_length=144, max_prediction_length=144,
        time_varying_known_categoricals=["hour_of_day", "day_of_week"],
        time_varying_unknown_reals=target_columns,
        target_normalizer=MultiNormalizer([GroupNormalizer(groups=["city_group"], transformation="softplus")] * 3),
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, wide_df[wide_df["time_idx"] <= validation_cutoff], predict=True, stop_randomization=True
    )
    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, wide_df, predict=True, stop_randomization=True
    )

    train_dl = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
    val_dl = validation_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=2)
    test_dl = test_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=2)

    return training_dataset, train_dl, val_dl, test_dl, adjacency_matrix