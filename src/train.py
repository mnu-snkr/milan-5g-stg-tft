import lightning.pytorch as pl
from pytorch_forecasting.metrics import QuantileLoss, MultiLoss
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

# 1. Import your custom modules
from data_pipeline import build_dataloaders
from tft_model import SpatioTemporalGraphTFT

def main():
    # 2. Initialize Data
    # Note: In a real system, you would pass this path via command line arguments (e.g., argparse)
    file_path = "../data/raw/milan-traffic-aggregated-parquet" 
    
    print("Building DataLoaders and Graph Topology...")
    training_dataset, train_dl, val_dl, test_dl, adj_matrix = build_dataloaders(file_path, batch_size=128)

    # 3. Build Model Blueprint
    print("Initializing STG-TFT Architecture...")
    stg_tft = SpatioTemporalGraphTFT.from_dataset(
        training_dataset,
        adjacency_matrix=adj_matrix, 
        learning_rate=0.001,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.2,
        hidden_continuous_size=64,
        output_size=[3, 3, 3],
        loss=MultiLoss([
            QuantileLoss([0.10, 0.50, 0.90]), 
            QuantileLoss([0.10, 0.50, 0.90]), 
            QuantileLoss([0.10, 0.50, 0.90])
        ]),
        log_interval=10, 
        reduce_on_plateau_patience=4,
    )

    # 4. Execute Training Loop
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
    trainer = pl.Trainer(
        max_epochs=30, 
        accelerator="auto", 
        devices=1, 
        enable_model_summary=True, 
        callbacks=[early_stop_callback, LearningRateMonitor()]
    )

    print("Starting Model Training...")
    trainer.fit(stg_tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

if __name__ == "__main__":
    main()