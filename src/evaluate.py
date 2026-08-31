import matplotlib.pyplot as plt
import torch
import os
from data_pipeline import build_dataloaders
from tft_model import SpatioTemporalGraphTFT

def plot_forecast(actuals, predictions, grid_idx, grid_name, save_path):
    actual_series = actuals[grid_idx][0, :].cpu().numpy()
    pred_series = predictions[grid_idx][0, :, 1].cpu().numpy() # Index 1 is the 50th percentile (median)

    plt.figure(figsize=(12, 6))
    plt.plot(actual_series, label="Actual Traffic", color="black", linewidth=2)
    plt.plot(pred_series, label="STG-TFT Forecast (Median)", color="red", linestyle="--", linewidth=2)
    
    plt.title(f"{grid_name} - Traffic Forecast", fontsize=16, fontweight="bold")
    plt.xlabel("Time Steps (10-min intervals)", fontsize=12)
    plt.ylabel("Internet Traffic Volume", fontsize=12)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to {save_path}")
    plt.close()

def main():
    file_path = "/kaggle/input/datasets/manusankarkm/milan-traffic-aggregated-parquet/milan_traffic_aggregated.parquet"
    checkpoint_path = "/kaggle/input/notebooks/manusankarkm/stgnn-milan-prediction/lightning_logs/version_0/checkpoints/epoch=17-step=1062.ckpt" 
    output_dir = "../assets/"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading test data...")
    _, _, _, test_dl, adj_matrix = build_dataloaders(file_path, batch_size=128)
    

    print("Loading STG-TFT Checkpoint...")
    model = SpatioTemporalGraphTFT.load_from_checkpoint(
        checkpoint_path,
        adjacency_matrix=adj_matrix # <-- Inject it here
    )
    model.eval()

    print("Forecasting test horizon...")
    with torch.no_grad():
        predictions, x, *_ = model.predict(test_dl, mode="prediction", return_x=True, trainer_kwargs={"accelerator": "gpu", "devices": 1})
        actuals = x["decoder_target"]

    print("Generating visual evidence...")
    plot_forecast(actuals, predictions, grid_idx=1, grid_name="University Grid (Parasitic Gain)", 
                  save_path=f"{output_dir}/university_gain.png")
    
    plot_forecast(actuals, predictions, grid_idx=0, grid_name="City Center (Hub Penalty)", 
                  save_path=f"{output_dir}/city_center_penalty.png")

if __name__ == "__main__":
    main()