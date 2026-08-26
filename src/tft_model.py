import torch
from pytorch_forecasting.models import TemporalFusionTransformer
from stg_layer import TemporalGraphConvolution

class SpatioTemporalGraphTFT(TemporalFusionTransformer):
    def __init__(self, adjacency_matrix, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gcn = TemporalGraphConvolution(in_features=3, out_features=3, adjacency_matrix=adjacency_matrix)
        
    def forward(self, x):
        enc_enriched = self.gcn(x["encoder_cont"][:, :, :3])
        enc_rest = x["encoder_cont"][:, :, 3:]
        x["encoder_cont"] = torch.cat([enc_enriched, enc_rest], dim=-1)
        
        dec_enriched = self.gcn(x["decoder_cont"][:, :, :3])
        dec_rest = x["decoder_cont"][:, :, 3:]
        x["decoder_cont"] = torch.cat([dec_enriched, dec_rest], dim=-1)
        
        return super().forward(x)