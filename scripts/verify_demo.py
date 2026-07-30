from pathlib import Path
import torch
from src.config import load_config
from src.models import build_model
for cfgp,ckptp in [('configs/resunet_paper.yaml','checkpoints/resunet_paper/seed_42/best.pt'),('configs/siglip2_full.yaml','checkpoints/siglip2_full/seed_42/best.pt')]:
    cfg=load_config(cfgp); model=build_model(cfg); state=torch.load(ckptp,map_location='cpu',weights_only=False); model.load_state_dict(state['model'],strict=True)
    print(cfg['experiment_name'],sum(p.numel() for p in model.parameters()),'OK')
