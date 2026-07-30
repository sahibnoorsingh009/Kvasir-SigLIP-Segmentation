from huggingface_hub import hf_hub_download
REPO='Sahibnoor1/kvasir-siglip2-segmentation-checkpoints'
FILES=['checkpoints/resunet_paper/seed_42/best.pt','checkpoints/siglip2_full/seed_42/best.pt','configs/resunet_paper.yaml','configs/siglip2_full.yaml']
for f in FILES:
    print('Downloaded:', hf_hub_download(repo_id=REPO, filename=f, local_dir='.'))
