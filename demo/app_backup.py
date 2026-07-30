from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
import gradio as gr, pandas as pd
from PIL import Image
from .inference import Service,gtmask,scores,maskimg,overlay,compare_overlay
B=Path(__file__).parent; meta=json.loads((B/'metadata.json').read_text())
def find(d,i):
    m=[p for p in d.glob(i+'.*') if p.is_file()];return m[0] if m else None
def choices():return [f"{v['title']} | {k}" for k,v in meta.items() if find(B/'examples/images',k)]
def load_example(c):
    if not c:return None,None,'No example selected.'
    i=c.rsplit('|',1)[1].strip();a=find(B/'examples/images',i);m=find(B/'examples/masks',i);v=meta[i]
    return Image.open(a).convert('RGB'),Image.open(m).convert('L') if m else None,f"### {v['title']}\n{v['description']}\n\n**Image ID:** `{i}`"
@lru_cache(maxsize=1)
def svc():return Service()
def run(image,mask,mode,choice):
    if mode=='Preloaded example': image,mask,_=load_example(choice)
    if image is None:raise gr.Error('Upload an image or select an example.')
    a,r,s=svc().compare(image);g=gtmask(mask,a)
    rows=[]
    for name,p in [('ResUNet',r),('SigLIP2 Full',s)]:
        z=scores(p.mask,g) if g is not None else {k:None for k in ['Dice','IoU','Precision','Recall','Specificity']}
        rows.append({'Model':name,**{k:(round(float(v),4) if v is not None else None) for k,v in z.items()},'Inference time (s)':round(p.seconds,4),'Predicted area (%)':round(100*p.mask.mean(),2)})
    status='Metrics use the supplied ground truth.' if g is not None else 'No ground truth supplied; overlap metrics are unavailable.'
    return a,maskimg(g) if g is not None else None,maskimg(r.mask),maskimg(s.mask),overlay(a,r.mask,(0,145,255)),overlay(a,s.mask,(0,220,80)),compare_overlay(a,r.mask,s.mask),pd.DataFrame(rows),status
CSS='.gradio-container{max-width:1500px!important}.hero{border-radius:22px;padding:26px;background:linear-gradient(135deg,#0f2747,#174b78);color:white}.notice{border:1px solid #e2b93b;background:#fff8df;border-radius:12px;padding:12px}'
with gr.Blocks(title='Kvasir-SEG Live Comparison',css=CSS,theme=gr.themes.Soft()) as demo:
    gr.HTML('<div class="hero"><h1>Kvasir-SEG Live Segmentation Comparison</h1><p>ResUNet versus fully fine-tuned SigLIP2</p></div>')
    gr.HTML('<div class="notice"><b>Research demonstration only.</b> Not for clinical diagnosis. Do not upload identifiable patient data.</div>')
    with gr.Row():
      with gr.Column(scale=1):
        mode=gr.Radio(['Preloaded example','Upload custom image'],value='Preloaded example',label='Input mode')
        ch=gr.Dropdown(choices=choices(),value=choices()[0] if choices() else None,label='Curated example')
        info=gr.Markdown(); image=gr.Image(type='pil',label='Endoscopy image'); mask=gr.Image(type='pil',image_mode='L',label='Optional ground-truth mask'); btn=gr.Button('Run comparison',variant='primary')
      with gr.Column(scale=2):
        with gr.Row(): original=gr.Image(label='Original'); gt=gr.Image(label='Ground truth')
        with gr.Row(): rm=gr.Image(label='ResUNet mask'); sm=gr.Image(label='SigLIP2 Full mask')
        with gr.Row(): ro=gr.Image(label='ResUNet overlay'); so=gr.Image(label='SigLIP2 overlay')
        co=gr.Image(label='Blue: ResUNet | Green: SigLIP2'); table=gr.Dataframe(label='Metrics',interactive=False); status=gr.Markdown()
    ch.change(load_example,ch,[image,mask,info]); demo.load(load_example,ch,[image,mask,info]); btn.click(run,[image,mask,mode,ch],[original,gt,rm,sm,ro,so,co,table,status],api_name='compare_models')
    gr.Markdown('[GitHub](https://github.com/sahibnoorsingh009/Kvasir-SigLIP-Segmentation) · [Checkpoints](https://huggingface.co/Sahibnoor1/kvasir-siglip2-segmentation-checkpoints)')
