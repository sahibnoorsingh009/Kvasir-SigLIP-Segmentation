from __future__ import annotations
import os, threading, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import cv2, numpy as np, torch
from huggingface_hub import hf_hub_download
from PIL import Image
from src.config import load_config
from src.models import build_model
REPO=os.getenv('MODEL_REPO_ID','Sahibnoor1/kvasir-siglip2-segmentation-checkpoints')
LOCK=threading.Lock()
@dataclass
class Pred: mask:np.ndarray; prob:np.ndarray; seconds:float

def rgb(x):
    a=np.asarray(x.convert('RGB') if isinstance(x,Image.Image) else x)
    if a.ndim==2:a=np.repeat(a[...,None],3,2)
    return a[...,:3].astype(np.uint8)
def load_weights(m,p):
    s=torch.load(p,map_location='cpu',weights_only=False); m.load_state_dict(s['model'] if 'model'in s else s,strict=True)
def getfile(f):
    p=Path(f)
    return p if p.exists() else Path(hf_hub_download(repo_id=REPO,filename=f,local_dir='.'))
def norm_res(a,n):
    a=cv2.resize(a,(n,n),interpolation=cv2.INTER_LINEAR)
    t=torch.from_numpy(a).permute(2,0,1).float()/255
    mean=torch.tensor([.485,.456,.406])[:,None,None]; std=torch.tensor([.229,.224,.225])[:,None,None]
    return ((t-mean)/std).unsqueeze(0)
class Service:
    def __init__(self):
        self.dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype=torch.bfloat16 if self.dev.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16
        self.rc=load_config('configs/resunet_paper.yaml'); self.sc=load_config('configs/siglip2_full.yaml')
        self.r=build_model(self.rc); load_weights(self.r,getfile('checkpoints/resunet_paper/seed_42/best.pt')); self.r.eval().to(self.dev)
        self.s=build_model(self.sc); load_weights(self.s,getfile('checkpoints/siglip2_full/seed_42/best.pt')); self.s.eval().to(self.dev)
    def ac(self): return torch.autocast('cuda',dtype=self.dtype) if self.dev.type=='cuda' else torch.autocast('cpu',enabled=False)
    @torch.inference_mode()
    def pres(self,a):
        h,w=a.shape[:2]; x=norm_res(a,int(self.rc['data']['image_size'])).to(self.dev)
        if self.dev.type=='cuda':torch.cuda.synchronize()
        t=time.perf_counter()
        with self.ac(): y=self.r(x)
        y=torch.nn.functional.interpolate(y,size=(h,w),mode='bilinear',align_corners=False)
        if self.dev.type=='cuda':torch.cuda.synchronize()
        q=torch.sigmoid(y)[0,0].float().cpu().numpy(); return Pred((q>=float(self.rc['training'].get('threshold',.5))).astype(np.uint8),q,time.perf_counter()-t)
    @torch.inference_mode()
    def psig(self,a):
        h,w=a.shape[:2]; z=self.s.processor(images=[Image.fromarray(a)],return_tensors='pt',padding='max_length',max_num_patches=int(self.sc['model'].get('max_num_patches',400)))
        kw={'pixel_values':z['pixel_values'].to(self.dev),'output_size':(h,w)}
        for k in ('pixel_attention_mask','spatial_shapes'):
            if z.get(k) is not None:kw[k]=z[k].to(self.dev)
        if self.dev.type=='cuda':torch.cuda.synchronize()
        t=time.perf_counter()
        with self.ac(): y=self.s(**kw)
        if y.shape[-2:]!=(h,w):y=torch.nn.functional.interpolate(y,size=(h,w),mode='bilinear',align_corners=False)
        if self.dev.type=='cuda':torch.cuda.synchronize()
        q=torch.sigmoid(y)[0,0].float().cpu().numpy(); return Pred((q>=float(self.sc['training'].get('threshold',.5))).astype(np.uint8),q,time.perf_counter()-t)
    def compare(self,x):
        a=rgb(x)
        with LOCK:return a,self.pres(a),self.psig(a)
def gtmask(m,a):
    if m is None:return None
    x=np.asarray(m.convert('L') if isinstance(m,Image.Image) else m)
    if x.ndim==3:x=cv2.cvtColor(x.astype(np.uint8),cv2.COLOR_RGB2GRAY)
    h,w=a.shape[:2]; return (cv2.resize(x.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>=128).astype(np.uint8)
def scores(p,t,e=1e-7):
    p=p.astype(bool);t=t.astype(bool);tp=np.logical_and(p,t).sum();fp=np.logical_and(p,~t).sum();fn=np.logical_and(~p,t).sum();tn=np.logical_and(~p,~t).sum()
    return {'Dice':(2*tp+e)/(2*tp+fp+fn+e),'IoU':(tp+e)/(tp+fp+fn+e),'Precision':(tp+e)/(tp+fp+e),'Recall':(tp+e)/(tp+fn+e),'Specificity':(tn+e)/(tn+fp+e)}
def maskimg(m):return np.repeat((m*255)[...,None],3,2)
def overlay(a,m,c):
    o=a.astype(float).copy();f=m.astype(bool);o[f]=.62*o[f]+.38*np.array(c);o=np.clip(o,0,255).astype(np.uint8);cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);b=cv2.cvtColor(o,cv2.COLOR_RGB2BGR);cv2.drawContours(b,cs,-1,tuple(reversed(c)),2);return cv2.cvtColor(b,cv2.COLOR_BGR2RGB)
def compare_overlay(a,r,s):
    b=cv2.cvtColor(a.copy(),cv2.COLOR_RGB2BGR)
    rc,_=cv2.findContours(r,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);sc,_=cv2.findContours(s,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(b,rc,-1,(255,100,0),3);cv2.drawContours(b,sc,-1,(0,255,0),3);return cv2.cvtColor(b,cv2.COLOR_BGR2RGB)
