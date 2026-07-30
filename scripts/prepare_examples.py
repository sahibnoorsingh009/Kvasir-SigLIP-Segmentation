from pathlib import Path
import shutil, kagglehub
IDS=['cju13hp5rnbjx0835bf0jowgx','cju16whaj0e7n0855q7b6cjkm','cju14pxbaoksp0835qzorx6g6','ck2bxw18mmz1k0725litqq2mc','cju87xn2snfmv0987sc3d9xnq','cju32srle1xfq083575i3fl75']
root=Path(kagglehub.dataset_download('debeshjha1/kvasirseg'))
def locate(name):
    ds=[p for p in root.rglob('*') if p.is_dir() and p.name.lower()==name]
    if not ds: raise RuntimeError(f'No {name} folder found under {root}')
    return max(ds,key=lambda d:sum(x.is_file() for x in d.iterdir()))
def one(d,i):
    m=[p for p in d.glob(i+'.*') if p.is_file()]
    if len(m)!=1: raise RuntimeError(f'{i}: found {len(m)} files in {d}')
    return m[0]
img,msk=locate('images'),locate('masks')
outi,outm=Path('demo/examples/images'),Path('demo/examples/masks')
outi.mkdir(parents=True,exist_ok=True); outm.mkdir(parents=True,exist_ok=True)
for i in IDS:
    a,b=one(img,i),one(msk,i); shutil.copy2(a,outi/a.name); shutil.copy2(b,outm/b.name); print('Copied',i)
