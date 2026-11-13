"""Problem 1: HAR pipelines, from-scratch recurrent cells and experiments."""
import os, time, zipfile, urllib.request, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from config import *

CHANNELS = ['body_acc_x','body_acc_y','body_acc_z','body_gyro_x','body_gyro_y','body_gyro_z','total_acc_x','total_acc_y','total_acc_z']
HAR_URL = 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_device(): return torch.device('cuda' if DEVICE == 'cuda' and torch.cuda.is_available() else 'cpu')

def append_result(row):
    """Persist one completed experiment immediately using the assignment's exact schema."""
    cols=['problem','model','seed','seq_len','test_acc','macro_f1','test_mse','params','sec_per_epoch','val_loss_nats','bits_per_char','notes']
    path=ROOT/'results.csv'; pd.DataFrame([row],columns=cols).to_csv(path,mode='a',header=not path.exists() or path.stat().st_size==0,index=False)


def ensure_har(data_dir=None):
    """Download/extract the official archive; also supports its nested ZIP layout."""
    data_dir = Path(data_dir or ROOT / 'data'); data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / 'UCI HAR Dataset'
    if (target / 'train' / 'y_train.txt').exists(): return target
    outer = data_dir / 'har_download.zip'
    if not outer.exists(): urllib.request.urlretrieve(HAR_URL, outer)
    with zipfile.ZipFile(outer) as z: z.extractall(data_dir)
    nested = data_dir / 'UCI HAR Dataset.zip'
    if nested.exists():
        with zipfile.ZipFile(nested) as z: z.extractall(data_dir)
    if not (target / 'train' / 'y_train.txt').exists():
        # Search extracted tree to tolerate archive wrapper folders.
        matches = list(data_dir.rglob('y_train.txt'))
        if not matches: raise FileNotFoundError('Could not locate UCI HAR Dataset/train/y_train.txt')
        target = matches[0].parent.parent
    return target

def load_har(seed, data_dir=None):
    """Subject-wise train/validation split; channel stats fitted on training subjects only."""
    root = ensure_har(data_dir)
    def load_split(split):
        base = root / split / 'Inertial Signals'
        arrays = [np.loadtxt(base / f'{name}_{split}.txt', dtype=np.float32) for name in CHANNELS]
        x = np.stack(arrays, axis=-1)  # (N, 128, 9)
        y = np.loadtxt(root / split / f'y_{split}.txt', dtype=np.int64) - 1
        subjects = np.loadtxt(root / split / f'subject_{split}.txt', dtype=np.int64)
        return x, y, subjects
    x, y, s = load_split('train'); xt, yt, _ = load_split('test')
    subjects = np.unique(s)
    val_subjects = np.random.default_rng(seed).choice(subjects, 4, replace=False)
    val_mask = np.isin(s, val_subjects); train_mask = ~val_mask
    mean = x[train_mask].mean(axis=(0,1), keepdims=True)
    std = x[train_mask].std(axis=(0,1), keepdims=True).clip(1e-6)
    x = (x-mean)/std; xt=(xt-mean)/std
    return (x[train_mask], y[train_mask]), (x[val_mask],y[val_mask]), (xt,yt), y[train_mask], val_subjects

def plot_class_balance(y, path=None):
    counts=np.bincount(y, minlength=6); names=['Walking','Upstairs','Downstairs','Sitting','Standing','Laying']
    plt.figure(figsize=(7,4)); plt.bar(names,counts); plt.ylabel('Windows'); plt.title('HAR training class balance'); plt.xticks(rotation=20); plt.tight_layout()
    if path: plt.savefig(path,dpi=160)
    plt.show(); plt.close()

def make_adding(n,T,seed):
    rng=np.random.default_rng(seed); x=np.zeros((n,T,2),dtype=np.float32); x[:,:,0]=rng.random((n,T))
    i1=rng.integers(0,T//2,n); i2=rng.integers(T//2,T,n)
    x[np.arange(n),i1,1]=1; x[np.arange(n),i2,1]=1
    y=x[np.arange(n),i1,0]+x[np.arange(n),i2,0]
    return x,y.astype(np.float32)

def test_adding_generator():
    x,y=make_adding(1000,50,7); marks=x[:,:,1]
    assert np.all(marks.sum(1)==2)
    assert np.all(marks[:,:25].sum(1)==1) and np.all(marks[:,25:].sum(1)==1)
    assert np.allclose(y,(x[:,:,0]*marks).sum(1))
    return True

class MyRNNCell(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__(); self.weight_ih=nn.Parameter(torch.empty(hidden_size,input_size)); self.weight_hh=nn.Parameter(torch.empty(hidden_size,hidden_size)); self.bias_ih=nn.Parameter(torch.zeros(hidden_size)); self.bias_hh=nn.Parameter(torch.zeros(hidden_size)); nn.init.xavier_uniform_(self.weight_ih); nn.init.orthogonal_(self.weight_hh)
    def forward(self,x,h): return torch.tanh(x@self.weight_ih.T+self.bias_ih+h@self.weight_hh.T+self.bias_hh)

class MyLSTMCell(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__(); self.input_size=input_size; self.hidden_size=hidden_size
        self.weight_ih=nn.Parameter(torch.empty(4*hidden_size,input_size)); self.weight_hh=nn.Parameter(torch.empty(4*hidden_size,hidden_size)); self.bias_ih=nn.Parameter(torch.zeros(4*hidden_size)); self.bias_hh=nn.Parameter(torch.zeros(4*hidden_size)); nn.init.xavier_uniform_(self.weight_ih); nn.init.orthogonal_(self.weight_hh)
    def forward(self,x,state):
        h,c=state; gates=x@self.weight_ih.T+self.bias_ih+h@self.weight_hh.T+self.bias_hh
        i,f,g,o=gates.chunk(4,dim=-1); i=torch.sigmoid(i); f=torch.sigmoid(f); g=torch.tanh(g); o=torch.sigmoid(o)
        c_new=f*c+i*g; h_new=o*torch.tanh(c_new); return h_new,c_new

class MyGRUCell(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__(); self.weight_ih=nn.Parameter(torch.empty(3*hidden_size,input_size)); self.weight_hh=nn.Parameter(torch.empty(3*hidden_size,hidden_size)); self.bias_ih=nn.Parameter(torch.zeros(3*hidden_size)); self.bias_hh=nn.Parameter(torch.zeros(3*hidden_size)); nn.init.xavier_uniform_(self.weight_ih); nn.init.orthogonal_(self.weight_hh)
    def forward(self,x,h):
        gi=x@self.weight_ih.T+self.bias_ih; gh=h@self.weight_hh.T+self.bias_hh
        ir,iz,inn=gi.chunk(3,-1); hr,hz,hn=gh.chunk(3,-1)
        r=torch.sigmoid(ir+hr); z=torch.sigmoid(iz+hz); n=torch.tanh(inn+r*hn)
        return (1-z)*n+z*h

def verify_cells(batch=5,seq=7,d=9,h=HIDDEN_SIZE):
    """Copy identical parameters into scratch cells and PyTorch references; max error < 1e-5."""
    torch.manual_seed(9); x=torch.randn(seq,batch,d); h0=torch.randn(1,batch,h); c0=torch.randn(1,batch,h)
    errors={}
    a=nn.RNN(d,h,nonlinearity='tanh'); b=MyRNNCell(d,h)
    with torch.no_grad():
        b.weight_ih.copy_(a.weight_ih_l0); b.weight_hh.copy_(a.weight_hh_l0); b.bias_ih.copy_(a.bias_ih_l0); b.bias_hh.copy_(a.bias_hh_l0)
    ref,_=a(x,h0); hs=h0[0]; out=[]
    for t in range(seq): hs=b(x[t],hs); out.append(hs)
    errors['RNN']=(ref-torch.stack(out)).abs().max().item()
    a=nn.LSTM(d,h); b=MyLSTMCell(d,h)
    with torch.no_grad():
        b.weight_ih.copy_(a.weight_ih_l0); b.weight_hh.copy_(a.weight_hh_l0); b.bias_ih.copy_(a.bias_ih_l0); b.bias_hh.copy_(a.bias_hh_l0)
    ref,_=a(x,(h0,c0)); hs,cs=h0[0],c0[0]; out=[]
    for t in range(seq): hs,cs=b(x[t],(hs,cs)); out.append(hs)
    errors['LSTM']=(ref-torch.stack(out)).abs().max().item()
    a=nn.GRU(d,h); b=MyGRUCell(d,h)
    with torch.no_grad():
        b.weight_ih.copy_(a.weight_ih_l0); b.weight_hh.copy_(a.weight_hh_l0); b.bias_ih.copy_(a.bias_ih_l0); b.bias_hh.copy_(a.bias_hh_l0)
    ref,_=a(x,h0); hs=h0[0]; out=[]
    for t in range(seq): hs=b(x[t],hs); out.append(hs)
    errors['GRU']=(ref-torch.stack(out)).abs().max().item()
    assert max(errors.values())<1e-5, errors
    return errors

def scratch_parameter_counts(d=9,h=HIDDEN_SIZE):
    return {'RNN':h*d+h*h+h,'LSTM':4*h*d+4*h*h+4*h,'GRU':3*h*d+3*h*h+3*h}

class RecurrentClassifier(nn.Module):
    def __init__(self,kind,input_size=9,hidden_size=HIDDEN_SIZE,num_classes=6):
        super().__init__(); cls={'RNN':nn.RNN,'LSTM':nn.LSTM,'GRU':nn.GRU}[kind]; self.rnn=cls(input_size,hidden_size,batch_first=True); self.fc=nn.Linear(hidden_size,num_classes)
    def forward(self,x):
        out,_=self.rnn(x); return self.fc(out[:,-1])

class AddingRegressor(nn.Module):
    def __init__(self,kind,input_size=2,hidden_size=HIDDEN_SIZE):
        super().__init__(); cls={'RNN':nn.RNN,'LSTM':nn.LSTM,'GRU':nn.GRU}[kind]; self.rnn=cls(input_size,hidden_size,batch_first=True); self.fc=nn.Linear(hidden_size,1)
    def forward(self,x): return self.fc(self.rnn(x)[0][:,-1]).squeeze(-1)

def train_eval(model,train_loader,val_loader,epochs,task='class',device=None,clip=None):
    device=device or get_device(); model.to(device); opt=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE); loss_fn=nn.CrossEntropyLoss() if task=='class' else nn.MSELoss(); history=[]; t0=time.time()
    for ep in range(epochs):
        model.train(); total=0
        for xb,yb in train_loader:
            xb=xb.to(device); yb=yb.to(device); opt.zero_grad(); pred=model(xb); loss=loss_fn(pred,yb); loss.backward()
            if clip: nn.utils.clip_grad_norm_(model.parameters(),clip)
            opt.step(); total+=loss.item()*len(xb)
        history.append(total/len(train_loader.dataset))
    sec=(time.time()-t0)/epochs
    model.eval(); ys=[]; ps=[]; losses=[]
    with torch.no_grad():
        for xb,yb in val_loader:
            xb=xb.to(device); yb=yb.to(device); pred=model(xb); losses.append(loss_fn(pred,yb).item()); ys.extend(yb.cpu().numpy()); ps.extend((pred.argmax(1) if task=='class' else pred).cpu().numpy())
    if task=='class': return model,history,sec,accuracy_score(ys,ps),f1_score(ys,ps,average='macro'),confusion_matrix(ys,ps),np.array(ys),np.array(ps)
    return model,history,sec,float(np.mean((np.array(ys)-np.array(ps))**2)),np.array(ys),np.array(ps)

def loaders(x,y,batch=BATCH_SIZE,shuffle=True):
    ds=TensorDataset(torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.long if np.asarray(y).dtype.kind in 'iu' else torch.float32)); return DataLoader(ds,batch_size=batch,shuffle=shuffle)

def evaluate_model(model, loader, task='class', device=None):
    device=device or get_device(); model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for xb,yb in loader:
            pred=model(xb.to(device)); ys.extend(yb.cpu().numpy()); ps.extend((pred.argmax(1) if task=='class' else pred).cpu().numpy())
    if task=='class': return accuracy_score(ys,ps), f1_score(ys,ps,average='macro'), confusion_matrix(ys,ps), np.asarray(ys), np.asarray(ps)
    return float(np.mean((np.asarray(ys)-np.asarray(ps))**2)), np.asarray(ys), np.asarray(ps)

def run_har(epochs=HAR_EPOCHS):
    rows=[]; best=None
    for seed in SEEDS:
        (x,y),(xv,yv),(xt,yt),ytr,_=load_har(seed); plot_class_balance(ytr,FIG1/f'class_balance_seed{seed}.png')
        tr=loaders(x,y); validation=loaders(xv,yv,shuffle=False); test=loaders(xt,yt,shuffle=False)
        for kind in ['RNN','LSTM','GRU']:
            set_seed(seed); m=RecurrentClassifier(kind); m,hist,sec,*_=train_eval(m,tr,validation,epochs,'class')
            acc,f1,cm,_,_=evaluate_model(m,test,'class')
            rows.append(dict(problem='P1-HAR',model=kind,seed=seed,seq_len=128,test_acc=acc,macro_f1=f1,test_mse='',params=sum(p.numel() for p in m.parameters()),sec_per_epoch=sec,val_loss_nats='',bits_per_char='',notes=f'epochs={epochs}; validation subjects held out'))
            append_result(rows[-1])
            if best is None or acc>best[0]: best=(acc,kind,cm)
            plt.plot(hist,label=kind)
        plt.xlabel('Epoch'); plt.ylabel('Training cross-entropy'); plt.legend(); plt.tight_layout(); plt.savefig(FIG1/f'har_learning_seed{seed}.png',dpi=150); plt.close()
    if best:
        plt.figure(figsize=(6,5)); plt.imshow(best[2]); plt.title(f'Best HAR confusion matrix: {best[1]}'); plt.xlabel('Predicted'); plt.ylabel('True'); plt.colorbar(); plt.tight_layout(); plt.savefig(FIG1/'har_confusion_best.png',dpi=160); plt.close()
    return pd.DataFrame(rows)

def run_adding(epochs=ADDING_EPOCHS,n_train=10000,n_val=1000,n_test=1000):
    test_rows=[]; curves={}; test_adding_generator()
    for T in [50,100,200,400]:
        for seed in SEEDS:
            x,y=make_adding(n_train,T,0); xv,yv=make_adding(n_val,T,1); xt,yt=make_adding(n_test,T,2)
            tr=DataLoader(TensorDataset(torch.tensor(x),torch.tensor(y)),batch_size=BATCH_SIZE,shuffle=True)
            validation=DataLoader(TensorDataset(torch.tensor(xv),torch.tensor(yv)),batch_size=BATCH_SIZE)
            test=DataLoader(TensorDataset(torch.tensor(xt),torch.tensor(yt)),batch_size=BATCH_SIZE)
            for kind in ['RNN','LSTM','GRU']:
                set_seed(seed); m=AddingRegressor(kind); m,hist,sec,*_=train_eval(m,tr,validation,epochs,'reg')
                mse,_,_=evaluate_model(m,test,'reg')
                test_rows.append(dict(problem='P1-Adding',model=kind,seed=seed,seq_len=T,test_acc='',macro_f1='',test_mse=mse,params=sum(p.numel() for p in m.parameters()),sec_per_epoch=sec,val_loss_nats='',bits_per_char='',notes=f'epochs={epochs};solved={mse<0.05}'))
                append_result(test_rows[-1])
                if T==200: curves[(kind,seed)]=hist
    plt.figure(figsize=(7,4))
    for (kind,seed),hist in curves.items(): plt.plot(hist,alpha=.45,label=f'{kind}-{seed}')
    plt.xlabel('Epoch'); plt.ylabel('Training MSE'); plt.title('Adding problem learning curves (T=200)'); plt.legend(ncol=3,fontsize=7); plt.tight_layout(); plt.savefig(FIG1/'adding_T200_curves.png',dpi=160); plt.close()
    df=pd.DataFrame(test_rows); summary=df.groupby(['model','seq_len']).test_mse.agg(['mean','std']).reset_index(); summary.to_csv(ROOT/'problem1'/'adding_summary.csv',index=False)
    solved=df.assign(solved=df.test_mse<.05).groupby(['model','seq_len']).solved.sum().reset_index(); solved.to_csv(ROOT/'problem1'/'adding_solved.csv',index=False)
    plt.figure(figsize=(7,4))
    for kind,g in summary.groupby('model'): plt.errorbar(g.seq_len,g['mean'],yerr=g['std'].fillna(0),marker='o',capsize=3,label=kind)
    plt.axhline(0.167,linestyle='--',label='constant baseline 0.167'); plt.xlabel('Sequence length T'); plt.ylabel('Test MSE (mean ± std)'); plt.legend(); plt.tight_layout(); plt.savefig(FIG1/'adding_mse_vs_T.png',dpi=160); plt.close()
    return df

def gradient_profile(kind='RNN',T=100,seed=42,trained=False):
    """Return ||dL/dh_t|| from an explicit unroll; h_t retains its gradient."""
    set_seed(seed); x,y=make_adding(64,T,seed); model=AddingRegressor(kind); device=get_device(); model.to(device)
    if trained:
        tr=DataLoader(TensorDataset(torch.tensor(x),torch.tensor(y)),batch_size=32,shuffle=True); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
        for _ in range(5):
            for xb,yb in tr: opt.zero_grad(); nn.functional.mse_loss(model(xb.to(device)),yb.to(device)).backward(); opt.step()
    Cell={'RNN':nn.RNNCell,'LSTM':nn.LSTMCell,'GRU':nn.GRUCell}[kind](2,HIDDEN_SIZE).to(device)
    # Transfer the trained sequence layer's weights into the corresponding one-step cell.
    src=model.rnn
    with torch.no_grad():
        for name in ['weight_ih','weight_hh','bias_ih','bias_hh']:
            getattr(Cell,name).copy_(getattr(src,name+'_l0'))
    xx=torch.tensor(x,device=device); h=torch.zeros(xx.size(0),HIDDEN_SIZE,device=device); c=torch.zeros_like(h); states=[]
    for t in range(T):
        if kind=='LSTM': h,c=Cell(xx[:,t,:],(h,c))
        else: h=Cell(xx[:,t,:],h)
        h.retain_grad(); states.append(h)
    pred=model.fc(h).squeeze(-1); loss=((pred-torch.tensor(y,device=device))**2).mean(); loss.backward()
    return np.array([st.grad.norm().item() for st in states])

def run_gradient_plots():
    plt.figure(figsize=(7,4))
    for kind in ['RNN','LSTM','GRU']:
        for trained in [False,True]:
            g=gradient_profile(kind,100,42,trained); plt.plot(np.arange(100),g+1e-12,label=f'{kind} '+('after' if trained else 'before'))
    plt.yscale('log'); plt.xlabel('Time step'); plt.ylabel('||dL/dx_t|| (proxy)'); plt.legend(ncol=2,fontsize=8); plt.tight_layout(); plt.savefig(FIG1/'gradient_norms.png',dpi=160); plt.close()

def run_hidden_sweep(epochs=HAR_EPOCHS):
    rows=[]
    for seed in SEEDS:
        (x,y),(xv,yv),(xt,yt),_,_=load_har(seed); tr=loaders(x,y); va=loaders(xv,yv,shuffle=False); te=loaders(xt,yt,shuffle=False)
        for h in [16,32,64,128]:
            for kind in ['LSTM','GRU']:
                set_seed(seed); m=RecurrentClassifier(kind,hidden_size=h); m,_,sec,*_=train_eval(m,tr,va,epochs,'class'); acc,f1,_,_,_=evaluate_model(m,te,'class')
                rows.append(dict(seed=seed,hidden_size=h,model=kind,test_acc=acc,macro_f1=f1,params=sum(p.numel() for p in m.parameters()),sec_per_epoch=sec))
    df=pd.DataFrame(rows); df.to_csv(ROOT/'problem1'/'hidden_sweep.csv',index=False)
    plt.figure(figsize=(7,4))
    for kind,g in df.groupby('model'):
        summary=g.groupby('hidden_size').test_acc.agg(['mean','std']).reset_index(); plt.errorbar(summary.hidden_size,summary['mean'],yerr=summary['std'].fillna(0),marker='o',capsize=3,label=kind)
    plt.xlabel('Hidden size'); plt.ylabel('HAR test accuracy (mean ± std)'); plt.legend(); plt.tight_layout(); plt.savefig(FIG1/'hidden_size_sweep.png',dpi=160); plt.close(); return df

def run_ablation(epochs=ADDING_EPOCHS):
    """Chosen variant: LSTM forget-gate bias 0 vs 1 at T=200 (documented assumption)."""
    rows=[]
    for seed in SEEDS:
        x,y=make_adding(10000,200,0); xt,yt=make_adding(1000,200,2)
        tr=DataLoader(TensorDataset(torch.tensor(x),torch.tensor(y)),batch_size=BATCH_SIZE,shuffle=True); va=DataLoader(TensorDataset(torch.tensor(xt),torch.tensor(yt)),batch_size=BATCH_SIZE)
        for bias in [0.,1.]:
            set_seed(seed); m=AddingRegressor('LSTM')
            # PyTorch gate order is input, forget, cell, output; set forget slice in both bias vectors.
            h=m.rnn.hidden_size
            with torch.no_grad(): m.rnn.bias_ih_l0[h:2*h].fill_(bias); m.rnn.bias_hh_l0[h:2*h].zero_()
            m, hist, sec, mse,_,_=train_eval(m,tr,va,epochs,'reg')
            rows.append(dict(seed=seed,forget_bias=bias,test_mse=mse,sec_per_epoch=sec))
    df=pd.DataFrame(rows); df.to_csv(ROOT/'problem1'/'ablation.csv',index=False)
    plt.figure();
    for b,g in df.groupby('forget_bias'): plt.plot(g.seed,g.test_mse,'o-',label=f'forget bias {b:g}')
    plt.xlabel('Seed'); plt.ylabel('Test MSE'); plt.legend(); plt.tight_layout(); plt.savefig(FIG1/'ablation_forget_bias.png',dpi=160); plt.close(); return df
