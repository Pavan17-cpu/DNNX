"""Problem 2: attention and a small Transformer implemented without nn.Transformer APIs."""
import math, time, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from config import *
from problem1.core import set_seed, get_device, load_har, loaders, append_result

SHAKESPEARE_URL='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

def scaled_dot_product_attention(Q,K,V,mask=None):
    """Q,K,V: (batch, heads, query/key length, head_dim); mask True means allowed."""
    scores=Q @ K.transpose(-2,-1) / math.sqrt(Q.size(-1))
    if mask is not None: scores=scores.masked_fill(~mask, float('-inf'))
    weights=torch.softmax(scores,dim=-1)
    return weights @ V, weights

def test_attention():
    torch.manual_seed(2); q=torch.randn(2,3,5,4); k=torch.randn(2,3,5,4); v=torch.randn(2,3,5,4)
    out,w=scaled_dot_product_attention(q,k,v)
    ref=torch.nn.functional.scaled_dot_product_attention(q,k,v)
    assert torch.allclose(out,ref,atol=1e-5,rtol=1e-5)
    assert torch.allclose(w.sum(-1),torch.ones_like(w.sum(-1)),atol=1e-6)
    n=5; causal=torch.tril(torch.ones(n,n,dtype=torch.bool))[None,None,:,:]
    _,wc=scaled_dot_product_attention(q[:,:,:n],k[:,:,:n],v[:,:,:n],causal)
    assert torch.allclose(wc.masked_select(~causal.expand_as(wc)),torch.zeros_like(wc.masked_select(~causal.expand_as(wc))))
    return {'max_error_vs_torch':(out-ref).abs().max().item(),'row_sum_max_error':(w.sum(-1)-1).abs().max().item(),'future_weight_max':wc.masked_select(~causal.expand_as(wc)).abs().max().item()}

def three_token_example():
    # 3 tokens, dimension 2; intentionally small integer vectors.
    Q=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
    K=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
    V=torch.tensor([[1.,0.],[0.,2.],[3.,1.]])
    scores=Q@K.T/math.sqrt(2); weights=torch.softmax(scores,dim=-1); out=weights@V
    # Direct independently-written NumPy calculation is the code-side cross-check.
    q,k,v=Q.numpy(),K.numpy(),V.numpy(); s=q@k.T/np.sqrt(2); e=np.exp(s-s.max(axis=1,keepdims=True)); wn=e/e.sum(axis=1,keepdims=True); on=wn@v
    assert np.allclose(out.numpy(),on,atol=1e-7)
    return Q,K,V,scores,weights,out

class MultiHeadAttention(nn.Module):
    def __init__(self,d,heads,dropout=0.):
        super().__init__(); assert d%heads==0; self.d=d; self.heads=heads; self.dk=d//heads
        self.W_Q=nn.Linear(d,d); self.W_K=nn.Linear(d,d); self.W_V=nn.Linear(d,d); self.W_O=nn.Linear(d,d); self.drop=nn.Dropout(dropout)
    def forward(self,x,mask=None,need_weights=False):
        B,T,D=x.shape
        # Each projection: (B,T,D) -> (B,H,T,D/H)
        q=self.W_Q(x).view(B,T,self.heads,self.dk).transpose(1,2)
        k=self.W_K(x).view(B,T,self.heads,self.dk).transpose(1,2)
        v=self.W_V(x).view(B,T,self.heads,self.dk).transpose(1,2)
        y,w=scaled_dot_product_attention(q,k,v,mask)
        y=y.transpose(1,2).contiguous().view(B,T,D)
        y=self.drop(self.W_O(y))
        return (y,w) if need_weights else y

class EncoderBlock(nn.Module):
    def __init__(self,d,heads,ff_width,dropout=.1):
        super().__init__(); self.norm1=nn.LayerNorm(d); self.attn=MultiHeadAttention(d,heads,dropout); self.norm2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,ff_width),nn.GELU(),nn.Dropout(dropout),nn.Linear(ff_width,d)); self.drop=nn.Dropout(dropout)
    def forward(self,x,mask=None): x=x+self.drop(self.attn(self.norm1(x),mask)); x=x+self.drop(self.ff(self.norm2(x))); return x

class SinusoidalPE(nn.Module):
    def __init__(self,d,max_len=4096):
        super().__init__(); pos=torch.arange(max_len).float().unsqueeze(1); div=torch.exp(torch.arange(0,d,2).float()*(-math.log(10000.)/d)); pe=torch.zeros(max_len,d)
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div[:pe[:,1::2].shape[1]])
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x): return x+self.pe[:,:x.size(1)]

class HARTransformer(nn.Module):
    def __init__(self,d=MODEL_WIDTH,heads=NUM_HEADS,layers=NUM_LAYERS,ff=FF_WIDTH,pos='sinusoidal',patch=1):
        super().__init__(); self.patch=patch; self.embed=nn.Linear(9*patch,d); self.cls=nn.Parameter(torch.zeros(1,1,d)); self.pos_type=pos
        self.sin=SinusoidalPE(d) if pos=='sinusoidal' else None; self.learned=nn.Parameter(torch.zeros(1,130//patch,d)) if pos=='learned' else None
        self.blocks=nn.ModuleList([EncoderBlock(d,heads,ff) for _ in range(layers)]); self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,6)
    def forward(self,x):
        B,T,C=x.shape
        if self.patch>1: x=x[:,:T//self.patch*self.patch,:].reshape(B,-1,C*self.patch)
        x=self.embed(x); x=torch.cat([self.cls.expand(B,-1,-1),x],dim=1)
        if self.sin is not None: x=self.sin(x)
        elif self.learned is not None: x=x+self.learned[:,:x.size(1)]
        for block in self.blocks: x=block(x)
        return self.head(self.norm(x[:,0]))

class CausalBlock(nn.Module):
    def __init__(self,d,heads,ff,dropout=.1):
        super().__init__(); self.n1=nn.LayerNorm(d); self.attn=MultiHeadAttention(d,heads,dropout); self.n2=nn.LayerNorm(d); self.ff=nn.Sequential(nn.Linear(d,ff),nn.GELU(),nn.Linear(ff,d)); self.drop=nn.Dropout(dropout)
    def forward(self,x,causal=True):
        T=x.size(1); mask=torch.tril(torch.ones(T,T,device=x.device,dtype=torch.bool))[None,None] if causal else None
        x=x+self.drop(self.attn(self.n1(x),mask)); x=x+self.drop(self.ff(self.n2(x))); return x

class CharTransformer(nn.Module):
    def __init__(self,vocab,d=MODEL_WIDTH,heads=NUM_HEADS,layers=NUM_LAYERS,ff=FF_WIDTH,context=128,causal=True):
        super().__init__(); self.context=context; self.causal=causal; self.token=nn.Embedding(vocab,d); self.pos=nn.Embedding(context,d)
        self.blocks=nn.ModuleList([CausalBlock(d,heads,ff) for _ in range(layers)]); self.norm=nn.LayerNorm(d); self.out=nn.Linear(d,vocab)
    def forward(self,idx):
        B,T=idx.shape; pos=torch.arange(T,device=idx.device); x=self.token(idx)+self.pos(pos)[None,:,:]
        for b in self.blocks: x=b(x,self.causal)
        return self.out(self.norm(x))
    @torch.no_grad()
    def generate(self,idx,n,temperature=.8,top_k=10):
        self.eval()
        for _ in range(n):
            ctx=idx[:,-self.context:]; logits=self(ctx)[:,-1,:]/temperature
            if top_k: vals,inds=torch.topk(logits,min(top_k,logits.size(-1))); filt=torch.full_like(logits,float('-inf')); filt.scatter_(1,inds,vals); logits=filt
            probs=torch.softmax(logits,dim=-1); nxt=torch.multinomial(probs,1); idx=torch.cat([idx,nxt],dim=1)
        return idx

def load_shakespeare(data_dir=None):
    p=Path(data_dir or ROOT/'data'); p.mkdir(parents=True,exist_ok=True); f=p/'tinyshakespeare.txt'
    if not f.exists(): urllib.request.urlretrieve(SHAKESPEARE_URL,f)
    text=f.read_text(encoding='utf-8'); chars=sorted(set(text)); stoi={c:i for i,c in enumerate(chars)}; itos={i:c for c,i in stoi.items()}; ids=torch.tensor([stoi[c] for c in text],dtype=torch.long)
    split=int(.9*len(ids)); return ids[:split],ids[split:],stoi,itos

def get_lm_batch(data,batch_size,context,device):
    ix=torch.randint(0,len(data)-context-1,(batch_size,)); x=torch.stack([data[i:i+context] for i in ix]); y=torch.stack([data[i+1:i+context+1] for i in ix]); return x.to(device),y.to(device)

def estimate_loss(model,data,device,context=128,steps=30):
    model.eval(); vals=[]
    with torch.no_grad():
        for _ in range(steps): x,y=get_lm_batch(data,32,context,device); vals.append(nn.functional.cross_entropy(model(x).reshape(-1,model.out.out_features),y.reshape(-1)).item())
    model.train(); return float(np.mean(vals))

def train_lm(epochs=LM_EPOCHS,max_steps=2500,context=128):
    train,val,stoi,itos=load_shakespeare(); device=get_device(); model=CharTransformer(len(stoi),context=context).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4); losses=[]; start=time.time()
    for step in range(max_steps):
        x,y=get_lm_batch(train,64,context,device); logits=model(x); loss=nn.functional.cross_entropy(logits.reshape(-1,len(stoi)),y.reshape(-1)); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if (step+1)%max(1,max_steps//10)==0: losses.append((step+1,estimate_loss(model,val,device,context)))
    val_loss=estimate_loss(model,val,device,context,100); bpc=val_loss/math.log(2)
    plt.figure(); plt.plot(*zip(*losses)); plt.xlabel('Training step'); plt.ylabel('Validation loss (nats)'); plt.tight_layout(); plt.savefig(FIG2/'lm_learning_curve.png',dpi=160); plt.close()
    start_idx=torch.zeros((1,1),dtype=torch.long,device=device); samples=[]
    with open(ROOT/'problem2'/'generated_samples.txt','w',encoding='utf-8') as f:
        for temp in [.2,.8,1.5]:
            out=model.generate(start_idx,500,temp,top_k=10)[0].cpu().tolist()[1:]; sample=''.join(itos[i] for i in out); samples.append(sample); f.write(f'===== temperature {temp} =====\n{sample}\n\n')
    metrics={'val_loss_nats':val_loss,'bits_per_char':bpc,'seconds':time.time()-start,'params':sum(p.numel() for p in model.parameters())}
    append_result(dict(problem='P2-TinyShakespeare',model='Transformer-LM',seed=42,seq_len=context,test_acc='',macro_f1='',test_mse='',params=metrics['params'],sec_per_epoch=metrics['seconds']/max_steps,val_loss_nats=val_loss,bits_per_char=bpc,notes=f'steps={max_steps};temperature samples saved'))
    return model,metrics,samples

def train_har_transformer(epochs=TRANSFORMER_EPOCHS):
    rows=[]
    for seed in SEEDS:
        (x,y),(xv,yv),(xt,yt),_,_=load_har(seed); tr=loaders(x,y); validation=loaders(xv,yv,shuffle=False); test=loaders(xt,yt,shuffle=False)
        for pos in ['none','sinusoidal','learned']:
            torch.manual_seed(seed); m=HARTransformer(pos=pos).to(get_device()); opt=torch.optim.Adam(m.parameters(),lr=1e-3); t0=time.time(); best_val=-1.; best_state=None
            for _ in range(epochs):
                m.train()
                for xb,yb in tr: xb=xb.to(get_device()); yb=yb.to(get_device()); opt.zero_grad(); loss=nn.functional.cross_entropy(m(xb),yb); loss.backward(); opt.step()
                m.eval(); vy=[]; vp=[]
                with torch.no_grad():
                    for xb,yb in validation: vp.extend(m(xb.to(get_device())).argmax(1).cpu().tolist()); vy.extend(yb.tolist())
                from sklearn.metrics import accuracy_score
                score=accuracy_score(vy,vp)
                if score>best_val: best_val=score; best_state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
            sec=(time.time()-t0)/epochs
            if best_state is not None: m.load_state_dict(best_state)
            m.eval(); ys=[]; ps=[]
            with torch.no_grad():
                for xb,yb in test: ps.extend(m(xb.to(get_device())).argmax(1).cpu().tolist()); ys.extend(yb.tolist())
            from sklearn.metrics import accuracy_score, f1_score
            rows.append(dict(problem='P2-HAR',model=f'Transformer-{pos}',seed=seed,seq_len=128,test_acc=accuracy_score(ys,ps),macro_f1=f1_score(ys,ps,average='macro'),test_mse='',params=sum(p.numel() for p in m.parameters()),sec_per_epoch=sec,val_loss_nats='',bits_per_char='',notes=f'epochs={epochs}; best validation accuracy={best_val:.4f}'))
            append_result(rows[-1])
    return pd.DataFrame(rows)

def run_head_ablation(steps=500):
    """Train fixed-width language models with 1, 4 and 8 heads for a controlled comparison."""
    train,val,stoi,itos=load_shakespeare(); device=get_device(); rows=[]
    for heads in [1,4,8]:
        set_seed(42); model=CharTransformer(len(stoi),d=MODEL_WIDTH,heads=heads,layers=NUM_LAYERS,ff=FF_WIDTH).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4); recent=[]
        for step in range(steps):
            x,y=get_lm_batch(train,32,128,device); loss=nn.functional.cross_entropy(model(x).reshape(-1,len(stoi)),y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step(); recent.append(loss.item())
        vl=estimate_loss(model,val,device,128,50)
        row=dict(heads=heads,params=sum(p.numel() for p in model.parameters()),train_loss_mean=float(np.mean(recent[-100:])),val_loss_nats=vl,bits_per_char=vl/math.log(2))
        rows.append(row)
    df=pd.DataFrame(rows); df.to_csv(ROOT/'problem2'/'heads_ablation.csv',index=False)
    plt.figure(); plt.plot(df.heads,df.val_loss_nats,'o-'); plt.xlabel('Number of heads'); plt.ylabel('Validation loss (nats)'); plt.tight_layout(); plt.savefig(FIG2/'heads_ablation.png',dpi=160); plt.close(); return df

def run_context_benchmark():
    rows=[]; device=get_device()
    for n in [64,128,256,512]:
        m=CharTransformer(65,context=n).to(device); x=torch.randint(0,65,(8,n),device=device)
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t=time.time(); _=m(x); 
        if device.type=='cuda': torch.cuda.synchronize(); mem=torch.cuda.max_memory_allocated()/1024**2
        else: mem=float('nan')
        rows.append(dict(context=n,seconds=time.time()-t,peak_memory_mb=mem,attention_complexity='O(n^2*d)'))
    df=pd.DataFrame(rows); df.to_csv(ROOT/'problem2'/'context_benchmark.csv',index=False)
    plt.figure(); plt.plot(df.context,df.seconds,'o-'); plt.xlabel('Context length'); plt.ylabel('Forward-pass time (seconds)'); plt.tight_layout(); plt.savefig(FIG2/'context_time.png',dpi=160); plt.close()
    if df.peak_memory_mb.notna().any():
        plt.figure(); plt.plot(df.context,df.peak_memory_mb,'o-'); plt.xlabel('Context length'); plt.ylabel('Peak allocated GPU memory (MB)'); plt.tight_layout(); plt.savefig(FIG2/'context_memory.png',dpi=160); plt.close()
    return df

def plot_attention_heatmaps(model,stoi,itos,sentence='To be, or not to be, that is the question:'):
    device=next(model.parameters()).device
    model.eval()
    ids=torch.tensor([[stoi.get(c,0) for c in sentence[:model.context]]],device=device)
    x=model.token(ids)+model.pos(torch.arange(ids.size(1),device=device))[None,:,:]
    block=model.blocks[0]
    with torch.no_grad(): _,w=block.attn(block.n1(x),need_weights=True)
    for head in [0,min(1,w.size(1)-1)]:
        plt.figure(figsize=(7,5)); plt.imshow(w[0,head].detach().cpu(),aspect='auto'); plt.title(f'Attention heatmap: head {head}'); plt.xlabel('Key position'); plt.ylabel('Query position'); plt.tight_layout(); plt.savefig(FIG2/f'attention_head_{head}.png',dpi=160); plt.close()

def run_no_mask_ablation(steps=500,context=128):
    """Train identical causal and non-causal models; record loss and a generated sample."""
    train,val,stoi,itos=load_shakespeare(); device=get_device(); rows=[]; generated=[]
    for causal in [True,False]:
        set_seed(42); model=CharTransformer(len(stoi),context=context,causal=causal).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4); losses=[]
        for _ in range(steps):
            x,y=get_lm_batch(train,32,context,device); loss=nn.functional.cross_entropy(model(x).reshape(-1,len(stoi)),y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        vl=estimate_loss(model,val,device,context,50); prompt=torch.zeros((1,1),dtype=torch.long,device=device); gen=model.generate(prompt,500,.8,10)[0].cpu().tolist()[1:]; sample=''.join(itos[i] for i in gen)
        generated.append((causal,sample)); rows.append(dict(causal_mask=causal,train_loss_mean=float(np.mean(losses[-100:])),val_loss_nats=vl,bits_per_char=vl/math.log(2)))
    with open(ROOT/'problem2'/'no_mask_generation.txt','w',encoding='utf-8') as f:
        for causal,sample in generated: f.write(('===== causal mask =====\n' if causal else '===== NO causal mask =====\n')+sample+'\n\n')
    df=pd.DataFrame(rows); df.to_csv(ROOT/'problem2'/'causal_mask_ablation.csv',index=False)
    plt.figure(); plt.bar(['Causal mask','No causal mask'],df.val_loss_nats); plt.ylabel('Validation loss (nats)'); plt.tight_layout(); plt.savefig(FIG2/'causal_mask_ablation.png',dpi=160); plt.close()
    plt.figure(); plt.bar(['Causal mask','No causal mask'],df.train_loss_mean); plt.ylabel('Mean training loss (last 100 updates)'); plt.tight_layout(); plt.savefig(FIG2/'causal_mask_train_loss.png',dpi=160); plt.close(); return df

