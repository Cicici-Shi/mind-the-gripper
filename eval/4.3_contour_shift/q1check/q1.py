# 零卡时核对（反方建议）：同键 C4 与 C5 在第 0 次查询画面相同（C5 不足 L 步时画第 0 步的轮廓），
# 所以第 1 次查询处状态、本体、噪声种子都相同，画面只差轮廓画在当前位置（C4）还是 4 步前的位置（C5）。
# 量：第 1 段动作的指令平移之差在「第 0 段运动方向」上的投影。事后描述，不作判定。
import numpy as np, glob, os, sys
R=sys.argv[1]; rng=np.random.default_rng(0)
def load(b,c):
    d={}
    for f in glob.glob(f'{R}/{b}/{c}/*.npz'):
        z=np.load(f,allow_pickle=True); d[str(z['key'])]=(z['raw_actions'].copy(),z['env_actions'].copy(),str(z['cluster']))
    return d
def boot(vals,cl,B=10000):
    cl=np.array(cl); u=np.unique(cl); idx={c:np.where(cl==c)[0] for c in u}; v=np.array(vals); out=[]
    for _ in range(B):
        s=np.concatenate([idx[c] for c in rng.choice(u,len(u))]); out.append(v[s].mean())
    return np.percentile(out,[2.5,97.5])
def spearman(a,b):
    ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b)); return float(np.corrcoef(ra,rb)[0,1])
for b in ('B1','B2'):
    C={c:load(b,c) for c in ('C0','C3','C4','C5')}
    keys=sorted(C['C4']); same0=sum(np.array_equal(C['C4'][k][0][0],C['C5'][k][0][0]) for k in keys)
    print(f'===== {b}: 键 {len(keys)}；C4 与 C5 第 0 次查询原始输出逐位相同 {same0}')
    P=[];D=[];O=[];CL=[];G=[];N1=[];Pxy=[]
    for k in keys:
        r4,e4,cl=C['C4'][k]; r5,e5,_=C['C5'][k]
        if not np.array_equal(r4[0],r5[0]): continue
        d=e4[0:4,:3].sum(0); nd=np.linalg.norm(d)
        if nd<1e-4: continue
        da=e5[4:8,:3].sum(0)-e4[4:8,:3].sum(0); p=float(da@d/nd)
        P.append(p); D.append(nd); O.append(float(np.linalg.norm(da-p*d/nd))); CL.append(cl)
        G.append(float(np.abs(e5[4:8,6]-e4[4:8,6]).max())); N1.append(float(np.linalg.norm(e4[4:8,:3].sum(0))))
        dxy=d.copy(); dxy[2]=0; n2=np.linalg.norm(dxy); Pxy.append(float(da@dxy/n2) if n2>1e-4 else np.nan)
    P=np.array(P)*100; D=np.array(D)*100; O=np.array(O)*100; N1=np.array(N1)*100; Pxy=np.array(Pxy)*100
    lo,hi=boot(P,CL); r=P/D
    print(f'进入统计的键 {len(P)}；第 0 段指令位移 |d| 中位数 {np.median(D):.2f} cm（四分位 {np.percentile(D,25):.2f} 到 {np.percentile(D,75):.2f}）')
    print(f'投影 p =（C5 减 C4 的第 1 段指令位移）在运动方向上的分量：均值 {P.mean():+.3f} cm，95% 区间 [{lo:+.3f}, {hi:+.3f}]，中位数 {np.median(P):+.3f}，为正的键占 {np.mean(P>0)*100:.0f}%')
    print(f'  只看水平分量：均值 {np.nanmean(Pxy):+.3f} cm，为正占 {np.nanmean(Pxy>0)*100:.0f}%')
    lo2,hi2=boot(r,CL); print(f'比值 p/|d|：均值 {r.mean():+.3f} [{lo2:+.3f}, {hi2:+.3f}]，中位数 {np.median(r):+.3f}')
    print(f'剂量：|d| 与 p 的秩相关 {spearman(D,P):+.2f}；按 |d| 三等分的 p 中位数 ' + ' / '.join(f'{np.median(P[(D>=q0)&(D<=q1)]):+.3f}' for q0,q1 in zip(np.percentile(D,[0,33.3,66.7]),np.percentile(D,[33.3,66.7,100]))))
    print(f'垂直于运动方向的差的大小：中位数 {np.median(O):.3f} cm；第 1 段自身指令位移大小中位数 {np.median(N1):.2f} cm；夹爪指令有差别的键 {int(np.sum(np.array(G)>0))}')
    # 第 0 次查询处的量级参照（同状态同噪声，画面不同）
    for a,c in (('C3','C4'),('C0','C4'),('C0','C3')):
        v=[np.linalg.norm(C[a][k][1][0:4,:3].sum(0)-C[c][k][1][0:4,:3].sum(0))*100 for k in keys if k in C[a]]
        print(f'  参照：第 0 段指令位移 |{a} 减 {c}| 中位数 {np.median(v):.3f} cm')
