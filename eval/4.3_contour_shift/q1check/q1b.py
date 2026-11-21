# 补充：C5 在第 1 次查询的动作，离「手真的在那个旧位置时（第 0 次查询）发出的动作」有多近。
# f = (a_C5(q1) - a_C4(q1)) 在 (a(q0) - a_C4(q1)) 上的投影比例：1 表示与第 0 次查询的动作相同，0 表示与 C4 相同。
# 注意第 0、1 次查询的噪声种子不同，只作描述。另给后续几次查询里 C4 与 C5 的状态已经分开，不再是同状态对照。
import numpy as np, glob, sys
R=sys.argv[1]; rng=np.random.default_rng(0)
def load(b,c):
    d={}
    for f in glob.glob(f'{R}/{b}/{c}/*.npz'):
        z=np.load(f,allow_pickle=True); d[str(z['key'])]=(z['env_actions'].copy(),str(z['cluster']),bool(z['success']))
    return d
for b in ('B1','B2'):
    C4=load(b,'C4'); C5=load(b,'C5'); F=[];CL=[];cosv=[];S4=[];P=[]
    for k in sorted(C4):
        e4,cl,s4=C4[k]; e5,_,_=C5[k]
        a0=e4[0:4,:3].sum(0); a1=e4[4:8,:3].sum(0); a15=e5[4:8,:3].sum(0)
        v=a0-a1; nv=v@v
        if nv<1e-8: continue
        F.append(float((a15-a1)@v/nv)); CL.append(cl); S4.append(s4)
        da=a15-a1; cosv.append(float(da@v/(np.linalg.norm(da)*np.sqrt(nv)+1e-12)))
        P.append(float(da@a0/np.linalg.norm(a0))*100)
    F=np.array(F); cl=np.array(CL); u=np.unique(cl); idx={c:np.where(cl==c)[0] for c in u}
    bs=[F[np.concatenate([idx[c] for c in rng.choice(u,len(u))])].mean() for _ in range(10000)]
    S4=np.array(S4); P=np.array(P)
    print(f'{b}: f 均值 {F.mean():+.3f} [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]，中位数 {np.median(F):+.3f}；差向量与 (a0-a1) 的夹角余弦中位数 {np.median(cosv):+.2f}；键 {len(F)}')
    print(f'    按该键 C4 最终成败分层的投影 p 中位数（cm）：C4 成功 {np.median(P[S4]):+.2f}（{S4.sum()} 键），C4 失败 {np.median(P[~S4]):+.2f}（{(~S4).sum()} 键）')
