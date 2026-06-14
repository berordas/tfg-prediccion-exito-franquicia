import numpy as np, json, time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")

# ---------- Cargas de la Tabla 2 (Cuesta et al., 2018) ----------
LG = {  # cargas en el factor general, ítems 1-10 por dimensión
 'SE':[.72,.75,.71,.69,.70,.79,.65,.72,.78,.70],
 'AU':[.43,.14,.61,.63,.32,.26,.30,.52,.55,.56],
 'IN':[.48,.62,.56,.63,.48,.56,.60,.74,.58,.56],
 'IL':[.47,.63,.50,.52,.42,.31,.60,.39,.43,.46],
 'AM':[.59,.67,.61,.52,.67,.70,.60,.69,.66,.58],
 'OP':[.59,.55,.56,.62,.79,.62,.50,.51,.71,.48],
 'ST':[.72,.27,.54,.37,.46,.42,.38,.42,.36,.53],
 'RT':[.61,.73,.50,.56,.66,.59,.57,.64,.56,.46]}
LS = {  # cargas en el factor de grupo (filas 1-10 de la Tabla 2)
 'SE':[-.02,-.28,.54,-.15,.24,-.02,.49,.12,.11,-.26],
 'AU':[.47,.61,.18,.02,.70,.66,.67,.52,.36,.21],
 'IN':[.48,.49,.30,.45,.32,.49,.49,.35,.41,.41],
 'IL':[.69,.40,.61,.30,.44,.43,.24,.64,.55,.66],
 'AM':[.40,.37,.34,.37,-.07,.05,.26,.06,.46,.58],
 'OP':[.57,.56,.39,.45,.31,.44,.58,.47,.19,.61],
 'ST':[.20,.47,.41,.42,.48,.74,.67,.47,.50,.23],
 'RT':[.39,.28,.55,.19,.21,.60,.14,.28,.18,.50]}
DIMS = list(LG.keys())
lam_g = np.concatenate([LG[k] for k in DIMS])          # (80,)
lam_s = np.concatenate([LS[k] for k in DIMS])          # (80,)
res_sd = np.sqrt(np.clip(1 - lam_g**2 - lam_s**2, 0.05, None))
dim_idx = np.repeat(np.arange(8), 10)                  # dimensión de cada ítem

# Umbrales Likert comunes (calibrados para medias de escala ~37-40/50, como Tabla 7)
THR = np.array([-2.1, -1.3, -0.55, 0.55])

rng_global = np.random.default_rng(12345)

def gen_population(n, beta1, beta0, rng):
    """Genera n sujetos: 8 PD + grit + y. Devuelve (X(9), y, g)."""
    g = rng.standard_normal(n)
    s = rng.standard_normal((n, 8))
    e = rng.standard_normal((n, 80)) * res_sd
    lat = np.outer(g, lam_g) + s[:, dim_idx] * lam_s + e          # (n,80)
    likert = 1 + np.searchsorted(THR, lat.ravel()).reshape(n, 80) # discretiza 1-5
    PD = likert.reshape(n, 8, 10).sum(axis=2).astype(float)       # (n,8)
    # grit: carga en g y en el específico de AM (índice 4), reescalado a 1-5
    grit_lat = 0.45*g + 0.45*s[:,4] + rng.standard_normal(n)*np.sqrt(1-0.45**2-0.45**2)
    grit = np.clip(np.round((grit_lat*0.55 + 4.0)*2)/2, 1, 5)
    X = np.column_stack([PD, grit])
    y = (rng.random(n) < 1/(1+np.exp(-(beta0 + beta1*g)))).astype(int)
    return X, y, g

def calibrate_beta(d_target, base_rate, tau, rng, N=200_000):
    """Calibra beta1 para d(g|y) objetivo y beta0 para tasa base, en la POBLACIÓN SELECCIONADA."""
    if d_target == 0:
        b1 = 0.0
    else:
        lo, hi = 0.01, 4.0
        for _ in range(28):
            b1 = (lo+hi)/2
            X, y, g = gen_population(N, b1, 0.0, rng)
            sel = np.argsort(-X[:,:8].sum(1))[:int(N*tau)] if tau<1 else np.arange(N)
            ys, gs = y[sel], g[sel]
            if ys.sum() in (0,len(ys)): dd=0
            else:
                sp = np.sqrt(((ys==1).sum()-1)*gs[ys==1].var(ddof=1)+((ys==0).sum()-1)*gs[ys==0].var(ddof=1))
                sp = np.sqrt((gs[ys==1].var(ddof=1)*((ys==1).sum()-1)+gs[ys==0].var(ddof=1)*((ys==0).sum()-1))/(len(ys)-2))
                dd = (gs[ys==1].mean()-gs[ys==0].mean())/sp
            if dd < d_target: lo=b1
            else: hi=b1
        b1=(lo+hi)/2
    # beta0 para tasa base en seleccionados
    lo,hi=-3,3
    for _ in range(28):
        b0=(lo+hi)/2
        X,y,g = gen_population(60_000, b1, b0, rng)
        sel = np.argsort(-X[:,:8].sum(1))[:int(60_000*tau)] if tau<1 else np.arange(60_000)
        if y[sel].mean() < base_rate: lo=b0
        else: hi=b0
    return b1,(lo+hi)/2

def one_replica(n, tau, b1, b0, rng):
    pool = int(np.ceil(n/tau)) if tau<1 else n
    X,y,g = gen_population(pool, b1, b0, rng)
    if tau<1:
        sel = np.argsort(-X[:,:8].sum(1))[:n]
        X,y = X[sel], y[sel]
    if y.sum()<3 or (1-y).sum()<3: return np.nan
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    try:
        cv = StratifiedKFold(5, shuffle=True, random_state=int(rng.integers(1e6)))
        pr = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba')[:,1]
        return roc_auc_score(y, pr)
    except Exception:
        return np.nan

# ---------- calibración del generador (población general) ----------
rngc = np.random.default_rng(777)
Xp, yp, gp = gen_population(100_000, 0, 0, rngc)
pop_means = Xp[:,:8].mean(0); pop_sds = Xp[:,:8].std(0)
P95 = np.percentile(Xp[:,:8], 95, axis=0)

# saturación bajo tau=0.2 con n grande
Xs, ys, _ = gen_population(100_000, 0, 0, rngc)
sel = np.argsort(-Xs[:,:8].sum(1))[:20_000]
sat = [float((Xs[sel,k]>=P95[k]).mean()*100) for k in range(8)]
calib = dict(pop_means=[round(float(m),1) for m in pop_means],
             pop_sds=[round(float(s),2) for s in pop_sds],
             saturacion_tau02=[round(s,1) for s in sat],
             intercorr_media_pop=round(float(np.corrcoef(Xp[:,:8].T)[np.triu_indices(8,1)].mean()),3))
print('CALIBRACION:', json.dumps(calib))

# ---------- rejilla ----------
NS  = [41, 80, 150, 300]
DS  = [0.0, 0.35, 0.5]
TAUS= [1.0, 0.2]
NREP = 200
BASE = 0.634

# piloto de tiempo
t0=time.time(); _=[one_replica(41,0.2,0.5,0.5,np.random.default_rng(k)) for k in range(5)]
print(f'tiempo/replica n=41 tau=0.2: {(time.time()-t0)/5:.2f}s')

results={}
betas={}
for tau in TAUS:
    for dd in DS:
        b1,b0 = calibrate_beta(dd, BASE, tau, np.random.default_rng(int(1000*tau+100*dd)))
        betas[f'tau{tau}_d{dd}']=(round(b1,3),round(b0,3))
for tau in TAUS:
    for dd in DS:
        b1,b0 = betas[f'tau{tau}_d{dd}']
        for n in NS:
            rng = np.random.default_rng(int(7919*tau*100 + 101*dd*100 + n))
            aucs = np.array([one_replica(n, tau, b1, b0, np.random.default_rng(rng.integers(1e9))) for _ in range(NREP)])
            aucs = aucs[~np.isnan(aucs)]
            results[f'n{n}_tau{tau}_d{dd}'] = aucs.tolist()
        print(f'hecho tau={tau} d={dd}', flush=True)

from pathlib import Path
_out = Path(__file__).resolve().parents[1] / 'resultados'
_out.mkdir(exist_ok=True)
json.dump(dict(results=results, betas=betas, calib=calib), open(_out / 'sim_results.json', 'w'))
print('SIMULACION COMPLETA. Tiempo total:', round(time.time()-t0,1),'s')
