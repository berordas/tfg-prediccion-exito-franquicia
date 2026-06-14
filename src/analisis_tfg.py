import pandas as pd, numpy as np, json
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, GridSearchCV, cross_val_predict
from sklearn.metrics import roc_auc_score
from statsmodels.stats.power import TTestIndPower
import warnings; warnings.filterwarnings("ignore")

rng = np.random.default_rng(20260610)

# Rutas locales (única adaptación respecto al script del director: E/S)
from pathlib import Path
DATA = Path(__file__).resolve().parents[1] / 'data' / 'dataset_franquicias_seudonimizado.xlsx'
OUT = Path(__file__).resolve().parents[1] / 'resultados'
OUT.mkdir(exist_ok=True)

df = pd.read_excel(DATA)
d = df.dropna(subset=['exito']).copy()
dims = ['AE','AU','IN','LI','ML','OP','TE','TR']
pdc = [f'BEPE_{x}_PD' for x in dims]
X_cols = pdc + ['ego']
res = {}

# ---------- CAP 4: correlaciones con IC y d de Cohen ----------
def r_ci(r_val, n):
    z = np.arctanh(r_val); se = 1/np.sqrt(n-3)
    return np.tanh(z-1.96*se), np.tanh(z+1.96*se)

def cohen_d(a,b):
    na,nb=len(a),len(b)
    sp=np.sqrt(((na-1)*a.std(ddof=1)**2+(nb-1)*b.std(ddof=1)**2)/(na+nb-2))
    return (a.mean()-b.mean())/sp

cap4=[]
for c in X_cols:
    r_v,p_v = stats.pointbiserialr(d['exito'], d[c])
    lo,hi = r_ci(r_v, len(d))
    a=d[d.exito==1][c]; b=d[d.exito==0][c]
    cap4.append(dict(var=c, r=round(r_v,3), p=round(p_v,3), ci=[round(lo,3),round(hi,3)], d=round(cohen_d(a,b),3)))
res['cap4_correlaciones']=cap4

# inversión de etiquetas por enseña
inv={}
for f in d.franquicia.unique():
    sub=d[d.franquicia==f]
    r_v,p_v=stats.pointbiserialr(sub['exito'], sub['BEPE_EMP_PD'])
    inv[f]=dict(n=len(sub), r=round(r_v,3), p=round(p_v,3))
res['cap4_por_ensena']=inv

# ---------- CAP 5: modelos con validación honesta ----------
X = d[X_cols].values; y = d['exito'].values
loo = LeaveOneOut()

# (a) elastic-net anidada: outer LOO, inner 5-fold
en = Pipeline([('sc',StandardScaler()),
               ('lr',LogisticRegression(penalty='elasticnet', solver='saga', max_iter=5000))])
grid = {'lr__C':[0.01,0.1,1.0], 'lr__l1_ratio':[0.0,0.5,1.0]}
gs = GridSearchCV(en, grid, cv=5, scoring='roc_auc')
proba_en = cross_val_predict(gs, X, y, cv=loo, method='predict_proba')[:,1]
auc_en = roc_auc_score(y, proba_en)

# resustitución elastic-net (ajustada con CV interna sobre todo)
gs.fit(X,y); auc_en_resub = roc_auc_score(y, gs.predict_proba(X)[:,1])

# (b) gradient boosting, outer LOO
gb = GradientBoostingClassifier(random_state=0)
proba_gb = cross_val_predict(gb, X, y, cv=loo, method='predict_proba')[:,1]
auc_gb = roc_auc_score(y, proba_gb)
gb.fit(X,y); auc_gb_resub = roc_auc_score(y, gb.predict_proba(X)[:,1])

# (c) ridge simple, outer LOO (pipeline de referencia, el de las permutaciones)
ridge = Pipeline([('sc',StandardScaler()),('lr',LogisticRegression(C=1.0, max_iter=2000))])
proba_rg = cross_val_predict(ridge, X, y, cv=loo, method='predict_proba')[:,1]
auc_rg = roc_auc_score(y, proba_rg)
ridge.fit(X,y); auc_rg_resub = roc_auc_score(y, ridge.predict_proba(X)[:,1])

# test de permutación (1000 perms) sobre el pipeline ridge+LOO
n_perm=1000; null_aucs=np.empty(n_perm)
for k in range(n_perm):
    yp = rng.permutation(y)
    pr = cross_val_predict(ridge, X, yp, cv=loo, method='predict_proba')[:,1]
    null_aucs[k] = roc_auc_score(yp, pr)
p_perm = (1 + np.sum(null_aucs >= auc_rg)) / (n_perm + 1)
np.save(OUT / 'null_aucs_real_n41.npy', null_aucs)

res['cap5_modelos']=dict(
    elasticnet=dict(auc_loo=round(auc_en,3), auc_resub=round(auc_en_resub,3)),
    gboost=dict(auc_loo=round(auc_gb,3), auc_resub=round(auc_gb_resub,3)),
    ridge=dict(auc_loo=round(auc_rg,3), auc_resub=round(auc_rg_resub,3)),
    permutacion=dict(p=round(p_perm,3), null_p95=round(np.percentile(null_aucs,95),3), null_mean=round(null_aucs.mean(),3)),
    tasa_base=round(y.mean(),3))

# ---------- CAP 6: techo, Thorndike, potencia ----------
# techo
techo={}
for x in dims+['EMP']:
    pc=d[f'BEPE_{x}_Pc']
    techo[x]=dict(p95=round(float((pc>=95).mean()*100),1), p99=round(float((pc==99).mean()*100),1))
res['cap6_techo']=techo
res['cap6_grit']=dict(media=round(float(d.ego.mean()),2), p_ge_4=round(float((d.ego>=4).mean()*100),1),
                      p_ge_45=round(float((d.ego>=4.5).mean()*100),1))

# comparación con normas del paper (Tabla 7, empleados) y Thorndike
norms_emp = dict(AE=(37.55,5.05), AU=(40.64,4.98), IN=(38.75,4.50), LI=(39.54,5.36),
                 ML=(39.91,4.43), OP=(38.77,5.27), TE=(32.66,6.57), TR=(36.84,5.31))
thorn=[]
for x in dims:
    c=f'BEPE_{x}_PD'
    r_obs,_=stats.pointbiserialr(d['exito'], d[c])
    s_obs = d[c].std(ddof=1); S_pop = norms_emp[x][1]
    u = S_pop/s_obs
    r_corr = r_obs*u/np.sqrt(1 + r_obs**2*(u**2-1))
    thorn.append(dict(dim=x, media_muestra=round(float(d[c].mean()),1), sd_muestra=round(float(s_obs),2),
                      media_norma=norms_emp[x][0], sd_norma=norms_emp[x][1],
                      r_obs=round(float(r_obs),3), U=round(float(u),2), r_corregida=round(float(r_corr),3)))
res['cap6_thorndike']=thorn

# potencia
pw = TTestIndPower()
n1,n2 = int((y==1).sum()), int((y==0).sum())
ratio = n2/n1
mde = pw.solve_power(effect_size=None, nobs1=n1, ratio=ratio, alpha=0.05, power=0.80, alternative='two-sided')
req={}
for dd in [0.2,0.3,0.35,0.5,0.8]:
    n_req = pw.solve_power(effect_size=dd, nobs1=None, ratio=1.0, alpha=0.05, power=0.80, alternative='two-sided')
    req[str(dd)]=int(np.ceil(n_req))
from scipy.stats import norm
res['cap6_potencia']=dict(n1=n1, n2=n2, MDE=round(float(mde),2),
                          AUC_equiv_MDE=round(float(norm.cdf(mde/np.sqrt(2))),3),
                          n_por_grupo_requerido=req,
                          AUC_d035=round(float(norm.cdf(0.35/np.sqrt(2))),3),
                          AUC_d05=round(float(norm.cdf(0.5/np.sqrt(2))),3))

# medias por grupo para tabla cap4
g=d.groupby('exito')[X_cols].mean().round(2)
res['cap4_medias']={'exito1':g.loc[1.0].to_dict(),'exito0':g.loc[0.0].to_dict()}

# intercorrelaciones (para calibración sim)
corr = d[pdc].corr().values
iu=np.triu_indices(8,1)
res['intercorr_media']=round(float(corr[iu].mean()),3)
res['grit_ML']=round(float(d['ego'].corr(d['BEPE_ML_PD'])),3)

with open(OUT / 'resultados_cap456_n41.json','w') as f:
    json.dump(res,f,indent=1,ensure_ascii=False)
print(json.dumps(res,indent=1,ensure_ascii=False))
