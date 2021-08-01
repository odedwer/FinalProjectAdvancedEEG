import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm import tqdm

# %% read data per subject and unify them
subjects = [331, 332, 333, 334, 335, 336,337,339,342,344,345,346,347,349]
all_dfs = pd.concat([pd.read_csv(f"S{s}_df_with_neutral.csv") for s in subjects])
print(all_dfs.groupby('congruent')['is_correct'].mean())
print(all_dfs.groupby('congruent')['RT'].mean())
print(all_dfs.groupby(['congruent', 'is_correct', "subject"])['mean_pip'].mean())
print(all_dfs.groupby(['congruent', 'is_correct'])['mean_pip'].std())
print(all_dfs.groupby('congruent')['mean_pip'].mean())
plt.figure()
print(all_dfs.groupby('subject')['mean_pip'].hist())

# %%
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from  sklearn.decomposition import PCA
from scipy import stats

# reg_model = linear_model.Lasso(alpha=0.01,max_iter=10000)
# reg_model = linear_model.RidgeCV(alphas=np.logspace(-5,2,15))
num_perm = 500
curr_df = all_dfs[all_dfs['subject'] == 336]
reg_model = linear_model.Lasso(alpha=0.001,max_iter=2000)
X = zscore((np.array(np.array(curr_df)[:, 12:], dtype=float)), axis=0)
# pca = PCA(n_components=3, svd_solver='full')
# pca.fit(X)
# pca.explained_variance_ratio_
# X=zscore(pca.fit_transform(X),axis=0)
y = np.array(curr_df['mean_pip'], dtype=float)
y_perm = np.zeros_like(y)
cor_val_true = np.zeros(num_perm)
cor_val_perm = np.zeros(num_perm)
num_test = y.size // 5
for i in tqdm(range(num_perm)):
    perm_test = np.random.choice(np.arange(y.size), num_test, replace=False)
    idx_vec = np.zeros_like(y, dtype=bool)
    idx_vec[perm_test] = True
    y_perm = np.random.choice(y, y_perm.size, replace=False)
    train_x, test_x, train_y, test_y, perm_train_y, perm_test_y = X[~idx_vec, :], X[idx_vec, :], y[~idx_vec], y[
        idx_vec], y_perm[~idx_vec], y_perm[idx_vec]
    reg_model.fit(train_x, train_y)
    y_pred = reg_model.predict(test_x)
    reg_model.fit(train_x, perm_train_y)
    y_perm_pred = reg_model.predict(test_x)
    # plt.scatter(test_y, y_pred)
    cor_val_true[i] = stats.pearsonr(test_y, y_pred)[0]
    cor_val_perm[i] = stats.pearsonr(perm_test_y, y_perm_pred)[0]
# %%
fig, ax = plt.subplots()
ax.hist(cor_val_true, alpha=0.7, color="darkblue", label="True")
ax.hist(cor_val_perm, alpha=0.7, color="darkred", label="Permutation")
ax.legend()
np.mean(cor_val_true.mean() < cor_val_perm)
# %%
plt.scatter(train_y, train_x[:, -1])

alphas = np.logspace(10, 100, 13)
clf = LassoCV(eps=1e-5, n_alphas=200, alphas=alphas, max_iter=10000, cv=10)
clf = clf.fit(zscore(np.log(X), axis=0), y)
y_pred = clf.predict(X)
plt.scatter(y, y_pred)
mean_squared_error(y, y_pred)
r2_score(y, y_pred)
# %%
all_dfs["prime"] = all_dfs["prime"].astype("category")
all_dfs["cue"] = all_dfs["cue"].astype("category")
all_dfs["choice"] = all_dfs["choice"].astype("category")
all_dfs["congruent"] = all_dfs["congruent"].astype("category")
all_dfs["is_correct"] = all_dfs["is_correct"].astype("category")
all_dfs["subject"] = all_dfs["subject"].astype("category")
all_dfs.to_csv("all_dfs_pip.csv")

# md = smf.mixedlm("RT~median_pip*congruent",data=all_dfs, groups=all_dfs["subject"])
# mdf=md.fit(method=["lbfgs"])
# print(mdf.summary())
#
# #%% sanity check for congruency effect
# md_noint = smf.mixedlm("RT~median_pip+congruent",data=all_dfs, groups=all_dfs["subject"])
# mdf_noint=md_noint.fit(method=["lbfgs"])
# print(mdf_noint.summary())
#
# #%% residuals
# plt.scatter(md['RT'] - md.resid, md.resid, alpha = 0.5)
# plt.title("Residual vs. Fitted in Python")
# plt.xlabel("Fitted Values")
# plt.ylabel("Residuals")
# # plt.savefig('python_plot.png',dpi=300)
# plt.show()
#
# md.predict('RT')
