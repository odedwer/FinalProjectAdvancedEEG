import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %% read data per subject and unify them
subjects = [331, 332, 333, 334, 335, 336,337,339,342,344,345,346,347,349]
all_dfs = pd.concat([pd.read_csv(f"S{s}_df.csv") for s in subjects])
print(all_dfs.groupby('congruent')['is_correct'].mean())
print(all_dfs.groupby('congruent')['RT'].mean())
print(all_dfs.groupby(['congruent', 'is_correct',"subject"])['mean_pip'].mean())
print(all_dfs.groupby(['congruent', 'is_correct'])['mean_pip'].std())
print(all_dfs.groupby('congruent')['mean_pip'].mean())
plt.figure()
print(all_dfs.groupby('subject')['mean_pip'].hist())

#%%
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