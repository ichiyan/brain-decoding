import pickle
import sys
import numpy as np
import sklearn.linear_model as skl
from sklearn.model_selection import RepeatedKFold 
import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]

num_latents = 39

nsd_features = np.load('data/extracted_features/subj{:02d}/nsd_vdvae_depth_features_{}l.npz'.format(sub,num_latents))
train_latents = nsd_features['train_latents']
test_latents = nsd_features['test_latents']

train_path = 'data/processed_data/subj{:02d}/nsd_train_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)
train_fmri = np.load(train_path)
test_path = 'data/processed_data/subj{:02d}/nsd_test_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)
test_fmri = np.load(test_path)

## Preprocessing fMRI

train_fmri = train_fmri/300
test_fmri = test_fmri/300


norm_mean_train = np.mean(train_fmri, axis=0)
norm_scale_train = np.std(train_fmri, axis=0, ddof=1)
train_fmri = (train_fmri - norm_mean_train) / norm_scale_train
test_fmri = (test_fmri - norm_mean_train) / norm_scale_train

print(np.mean(train_fmri),np.std(train_fmri))
print(np.mean(test_fmri),np.std(test_fmri))

print(np.max(train_fmri),np.min(train_fmri))
print(np.max(test_fmri),np.min(test_fmri))

num_voxels, num_train, num_test = train_fmri.shape[1], len(train_fmri), len(test_fmri)

## latents Features Regression
print('Training latents Feature Regression')

# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# reg = skl.RidgeCV(alphas=np.arange(10000, 100000, 10000), cv=cv, scoring='neg_mean_absolute_error', fit_intercept=True, store_cv_values=True)
reg = skl.Ridge(alpha=100000, max_iter=10000, fit_intercept=True)
reg.fit(train_fmri, train_latents)
pred_test_latent = reg.predict(test_fmri)
std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
pred_latents = std_norm_test_latent * np.std(train_latents,axis=0) + np.mean(train_latents,axis=0)
# print(reg.score(test_fmri,test_latents), 'alpha: %f' % reg.alpha_)
print(reg.score(test_fmri,test_latents))

np.save('data/predicted_features/subj{:02d}/nsd_vdvae_depth_nsdgeneral_pred_sub{}_{}l_alpha100k.npy'.format(sub,sub,num_latents),pred_latents)


datadict = {
    'weight' : reg.coef_,     
    'bias' : reg.intercept_,

}

with open('data/regression_weights/subj{:02d}/vdvae_regression_depth_{}l_weights.pkl'.format(sub,num_latents),"wb") as f:
  pickle.dump(datadict,f)