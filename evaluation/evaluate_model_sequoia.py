import pandas as pd
import numpy as np

import pickle as pl
import os

from sklearn.metrics import mean_squared_error
from statsmodels.stats.multitest import fdrcorrection
from scipy import stats

if __name__=='__main__':
    
    model_dir = 'results'
    folds = 5
    # cancers = ['brca', 'coad', 'luad']    
    cancer = 'brca'

    save_path = os.path.join(model_dir, 'sequoia-pub')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

   
    print(cancer.upper())
    with open(os.path.join(f'./rna_ckpts/sequoia-pub/output/{cancer.upper()}/run_train_sequoia', 'test_results.pkl'), 'rb') as f:
        test_res = pl.load(f)
    save_path = os.path.join(save_path, cancer.upper())
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    real = []
    pred = []
    wsi = []
    genes = test_res['genes']

    for k in range(folds):
        data = test_res[f'split_{k}']
        n_sample = len(data['preds'][:, 0])
        wsi.extend(data['wsi_file_name'])
        real.append(pd.DataFrame(data['real'], index = data['wsi_file_name'], columns = genes))
        pred.append(pd.DataFrame(data['preds'], index = data['wsi_file_name'], columns = genes))
               
    df_real = pd.concat(real)
    df_pred = pd.concat(pred)
    assert np.all(df_real.index == df_pred.index)

    pred_r = []
    test_p = []
    pearson_p = []
    rmse_pred = []
    rmse_quantile_norm = []
    rmse_mean_norm = []
    valid_genes = []

    for i, gene in enumerate(genes):
        real = df_real.loc[:, gene]
        pred = df_pred.loc[:, gene]
        
        if len(set(pred)) == 1 or len(set(real)) ==1:
            xy, xy, yz = 0, 0, 0
            p1, p2, p3, p = 1, 1, 1, 1
        else:
            xy, p1 = stats.pearsonr(real, pred)

        pred_r.append(xy)
        pearson_p.append(p1)

        # RMSE test
        rmse_p = mean_squared_error(real, pred, squared=False)
        rmse_q = rmse_p / (np.quantile(real, 0.75) - np.quantile(real, 0.25) + 1e-5)
        rmse_m = rmse_p / np.mean(real)

        rmse_pred.append(rmse_p)
        rmse_quantile_norm.append(rmse_q)
        rmse_mean_norm.append(rmse_m)
        valid_genes.append(gene)
    
    combine_res = pd.DataFrame({'pred_real_r': pred_r,\
                            'pearson_p': pearson_p,\
                            'rmse_pred': rmse_pred, \
                            'rmse_quantile_norm': rmse_quantile_norm,
                            'rmse_mean_norm': rmse_mean_norm}, 
                            index=valid_genes)

    combine_res = combine_res.sort_values('pred_real_r', ascending = False)

    combine_res['pred_real_r'] = combine_res['pred_real_r'].fillna(0)

    # Correct pearson p values
    combine_res['pearson_p'] = combine_res['pearson_p'].fillna(1)
    _, fdr_pearson_p = fdrcorrection(combine_res['pearson_p'])
    combine_res['fdr_pearson_p'] = fdr_pearson_p
    

    combine_res['cancer'] = cancer.upper()
    all_res = combine_res
    all_res = all_res.sort_values('pred_real_r', ascending = False)
    all_res_pcc = all_res.head(1000)

    top_1000_pred_real_r = all_res_pcc['pred_real_r'].mean()
    top_1000_rmse_mean_norm_same_genes = all_res_pcc['rmse_mean_norm'].mean()

    all_res_rmse_norm = all_res.sort_values('rmse_mean_norm', ascending = True).head(1000)
    top_1000_rmse_mean_norm_plain = all_res_rmse_norm['rmse_mean_norm'].mean()

    all_res_rmse_pred = all_res.sort_values('rmse_pred', ascending = True).head(1000)
    top_1000_rmse_pred_plain = all_res_rmse_pred['rmse_pred'].mean()
    # create a new dataframe with the above results
    report_results = pd.DataFrame({
        'top_1000_pred_real_r': top_1000_pred_real_r,
        'top_500_pred_real_r': all_res_pcc['pred_real_r'].head(500).mean(),
        'top_200_pred_real_r': all_res_pcc['pred_real_r'].head(200).mean(),
        'top_100_pred_real_r': all_res_pcc['pred_real_r'].head(100).mean(),
        'top_50_pred_real_r': all_res_pcc['pred_real_r'].head(50).mean(),
        'top_20_pred_real_r': all_res_pcc['pred_real_r'].head(20).mean(),
        'top_1000_rmse_mean_norm_plain': top_1000_rmse_mean_norm_plain,
        'top_500_rmse_mean_norm_plain': all_res_rmse_norm['rmse_mean_norm'].head(500).mean(),
        'top_200_rmse_mean_norm_plain': all_res_rmse_norm['rmse_mean_norm'].head(200).mean(),
        'top_100_rmse_mean_norm_plain': all_res_rmse_norm['rmse_mean_norm'].head(100).mean(),
        'top_50_rmse_mean_norm_plain': all_res_rmse_norm['rmse_mean_norm'].head(50).mean(),
        'top_20_rmse_mean_norm_plain': all_res_rmse_norm['rmse_mean_norm'].head(20).mean(),
        'top_1000_rmse_pred_plain': top_1000_rmse_pred_plain,
        'top_500_rmse_pred_plain': all_res_rmse_pred['rmse_pred'].head(500).mean(),
        'top_200_rmse_pred_plain': all_res_rmse_pred['rmse_pred'].head(200).mean(),
        'top_100_rmse_pred_plain': all_res_rmse_pred['rmse_pred'].head(100).mean(),
        'top_50_rmse_pred_plain': all_res_rmse_pred['rmse_pred'].head(50).mean(),
        'top_20_rmse_pred_plain': all_res_rmse_pred['rmse_pred'].head(20).mean(),
    }, index=[0])
    report_results.to_csv(os.path.join(save_path, 'report_results_all.csv'))
    sig_res = all_res[(all_res['pred_real_r'] > 0) & \
                    (all_res['pearson_p'] < 0.05)
                    ]

    all_res.to_csv(os.path.join(save_path, 'all_genes_rr.csv'))
    