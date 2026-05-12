import pandas as pd
import numpy as np
import argparse
import pickle as pl
import os

from sklearn.metrics import mean_squared_error
from statsmodels.stats.multitest import fdrcorrection
from scipy import stats


def result_generation(cancer, exp_name, save_path, folds, num_sampling_steps):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
   
    print(cancer.upper())
    file_path = f'./rna_ckpts/output/{cancer.upper()}/{exp_name}/'
    
    save_path = os.path.join(save_path, cancer.upper())
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    real = []
    pred = []
    wsi = []
    pred_ema = []
    

    for k in range(folds):
        with open(os.path.join(file_path, f'split_{k}', f'test_results_{k}_cfg_2.0.pkl'), 'rb') as f:
            test_res = pl.load(f)
        genes = test_res['genes']
        data = test_res[f'split_{k}']
        n_sample = len(data['preds'][:, 0])
        wsi.extend(data['wsi_file_name'])
        real.append(pd.DataFrame(data['real'], index = data['wsi_file_name'], columns = genes))
        pred.append(pd.DataFrame(data['preds'], index = data['wsi_file_name'], columns = genes))
        pred_ema.append(pd.DataFrame(data['preds_ema'], index = data['wsi_file_name'], columns = genes))
            
    df_real = pd.concat(real)
    df_pred = pd.concat(pred)
    df_pred_ema = pd.concat(pred_ema)

    #Make sure the index (samples) are identical in all the dataframes
    assert np.all(df_real.index == df_pred.index)
    assert np.all(df_real.index == df_pred_ema.index)
    pred_r = []
    pearson_p = []
    rmse_pred = []
    rmse_quantile_norm = []
    rmse_mean_norm = []
    valid_genes = []

    pred_r_ema = []
    test_p_ema = []
    pearson_p_ema = []
    rmse_pred_ema = []
    rmse_quantile_norm_ema = []
    rmse_mean_norm_ema = []

    for i, gene in enumerate(genes):
        real = df_real.loc[:, gene]
        pred = df_pred.loc[:, gene]
        pred_ema = df_pred_ema.loc[:, gene]
        
        xy, p1 = stats.pearsonr(real, pred)
        xy_ema, p1_ema = stats.pearsonr(real, pred_ema)

        pred_r.append(xy)
        pearson_p.append(p1)
        pred_r_ema.append(xy_ema)
        pearson_p_ema.append(p1_ema)
        # RMSE test
        rmse_p = mean_squared_error(real, pred, squared=False)
        rmse_q = rmse_p / (np.quantile(real, 0.75) - np.quantile(real, 0.25) + 1e-5)
        rmse_m = rmse_p / np.mean(real)
        rmse_p_ema = mean_squared_error(real, pred_ema, squared=False)
        rmse_q_ema = rmse_p_ema / (np.quantile(real, 0.75) - np.quantile(real, 0.25) + 1e-5)
        rmse_m_ema = rmse_p_ema / np.mean(real)
        rmse_pred.append(rmse_p)
        rmse_quantile_norm.append(rmse_q)
        rmse_mean_norm.append(rmse_m)
        valid_genes.append(gene)
        rmse_pred_ema.append(rmse_p_ema)
        rmse_quantile_norm_ema.append(rmse_q_ema)
        rmse_mean_norm_ema.append(rmse_m_ema)

    combine_res = pd.DataFrame({'pred_real_r': pred_r,\
                            'pearson_p': pearson_p,\
                            'rmse_pred': rmse_pred, \
                            'rmse_quantile_norm': rmse_quantile_norm, \
                            'rmse_mean_norm': rmse_mean_norm}, 
                            index=valid_genes)
    combine_res_ema = pd.DataFrame({'pred_real_r_ema': pred_r_ema, \
                                    'pearson_p_ema': pearson_p_ema, \
                                    'rmse_pred_ema': rmse_pred_ema, \
                                    'rmse_quantile_norm_ema': rmse_quantile_norm_ema, \
                                    'rmse_mean_norm_ema': rmse_mean_norm_ema}, index=valid_genes)

    combine_res = combine_res.sort_values('pred_real_r', ascending = False)
    combine_res_ema = combine_res_ema.sort_values('pred_real_r_ema', ascending = False)

    # In case of constant values, replace correlation coefficient to 0
    combine_res['pred_real_r'] = combine_res['pred_real_r'].fillna(0)
    combine_res_ema['pred_real_r_ema'] = combine_res_ema['pred_real_r_ema'].fillna(0)
    # Correct pearson p values
    combine_res['pearson_p'] = combine_res['pearson_p'].fillna(1)
    _, fdr_pearson_p = fdrcorrection(combine_res['pearson_p'])
    combine_res['fdr_pearson_p'] = fdr_pearson_p
    combine_res_ema['pearson_p_ema'] = combine_res_ema['pearson_p_ema'].fillna(1)
    _, fdr_pearson_p_ema = fdrcorrection(combine_res_ema['pearson_p_ema'])
    combine_res_ema['fdr_pearson_p_ema'] = fdr_pearson_p_ema
    
    combine_res['cancer'] = cancer.upper()
    combine_res_ema['cancer'] = cancer.upper()
    all_res = combine_res
    all_res = all_res.sort_values('pred_real_r', ascending = False)
    all_res_pcc = all_res.head(1000)
    all_res_pcc.to_csv(os.path.join(save_path, 'all_genes_sorted_by_pred_real_r.csv'))
    all_res_ema = combine_res_ema
    all_res_ema = all_res_ema.sort_values('pred_real_r_ema', ascending = False)
    all_res_pcc_ema = all_res_ema.head(1000)
    all_res_pcc_ema.to_csv(os.path.join(save_path, 'all_genes_sorted_by_pred_real_r_ema.csv'))

    top_1000_pred_real_r = all_res_pcc['pred_real_r'].mean()
    top_1000_pred_real_r_ema = all_res_pcc_ema['pred_real_r_ema'].mean()
    all_res_rmse = all_res.sort_values('rmse_mean_norm', ascending = True).head(1000)
    all_res_rmse.to_csv(os.path.join(save_path, 'all_genes_sorted_by_rmse_mean_norm.csv'))
    all_res_rmse_ema = all_res_ema.sort_values('rmse_mean_norm_ema', ascending = True).head(1000)
    all_res_rmse_ema.to_csv(os.path.join(save_path, 'all_genes_sorted_by_rmse_mean_norm_ema.csv'))
    
    top_1000_rmse_mean_norm_plain = all_res_rmse['rmse_mean_norm'].mean()
    top_1000_rmse_mean_norm_plain_ema = all_res_rmse_ema['rmse_mean_norm_ema'].mean()

    all_res_rmse_pred = all_res.sort_values('rmse_pred', ascending = True).head(1000)
    all_res_rmse_pred.to_csv(os.path.join(save_path, 'all_genes_sorted_by_rmse_pred.csv'))
    top_1000_rmse_pred_plain = all_res_rmse_pred['rmse_pred'].mean()
    all_res_rmse_pred_ema = all_res_ema.sort_values('rmse_pred_ema', ascending = True).head(1000)
    all_res_rmse_pred_ema.to_csv(os.path.join(save_path, 'all_genes_sorted_by_rmse_pred_ema.csv'))
    top_1000_rmse_pred_plain_ema = all_res_rmse_pred_ema['rmse_pred_ema'].mean()
    # create a new dataframe with the above results
    report_results = pd.DataFrame({
        'top_1000_pred_real_r': top_1000_pred_real_r,
        'top_500_pred_real_r': all_res_pcc['pred_real_r'].head(500).mean(),
        'top_200_pred_real_r': all_res_pcc['pred_real_r'].head(200).mean(),
        'top_100_pred_real_r': all_res_pcc['pred_real_r'].head(100).mean(),
        'top_50_pred_real_r': all_res_pcc['pred_real_r'].head(50).mean(),
        'top_20_pred_real_r': all_res_pcc['pred_real_r'].head(20).mean(),
        'top_1000_rmse_mean_norm_plain': top_1000_rmse_mean_norm_plain,
        'top_500_rmse_mean_norm_plain': all_res_rmse['rmse_mean_norm'].head(500).mean(),
        'top_200_rmse_mean_norm_plain': all_res_rmse['rmse_mean_norm'].head(200).mean(),
        'top_100_rmse_mean_norm_plain': all_res_rmse['rmse_mean_norm'].head(100).mean(),
        'top_50_rmse_mean_norm_plain': all_res_rmse['rmse_mean_norm'].head(50).mean(),
        'top_20_rmse_mean_norm_plain': all_res_rmse['rmse_mean_norm'].head(20).mean(),
        'top_1000_pred_real_r_ema': top_1000_pred_real_r_ema,
        'top_500_pred_real_r_ema': all_res_pcc_ema['pred_real_r_ema'].head(500).mean(),
        'top_200_pred_real_r_ema': all_res_pcc_ema['pred_real_r_ema'].head(200).mean(),
        'top_100_pred_real_r_ema': all_res_pcc_ema['pred_real_r_ema'].head(100).mean(),
        'top_50_pred_real_r_ema': all_res_pcc_ema['pred_real_r_ema'].head(50).mean(),
        'top_20_pred_real_r_ema': all_res_pcc_ema['pred_real_r_ema'].head(20).mean(),
        'top_1000_rmse_mean_norm_plain_ema': top_1000_rmse_mean_norm_plain_ema,
        'top_500_rmse_mean_norm_plain_ema': all_res_rmse_ema['rmse_mean_norm_ema'].head(500).mean(),
        'top_200_rmse_mean_norm_plain_ema': all_res_rmse_ema['rmse_mean_norm_ema'].head(200).mean(),
        'top_100_rmse_mean_norm_plain_ema': all_res_rmse_ema['rmse_mean_norm_ema'].head(100).mean(),
        'top_50_rmse_mean_norm_plain_ema': all_res_rmse_ema['rmse_mean_norm_ema'].head(50).mean(),
        'top_20_rmse_mean_norm_plain_ema': all_res_rmse_ema['rmse_mean_norm_ema'].head(20).mean(),
        'top_1000_rmse_pred_plain': top_1000_rmse_pred_plain,
        'top_500_rmse_pred_plain': all_res_rmse_pred['rmse_pred'].head(500).mean(),
        'top_200_rmse_pred_plain': all_res_rmse_pred['rmse_pred'].head(200).mean(),
        'top_100_rmse_pred_plain': all_res_rmse_pred['rmse_pred'].head(100).mean(),
        'top_50_rmse_pred_plain': all_res_rmse_pred['rmse_pred'].head(50).mean(),
        'top_20_rmse_pred_plain': all_res_rmse_pred['rmse_pred'].head(20).mean(),
        'top_1000_rmse_pred_plain_ema': top_1000_rmse_pred_plain_ema,
        'top_500_rmse_pred_plain_ema': all_res_rmse_pred_ema['rmse_pred_ema'].head(500).mean(),
        'top_200_rmse_pred_plain_ema': all_res_rmse_pred_ema['rmse_pred_ema'].head(200).mean(),
        'top_100_rmse_pred_plain_ema': all_res_rmse_pred_ema['rmse_pred_ema'].head(100).mean(),
        'top_50_rmse_pred_plain_ema': all_res_rmse_pred_ema['rmse_pred_ema'].head(50).mean(),
        'top_20_rmse_pred_plain_ema': all_res_rmse_pred_ema['rmse_pred_ema'].head(20).mean(),
    }, index=[0])
    report_results.to_csv(os.path.join(save_path, 'report_results_all.csv'))

    all_res.to_csv(os.path.join(save_path, 'all_genes.csv'))
    all_res_ema.to_csv(os.path.join(save_path, 'all_genes_ema.csv'))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cancer', type=str, default='COAD')
    parser.add_argument('--model_dir', type=str, default='results')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--exp_name', type=str, default='diff_regene_filtered_normalized_new_1500_uniform_pathway_newloss_filtered_refine_nonormloss_nogate_bg_noval_rampema_COAD_nograph')
    parser.add_argument('--num_sampling_steps', type=int, default=20)
    args = parser.parse_args()
    save_path = os.path.join(args.model_dir, args.exp_name)
    result_generation(args.cancer, args.exp_name, save_path, args.folds, 20)