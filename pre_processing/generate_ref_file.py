import os
import glob
import argparse
import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm

def find_wsi_files(slides_dir: str) -> List[str]:
    """Recursively find slide files.

    Currently searches for .svs files.
    """
    results = sorted(glob.glob(os.path.join(slides_dir, "**/*.svs"), recursive=True))
    # results = sorted(glob.glob(os.path.join(slides_dir, "*.svs")))
    # create a dataframe with index patient_id and column wsi_file_name
    df = pd.DataFrame(results, columns=['wsi_file_name'])
    df['patient_id'] = df['wsi_file_name'].apply(lambda x: x.split('/')[-1][:12])
    print(df.head())
    return df


def find_rna_files(rna_dir: str, gene_list: str) -> Dict[str, str]:
    """Index RNA files by inferred patient prefix.

    Heuristic: any file whose stem starts with a TCGA patient prefix.
    Supports common text formats (csv/tsv/txt). If multiple files map to the
    same patient, the first in sorted order is kept.
    """
    # make a dataframe with column of gene names
    results = sorted(glob.glob(os.path.join(rna_dir, '**/*.tsv'), recursive=True))
    print(f"Number of RNA files: {len(results)}")
    df = pd.DataFrame(results, columns=['rna_file_name'])
    df['sample_id'] = df['rna_file_name'].apply(lambda x: x.split('/')[-1].split('_')[1].split('.')[0][-3:])
    df['patient_id'] = df['rna_file_name'].apply(lambda x: x.split('/')[-1].split('_')[1].split('.')[0][:12])
    # check if sample_id is other than 01A and still has 01A for other samples, delete the row
    df_other = df[df['sample_id'] != '01A']
    print(f"Number of RNA files other than 01A: {df_other.shape[0]}")
    df_01A = df[df['sample_id'] == '01A']
    print(f"Number of RNA files 01A: {df_01A.shape[0]}")
    if df_other.shape[0] > 0:
        #filter out the rows in df_other that have 01A in the corresponding sample_id in df_01A
        df_other = df_other[~df_other['patient_id'].isin(df_01A['patient_id'])]
    # print(df_01A.shape)
    # only keep the first occurrence of rows in df_01A with duplicate patient_id
    df_01A = df_01A.drop_duplicates(subset=['patient_id'])
    print(f"Number of RNA files 01A after dropping duplicates: {df_01A.shape[0]}")
    # print(df_01A[df_01A['patient_id'].duplicated()]['rna_file_name'])
    df_other = df_other.drop_duplicates(subset=['patient_id'])
    print(f"Number of RNA files other than 01A after dropping duplicates: {df_other.shape[0]}")

    # print(df_other.shape)
    # print(df_other.shape)
    # print(df_other)
    df_all = pd.concat([df_01A, df_other])
    print(f"Number of RNA files after concatenating: {df_all.shape[0]}")
    # reset the index
    df_all = df_all.reset_index(drop=True)

    # reading rna_file_name files from the df_all
    gene_ls_df = pd.read_csv(gene_list)
    genes = gene_ls_df.iloc[:, 0].astype(str).tolist()
    # print(genes)
    # insert all gene in genes listto the df_all dataframe as each column
    # rna_value = df_all.copy()
    # for gene in genes:
    #     rna_value[gene] = float('nan')
    # insert patient_id to the rna_value dataframe
    rna_value = pd.DataFrame(index=df_all['rna_file_name'], columns=[f'rna_{gene}' for gene in genes])
    rna_value['patient_id'] = df_all['patient_id']
    #insert patient_id to the rna_value dataframe
    # get fpkm_uq_unstranded value for each gene in genes 
    print(rna_value.shape)
    print(rna_value)
    df_all.to_csv('examples/df_all.csv', index=True)
    for i in tqdm(range(len(df_all['rna_file_name']))): 
        rna_file = df_all['rna_file_name'][i]
        print(rna_file)
        rna_value.loc[rna_file, 'patient_id'] = df_all[df_all['rna_file_name'] == rna_file]['patient_id'].iloc[0]
        rna_df = pd.read_csv(df_all['rna_file_name'][i], 
                sep='\t', 
                skiprows=[0, 2, 3, 4, 5],  # Skip lines 1, 3, 4, 5, 6 (0-indexed)
                comment='#')  # Also skip any lines starting with #
        for gene in genes:
            # print(rna_file)
            # print(gene)
            if gene in rna_df['gene_name'].values:
                rna_value.loc[rna_file, f'rna_{gene}'] = rna_df[rna_df['gene_name'] == gene]['fpkm_uq_unstranded'].iloc[0]
                # print(rna_value)
            else:
                rna_value.loc[rna_file, f'rna_{gene}'] = float('nan')
                print('gene not found in sample')
        rna_value.to_csv(f'examples/rna_value_temp.csv', index=True)
    #save the rna_value to a csv file
    rna_value.to_csv('examples/rna_value.csv', index=False)

    return rna_value


def main():
    parser = argparse.ArgumentParser(description='Generate ref_file.csv by pairing WSI slides with RNA files using TCGA patient prefix')
    parser.add_argument('--slides_dir', type=str, default='/srv2/yson2999/bulk_rna/TCGA_slides/TCGA_COAD', help='Directory containing WSI slides (.svs)')
    parser.add_argument('--rna_dir', type=str, default='/srv2/yson2999/bulk_rna/TCGA_RNA/TCGA_COAD', help='Directory containing RNA files (csv/tsv/txt)')
    parser.add_argument('--gene_list', type=str, default='examples/gene_list.csv', help='CSV file listing genes to include')
    parser.add_argument('--output', type=str, default='examples/reference_TCGA_COAD.csv', help='Output CSV path to write')
    args = parser.parse_args()

    wsis = find_wsi_files(args.slides_dir)
    print(wsis.head())
    print(f"Number of WSI files: {wsis.shape[0]}")
    rnas = find_rna_files(args.rna_dir, args.gene_list)
    # print(rnas.shape)
    # print(wsis)
    # print(rnas)
    rnas.to_csv('examples/rna_df_TCGA_COAD.csv', index=True)
    # merge the wsis and rnas with the patient_id
    # rnas = pd.read_csv('examples/rna_df_TCGA_COAD.csv')
    print(f"Number of RNA files: {len(rnas['patient_id'])}")
    print(f"Number of unique RNA files: {len(set(rnas['patient_id']))}")

    # merge wsis with common column patient_id from rnas with shape of (wsis.shape[0], wsis.shape[1]+rnas.shape[1]-1)
    print(wsis.shape)
    print(f"Number of WSI files: {wsis.shape[0]}")
    df = pd.merge(wsis, rnas, on='patient_id', how='left')
    print(f"Number of merged files: {df.shape[0]}")
    # print(df)
    df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()