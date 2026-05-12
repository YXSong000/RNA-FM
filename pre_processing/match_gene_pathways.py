import pandas as pd
import numpy as np
import os
import sys
sys.path.append('/home/yson2999/sequoia-pub')
import json
import argparse

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pathway_file', type=str, default='/home/yson2999/sequoia-pub/examples/molecular_function_pathways.csv', help='pathway file')
    parser.add_argument('--gene_list_file', type=str, default='/home/yson2999/sequoia-pub/examples/gene_list.csv', help='gene list file')
    # parser.add_argument('--output_file', type=str, default='/home/yson2999/sequoia-pub/examples/gene_in_pathway_processed_indices.npy', help='output file')
    return parser.parse_args()

def preprocess_pathways(args):
    gene_list = pd.read_csv(args.gene_list_file)
    gene_list = gene_list['gene'].tolist()
    print(len(gene_list))
    # print(gene_list)
    df = pd.read_csv(args.pathway_file)
    df_ = df['gene_in_pathway'].apply(lambda x: x.split(','))
    print(df_)
    # get the indices of the genes in the gene list
    all_gene_indices = []
    print(len(df_))
    # create a list of 0-20819 indices
    for i, genes in enumerate(df_):
        # print(df.iloc[i]['pathway_name'])
        gene_indices = []
        # print(f'Processing pathway {i} of starting with {genes[:5]}')
        for gene in genes:
            # get all indices of the gene in gene_list (strip '.*' from the gene if it exists)
            gene_indices.extend([i for i, gene_name in enumerate(gene_list) if gene_name.split('.')[0] == gene])
        # print(gene_indices[:5])
        all_gene_indices.append(gene_indices)

    # save in json file with key as pathway_name and value as gene_indices
    all_gene_indices = {df.iloc[i]['pathway_name']: all_gene_indices[i] for i in range(len(all_gene_indices))}
    # with open('/home/yson2999/sequoia-pub/examples/all_gene_indices_1000.json', 'w') as f:
    #     json.dump(all_gene_indices, f)
    # print(all_gene_indices)
    # save the all_gene_indices to a file
    # check if all the indices are in the indices list
    # all_gene_indices = json.load(open('/home/yson2999/sequoia-pub/examples/all_gene_indices.json'))
    print(len(list(all_gene_indices.keys())))
    # print(len(list(all_gene_indices_filtered.keys())))
    # all_gene_indices = json.load(open('/home/yson2999/sequoia-pub/examples/all_gene_indices.json'))
    # print(len(list(all_gene_indices.keys())))
    # for k, v in list(all_gene_indices.items()):
    #     if len(v) <=50:
    #         del all_gene_indices[k]
    # print(len(list(all_gene_indices.keys())))
    # minus_pathways = json.load(open('/home/yson2999/sequoia-pub/examples/all_gene_indices.json'))

    # print(f'minus_pathways: {minus_pathways.keys()}')
    # del all_gene_indices['biological regulation']
    # del all_gene_indices['response to stimulus']
    # del all_gene_indices['viral process']
    # del all_gene_indices['biological process involved in interspecies interaction between organisms']
    # del all_gene_indices['locomotion']
    # del all_gene_indices['homeostatic process']
    # del all_gene_indices['maternal process involved in female pregnancy']
    # del all_gene_indices['ossification']
    # del all_gene_indices['behavior']
    # del all_gene_indices['tissue remodeling']
    
    # sorted_all_gene_indices = sorted(all_gene_indices.items(), key=lambda x: len(x[1]), reverse=True)
    # print(len(sorted_all_gene_indices))
    # for k, v in list(all_gene_indices.items()):
    #     if len(v) < 100:
    #         del all_gene_indices[k]
        
    # with open('/home/yson2999/sequoia-pub/examples/all_gene_indices_filtered_2.json', 'w') as f:
    #     json.dump(all_gene_indices, f)
    # for k, v in sorted_all_gene_indices:
    #     for k2, v2 in sorted_all_gene_indices:
    #         if k2 != k:
    #             for gene in v:
    #                 if gene in v2:
    #                     break
    #     del all_gene_indices[k]
    #     break
    # with open('/home/yson2999/sequoia-pub/examples/all_gene_indices_filtered_1420.json', 'w') as f:
    #     json.dump(all_gene_indices, f)
    indices = np.zeros(20820, dtype=bool)
    n = 0
    for pathway_name, gene_indices in all_gene_indices.items():
        # print(f'Processing pathway {pathway_name} with {len(gene_indices)} genes')
        n += 1
        # print(len(gene_indices))
        for index in gene_indices:
            indices[index] = True
    print(np.sum(indices))
    print(len(np.where(indices == False)[0]))
    # make these indices that not included in the all_gene_indices as another list
    not_included_indices = np.where(indices == False)[0]
    not_included_indices = not_included_indices.tolist()
    print(len(not_included_indices))
    # print(not_included_indices)
    # print(len(all_gene_indices['UNCLASSIFIED']))
    # print(n)
    # print(len(not_included_indices))

    all_gene_indices['UNMAPPED'] = not_included_indices
    print(len(list(all_gene_indices.keys())))
    with open('/home/yson2999/sequoia-pub/examples/all_gene_indices_molecular_function.json', 'w') as f:
        json.dump(all_gene_indices, f)
    # save a list of lists in to numpy array
    # all_gene_indices = np.array(all_gene_indices)
    # np.save(args.output_file, all_gene_indices)
    # print(f'Saved the all_gene_indices to {args.output_file}')
if __name__ == '__main__':
    args = get_parse()
    preprocess_pathways(args)