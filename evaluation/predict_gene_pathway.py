# read json pathway file for all genes
import json
import pandas as pd
from pathlib import Path
def read_pathway_file(pathway_file):
    """
    Find all pathways that contain a specific gene
    """
    with open(pathway_file, 'r') as f:
        data = json.load(f)
    
    gene_in_pathway = []
    pathway_id = []
    pathway_label = []
    i = 0
    groups_data = data['overrepresentation']['group']
    print(len(groups_data))
    for result in groups_data:
        print(len(result))
        result_list = result['result']
        if isinstance(result_list, list):
            for sub_result in result_list:
                genes = sub_result['input_list']['mapped_id_list']['mapped_id']
                gene_in_pathway.append(genes)
                # ids = sub_result['term']['id']
                # pathway_id.append(ids)
                labels = sub_result['term']['label']
                pathway_label.append(labels)
        elif isinstance(result_list, dict):
            genes = result_list['input_list']['mapped_id_list']['mapped_id']
            gene_in_pathway.append(genes)
            labels = result_list['term']['label']
            pathway_label.append(labels)
        else:
            exit('error')

    return gene_in_pathway, pathway_label

if __name__ == '__main__':
    all_genes_path = './results/RNAFM_BRCA/all_genes.csv'
    gene_in_pathway, pathway_label = read_pathway_file('./examples/analysis.json')
    
    all_genes = pd.read_csv(all_genes_path)
    print(all_genes.head())
    df = pd.DataFrame(columns=['pathway_name', 'pathway_PCC', 'pathway_rmse'])
    for i, genes in enumerate(gene_in_pathway):
        # Convert genes to list if it's a string
        if isinstance(genes, str):
            # If it's a single gene name, make it a list
            genes_list = [genes]
        elif isinstance(genes, list):
            # If it's already a list, use it as is
            genes_list = genes
        else:
            # Skip if it's neither string nor list
            exit(f"Warning: Unexpected data type for genes: {type(genes)}")
            
        pathway_name = pathway_label[i]
        # find pred_real_r for genes in all_genes
        pathway_PCC = all_genes[all_genes['Unnamed: 0'].isin(genes_list)]['pred_real_r'].mean()
        pathway_rmse = all_genes[all_genes['Unnamed: 0'].isin(genes_list)]['rmse_mean_norm'].mean()
        df = df.append({'pathway_name': pathway_name, 'pathway_PCC': pathway_PCC, 'pathway_rmse': pathway_rmse}, ignore_index=True)

    df.sort_values(by='pathway_PCC', ascending=False, inplace=True)
    df.to_csv(Path(all_genes_path).parent / 'pathway_PCC_rmse.csv', index=False)
