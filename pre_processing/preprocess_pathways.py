# read json pathway file for all genes
import json
import pandas as pd
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
                # print(sub_result)
                if 'mapped_id_list' in sub_result['input_list']:
                    genes = sub_result['input_list']['mapped_id_list']['mapped_id']
                    gene_in_pathway.append(genes)
                    # ids = sub_result['term']['id']
                    # pathway_id.append(ids)
                    labels = sub_result['term']['label']
                    pathway_label.append(labels)
                else:
                    print(sub_result['input_list'])
        elif isinstance(result_list, dict):
            # print(result_list['input_list'])
            if 'mapped_id_list' in result_list['input_list']:
                genes = result_list['input_list']['mapped_id_list']['mapped_id']
                gene_in_pathway.append(genes)
                labels = result_list['term']['label']
                pathway_label.append(labels)
            else:
                print(result_list['input_list'])
        else:
            exit('error')

    return gene_in_pathway, pathway_label
# def predict_gene_pathway(gene_list, pathway_list):
#     pathway_data = read_pathway_file(pathway_file)
#     for gene in gene_list:
#         if gene in pathway_data:
#             pathway_list.append(pathway_data[gene])
#     return pathway_list

if __name__ == '__main__':
    gene_in_pathway, pathway_label = read_pathway_file('/home/yson2999/sequoia-pub/examples/analysis_molecular_function.json')
    print(len(gene_in_pathway))
    print(len(pathway_label))
    # create a dataframe with pathway_label and gene_in_pathway
    df = pd.DataFrame(columns=['pathway_name', 'gene_in_pathway'])
    # filter out have identical gene elements in gene_in_pathway list, if have, keep the first one, no matter the order
    # filter out the gene elements in gene_in pathway list that is included in another element
    for i, genes in enumerate(gene_in_pathway):
        if isinstance(genes, list):
            for j, genes_2 in enumerate(gene_in_pathway):
                if isinstance(genes_2, list):
                    # check if genes has the same elements as genes_2
                    if i != j and sorted(set(genes)) == sorted(set(genes_2)):
                        gene_in_pathway[i] = None
                        pathway_label[i] = None
        else:
            gene_in_pathway[i] = None
            pathway_label[i] = None

    for i, genes in enumerate(gene_in_pathway):
        if genes is not None:
            # make a string of genes separated by commas
            gene_str = ''
            for gene in genes:
                gene_str = gene_str + gene + ','
            gene_str = gene_str[:-1]
            pathway_name = pathway_label[i]
            df = df.append({'pathway_name': pathway_name, 'gene_in_pathway': gene_str}, ignore_index=True)
    # print(len(df))
    df.to_csv('/home/yson2999/sequoia-pub/examples/molecular_function_pathways.csv', index=False)
