import pandas as pd


def main():

    ref_file = pd.read_csv('examples/reference_LUAD_filtered_1000.csv')
    # calculate the variance of each gene
    # ref_file_var = ref_file.iloc[:, 3:].var()
    # ref_file = pd.concat([ref_file.iloc[:, :3], ref_file[sorted(ref_file_var.sort_values(ascending=False).index[:1000])]], axis=1)
    # print(ref_file.shape)
    # print(ref_file.head())
    # ref_file.to_csv('examples/reference_LUAD_filtered.csv', index=False)

    genes = ref_file.columns.str.split('_').str[1]
    print(genes)
    # write all genes to csv file
    with open('examples/gene_list_1000.csv', 'w') as f:
        for gene in genes:
            f.write(gene + '\n')


if __name__ == '__main__':
    main()