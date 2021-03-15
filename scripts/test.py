from tamf.utility import *

if __name__ == '__main__':
    my_model = Model('./data/harding_and_union_grid_v2.csv',
                     './data/new_route_assoc.pkl')
    my_model.create_grid_category_bar_chart(
        './plots/harding_and_union_grid_category_bar_chart.png'
    )
    print("NAIVE:")
    results_naive = my_model.solve(-1, Heuristic.NAIVE)
    result_file_naive = './results/grid_df_for_harding_and_union_counties_naive_n=-1.csv'
    print("EOC_BINARY_DECISIONS:")
    results_eoc = my_model.solve(-1, Heuristic.EOC_BINARY_DECISIONS)
    result_file_eoc = './results/grid_df_for_harding_and_union_counties_eoc_n=-1.csv'
    my_model.filter_grid_by_route_ids(results_naive, result_file_naive)
    my_model.filter_grid_by_route_ids(results_eoc, result_file_eoc)
    my_model.print_grid_category_counts()
    print(f"kendalltau results: {perform_kendalltau(results_naive, results_eoc)}")
    print("Done!")
