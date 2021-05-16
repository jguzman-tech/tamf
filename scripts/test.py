from tamf.model import *
from tamf.analysis import *
from tamf.driver import *

# This is the "sanity check"
# I want to show all the intermediate utility values per route

if __name__ == '__main__':
    my_model = Model('./data/harding_and_union_grid_v2.csv',
                     './data/new_route_assoc.pkl')
    my_model.create_grid_category_bar_chart(
        './plots/harding_and_union_grid_category_bar_chart.png'
    )
    prefix = "./results/harding_and_union"
    print("UNIQUE_BLOCKS:")
    results_naive = my_model.solve(-1, Utility.UNIQUE_BLOCKS, f"{prefix}_naive_n=-1")
    print("POPULATED_BLOCKS_W_CONFLICT:")
    results_eoc = my_model.solve(-1, Utility.POPULATED_BLOCKS_W_CONFLICT, f"{prefix}_eoc_n=-1")
    my_model.filter_grid_by_route_ids(results_naive, f"{prefix}_naive_n=-1_grid.csv")
    my_model.filter_grid_by_route_ids(results_eoc, f"{prefix}_eoc_n=-1_grid.csv")
    my_model.print_grid_category_counts()
    print(f"kendalltau results: {perform_kendalltau(results_naive, results_eoc)}")
    print("Done!")
