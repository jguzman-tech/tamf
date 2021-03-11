from tamf.utility import *

if __name__ == '__main__':
    my_model = model('./data/harding_and_union_grid.csv',
                     './data/all_route_assoc.pkl')
    my_model.create_grid_category_bar_chart(
        './plots/harding_and_union_grid_category_bar_chart.png'
    )
    print("NAIVE:")
    my_model.solve(5, tamf_algorithm.NAIVE)
    print("EOC_BINARY_DECISIONS:")
    my_model.solve(5, tamf_algorithm.EOC_BINARY_DECISIONS)
    print("Done!")
