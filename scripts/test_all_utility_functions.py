from tamf.model import *

if __name__ == '__main__':
    my_model = Model('./data/harding_and_union_grid_v2.csv',
                     './data/new_route_assoc.pkl')
    my_model.create_grid_category_bar_chart(
        './plots/harding_and_union_grid_category_bar_chart.png'
    )
    result_dir = "./results/harding_and_union"

    for util in Utility:
        my_model.solve(-1, util, f"{result_dir}/{util}_n=-1")
        print(f"completed {util}\n")

    print("Done!")
