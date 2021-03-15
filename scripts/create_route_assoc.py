import pandas as pd
import pickle
import numpy as np

# might as well create the route_assoc here!
# need to combine columns in:
# /scratch/jfg95/research/old_tamf_work/tarun_work/Data/grid/grid_1km_sh_fcc_eoc_voc.csv
# and:
# /scratch/jfg95/research/old_tamf_work/route_grid.csv
if __name__ == '__main__':
    grid_df = pd.read_csv("/scratch/jfg95/research/old_tamf_work/tarun_work/Data/grid/grid_1km_sh_fcc_eoc_voc.csv", sep='|')
    grid_df = grid_df[["id", "grid_polygon", "verizon_sh", "tmobile_sh", "att_sh", "sprint_sh", "verizon_fcc", "tmobile_fcc", "att_fcc", "sprint_fcc", "verizon_eoc", "tmobile_eoc", "att_eoc", "sprint_eoc", "UR10", "b_type", "HOUSING10", "POP10"]]
    route_df = pd.read_csv("/scratch/jfg95/research/old_tamf_work/route_grid.csv", sep='|')
    route_df = route_df[["id", "grid_polygon", "route_ids", "contains_route"]]


    route_assoc = dict()
    # result_df = pd.DataFrame(columns=["id", "contains_route", "route_ids", "grid_polygon", "verizon_sh", "tmobile_sh", "att_sh", "sprint_sh", "verizon_fcc", "tmobile_fcc", "att_fcc", "sprint_fcc", "verizon_eoc", "tmobile_eoc", "att_eoc", "sprint_eoc", "UR10", "b_type", "HOUSING10", "POP10"])

    count_nans = 0
    for i, row in route_df.iterrows():
        if row['contains_route']:
            route_list = route_df['route_ids'][i]
            my_slice = grid_df.index[grid_df['id'] == row['id']]
            if len(my_slice) == 0:
                raise Exception("OOB!")
            else:
                my_index = my_slice[0]
            inner_dict = {"id":grid_df["id"][my_index],
                          "grid_polygon":grid_df["grid_polygon"][my_index],
                          "verizon_sh":grid_df["verizon_sh"][my_index],
                          "tmobile_sh":grid_df["tmobile_sh"][my_index],
                          "att_sh":grid_df["att_sh"][my_index],
                          "sprint_sh":grid_df["sprint_sh"][my_index],
                          "verizon_fcc":grid_df["verizon_fcc"][my_index],
                          "tmobile_fcc":grid_df["tmobile_fcc"][my_index],
                          "att_fcc":grid_df["att_fcc"][my_index],
                          "sprint_fcc":grid_df["sprint_fcc"][my_index],
                          "verizon_eoc":grid_df["verizon_eoc"][my_index],
                          "tmobile_eoc":grid_df["tmobile_eoc"][my_index],
                          "att_eoc":grid_df["att_eoc"][my_index],
                          "sprint_eoc":grid_df["sprint_eoc"][my_index],
                          "UR10":grid_df["UR10"][my_index],
                          "b_type":grid_df["b_type"][my_index],
                          "HOUSING10":grid_df["HOUSING10"][my_index],
                          "POP10":grid_df["POP10"][my_index]}
            if np.isnan(inner_dict['POP10']):
                inner_dict['POP10'] = 0.0
                count_nans += 1
            for route_id in route_list.split(','):
                if route_id not in route_assoc:
                    route_assoc[route_id] = [inner_dict]
                else:
                    route_assoc[route_id].append(inner_dict)
        if i % 2000 == 0:
            print(f"{i} out of {len(route_df)}")
    print(f"found {count_nans} nans")
    pickle.dump(route_assoc, open('./data/new_route_assoc.pkl', 'wb'))
    print(f"wrote: ./data/new_route_assoc.pkl")
