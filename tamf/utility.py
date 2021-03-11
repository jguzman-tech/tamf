import pandas as pd
import shapely.wkt
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure as fig
import pandas as pd
import statistics
import copy
import scipy.stats
from enum import Enum

class tamf_algorithm(Enum):
    NAIVE = 1
    EOC_BINARY_DECISIONS = 2

class model():
    def __init__(self, grid_file, route_file):
        print("in model contructor!")
        # DataFrame containing all grid data
        # IMPORTANT: We assume the grid is "clipped" already
        self.grid_df = pd.read_csv(grid_file, sep='|')
        # dictionary formatted as: route_assoc[route_id] = <list-of-grid-ids>
        self.route_assoc = pickle.load(open(route_file, 'rb'))
    def create_grid_category_bar_chart(self, out_file):
        # bar chart will show the block counts of the following:
        # 1. Total
        # 2. EoC != 0.5
        # 3. EoC == 0.5
        # 4. VoC == 0
        # 5. VoC > 1
        # 6. EoC == 0.5 & VoC > 1
        bar_names = ['Total',
                     'EoC != 0.5',
                     'EoC == 0.5',
                     'VoC == 0',
                     'VoC == 1',
                     'EoC == 0.5 &\nVoC == 1']
        y = [len(self.grid_df),
             len(self.grid_df[self.grid_df['verizon_eoc'] != 0.5]),
             len(self.grid_df[self.grid_df['verizon_eoc'] == 0.5]),
             len(self.grid_df[self.grid_df['POP10'] == 0]),
             len(self.grid_df[self.grid_df['POP10'] > 0]),
             len(self.grid_df[(self.grid_df['verizon_eoc'] == 0.5) & \
                         (self.grid_df['POP10'] > 0)])]
        colors = ['green','blue','purple','grey','teal', 'red']
        fig(figsize=(8,4))
        plt.bar(bar_names, y, color=colors)
        plt.title('Overall block categories')
        plt.xlabel('Category')
        plt.ylabel('Number of Blocks')
        plt.tight_layout()
        plt.savefig(out_file)
        plt.clf()
        print(f"wrote: {out_file}")
    def evaluate_utility_w_eoc(self, route_assoc, prev_grid_ids):
        value_dict = dict()
        for route_id in route_assoc:
            value_dict[route_id] = 0
            for grid_id in route_assoc[route_id]:
                my_slice = self.grid_df.index[self.grid_df['id'] == grid_id]
                if len(my_slice) == 0:
                    # out of bounds grid_id
                    continue
                else:
                    my_index = my_slice[0]
                if grid_id not in prev_grid_ids and \
                   np.isclose(self.grid_df['verizon_eoc'][my_index], 0.5) and \
                   self.grid_df['POP10'][my_index] > 0.0:
                    value_dict[route_id] += 1
        return value_dict
    def evaluate_utility_naive(self):
        pass
    def filter_route_assoc(self, route_assoc):
        for route_id in route_assoc:
            # out-of-bounds grid_ids, will remove
            oob_grid_ids = set()
            for grid_id in route_assoc[route_id]:
                my_slice = self.grid_df.index[self.grid_df['id'] == grid_id]
                if len(my_slice) == 0:
                    # out of bounds grid_id
                    oob_grid_ids.add(grid_id)
                else:
                    my_index = my_slice[0]
            route_assoc[route_id] = route_assoc[route_id] - oob_grid_ids

        # filter out any keys with empty sets
        route_assoc = dict((k,v) for k,v in route_assoc.items() \
                           if len(route_assoc[k]) > 0)
        return route_assoc
    def solve(self, n, algorithm):
        if algorithm == tamf_algorithm.NAIVE:
            pass
        elif algorithm == tamf_algorithm.EOC_BINARY_DECISIONS:
            count = 0
            route_assoc = copy.deepcopy(self.route_assoc)
            print(f"before ---- {len(route_assoc)}:{len(self.route_assoc)}")
            route_assoc = self.filter_route_assoc(route_assoc)
            print(f"after ---- {len(route_assoc)}:{len(self.route_assoc)}")
            value_dict = self.evaluate_utility_w_eoc(route_assoc, set())
            selected_route_ids = []
            selected_grid_ids = set()
            while count < n:
                selected_route_id = max(value_dict, key=value_dict.get)
                selected_route_ids.append(selected_route_id)
                selected_grid_ids |= route_assoc[selected_route_id]
                del route_assoc[selected_route_id]
                value_dict = self.evaluate_utility_w_eoc(route_assoc,
                                                    selected_grid_ids)
                count += 1
                print(f"{count}: {selected_route_id}\n")
        return selected_route_ids

# # 1. total number of blocks where EoC = 0.5
# # 2. total number of blocks where EoC = 0.5 and VoC = 1
# def print_block_counts(df, route_assoc, selected_route_ids):
#     visited = set() # list of grid_ids already visited
#     num_blocks_eoc = 0
#     num_blocks_eoc_voc = 0
#     for route_id in selected_route_ids:
#         for grid_id in route_assoc[route_id]:
#             if not (grid_id in visited):
#                 # visited.add(grid_id)
#                 my_slice = df.index[df['id'] == grid_id]
#                 if len(my_slice) == 0:
#                     # out of bounds grid_id
#                     continue
#                 else:
#                     my_index = my_slice[0]
#                 if np.isclose(df['verizon_eoc'][my_index], 0.5):
#                     num_blocks_eoc += 1
#                     if df['POP10'][my_index] > 0.0:
#                         num_blocks_eoc_voc += 1
#     print(f"total number of blocks where EoC = 0.5: {num_blocks_eoc}")
#     print(f"total number of blocks where EoC = 0.5 and VoC = 1: {num_blocks_eoc_voc}")

# def print_kendall_tau(ranks1, ranks2):
#     # to get that order I need to run both algos with n=15
#     # kendalltau needs two lists of ranks, in corresponding order
#     print(f"Kendall Tau: {scipy.stats.kendalltau({ranks1}, {ranks2})}")
                    

# def create_box_plots(df, route_assoc, selected_route_ids, prefix):
#     # 1. number of blocks per route (for each EoC value)
#     # 2. number of blocks per route (for each VoC value)
#     # 3. number of blocks per route (with EoC = 0.5 and VoC = 1)
#     eoc_data = dict()
#     eoc_data[0] = []
#     eoc_data[0.5] = []
#     eoc_data[1] = []

#     voc_data = dict()
#     voc_data[0] = []
#     voc_data[1] = []

#     valued_blocks_per_route = []

#     for route_id in selected_route_ids:
#         eoc_data[0].append(0)
#         eoc_data[0.5].append(0)
#         eoc_data[1].append(0)
#         voc_data[0].append(0)
#         voc_data[1].append(0)
#         valued_blocks_per_route.append(0)
#         for grid_id in route_assoc[route_id]:
#             my_slice = df.index[df['id'] == grid_id]
#             if len(my_slice) == 0:
#                 # out of bounds grid_id
#                 continue
#             else:
#                 my_index = my_slice[0]
#             if np.isclose(df['verizon_eoc'][my_index], 0.0):
#                 eoc_data[0][-1] += 1
#             elif np.isclose(df['verizon_eoc'][my_index], 0.5):
#                 eoc_data[0.5][-1] += 1
#                 if df['POP10'][my_index] > 0.0:
#                     valued_blocks_per_route[-1] += 1
#             else:
#                 eoc_data[1][-1] += 1
#             if df['POP10'][my_index] > 0.0:
#                 voc_data[1][-1] += 1
#             else:
#                 voc_data[0][-1] += 1

#     # plt.figure(figsize=(10, 8))
#     all_data = [eoc_data[0], eoc_data[0.5], eoc_data[1], voc_data[0], voc_data[1], valued_blocks_per_route]
#     print(f"EoC=0.5 & VoC=1: {print}")
#     labels = ['EoC=0', 'EoC=0.5', 'EoC=1', 'VoC=0', 'VoC=1', 'EoC=0.5 & VoC=1']
#     fig, ax = plt.subplots()
#     ax.boxplot(all_data, vert=False)
#     ax.set_yticklabels(labels)
#     plt.tight_layout()
#     plt.savefig(f'./boxplots/{prefix}.png')
#     plt.clf()

#     # plt.figure(figsize=(10, 3))
#     # plt.rcParams.update({'font.size': 18})
                
#     # plt.title(f"#-of-blocks with EoC = 0")
#     # plt.boxplot(eoc_data[0], vert=False, \
#     #             whiskerprops = dict(linestyle='-',linewidth=3.0, color='black'))
#     # plt.savefig(f"./boxplots/{prefix}_eoc_boxplot_0.png")
#     # plt.clf()

#     # plt.title(f"#-of-blocks with EoC = 0.5")
#     # plt.boxplot(eoc_data[0.5], vert=False, \
#     #             whiskerprops = dict(linestyle='-',linewidth=3.0, color='black'))
#     # plt.savefig(f"./boxplots/{prefix}_eoc_boxplot_0.5.png")
#     # plt.clf()

#     # plt.title(f"#-of-blocks with EoC = 1")
#     # plt.boxplot(eoc_data[1], vert=False, \
#     #             whiskerprops = dict(linestyle='-',linewidth=3.0, color='black'))
#     # plt.savefig(f"./boxplots/{prefix}_eoc_boxplot_1.png")
#     # plt.clf()

#     # plt.title(f"#-of-blocks with VoC = 0")
#     # plt.boxplot(voc_data[0], vert=False, \
#     #             whiskerprops = dict(linestyle='-',linewidth=3.0, color='black'))
#     # plt.savefig(f"./boxplots/{prefix}_voc_boxplot_0.png")
#     # plt.clf()

#     # plt.title(f"#-of-blocks with VoC = 1")
#     # plt.boxplot(voc_data[1], vert=False, \
#     #             whiskerprops = dict(linestyle='-',linewidth=3.0, color='black'))
#     # plt.savefig(f"./boxplots/{prefix}_voc_boxplot_1.png")
#     # plt.clf()

#     # plt.title(f"#-of-blocks with VoC = 1 and EoC = 0.5")
#     # plt.boxplot(valued_blocks_per_route, vert=False, \
#     #             whiskerprops = dict(linestyle='-',linewidth=3.0, color='black'))
#     # plt.savefig(f"./boxplots/{prefix}_value_boxplot.png")
#     # plt.clf()

# def create_grid_file_of_selected_routes(grid_df, selected_route_ids, prefix):
#     new_df = pd.DataFrame(columns=["id", "grid_polygon", "route_ids", "contains_route", "verizon_eoc", "tmobile_eoc", "att_eoc", "sprint_eoc", "UR10", "b_type", "HOUSING10", "POP10"])
#     grid_df = grid_df.dropna(subset=['route_ids']).reset_index(drop=True)
#     for i in range(len(grid_df)):
#         route_ids = set(grid_df['route_ids'][i].split(','))
#         if len(route_ids.intersection(selected_route_ids)) > 0:
#             # keep the grid_id
#             new_df.loc[len(new_df)] = grid_df.loc[i]
#         if i % 1000 == 0:
#             print(f"{i} out of {len(grid_df)}")
#     new_df.to_csv(f'./data/{prefix}_route_grid.csv', index=False, sep="|")

# # will remove any route blocks if they are out of bounds
# # if the route is empty it'll be removed from route_assoc
# def filter_routes(grid_df, route_assoc):
#     for route_id in route_assoc:
#         # out-of-bounds grid_ids, will remove
#         oob_grid_ids = set()
#         for grid_id in route_assoc[route_id]:
#             my_slice = grid_df.index[df['id'] == grid_id]
#             if len(my_slice) == 0:
#                 # out of bounds grid_id
#                 oob_grid_ids.add(grid_id)
#             else:
#                 my_index = my_slice[0]
#         route_assoc[route_id] = route_assoc[route_id] - oob_grid_ids

#     # filter out any keys with empty sets
#     route_assoc = dict((k,v) for k,v in route_assoc.items() \
#                        if len(route_assoc[k]) > 0)
#     return route_assoc

# def create_six_number_summary(x, x_label, prefix):
#     # mean, median, mode, stdev, min, max
#     mean = statistics.mean(x)
#     median = statistics.median(x)
#     try:
#         mode = statistics.mode(x)
#     except:
#         mode = "N/A"
#     stdev = statistics.stdev(x)

#     summary_fp = open(f"./data/{prefix}_summary.txt", 'w')
#     summary_fp.write(f"{x_label}\n")
#     summary_fp.write(f"mean: {mean}\n")
#     summary_fp.write(f"median: {median}\n")
#     summary_fp.write(f"mode: {mode}\n")
#     summary_fp.write(f"stdev: {stdev}\n")
#     summary_fp.write(f"min: {min(x)}\n")
#     summary_fp.write(f"max: {max(x)}\n")
#     summary_fp.close()

# def create_cdf(grid_df, route_assoc, title, prefix):
#     # will create a CDF and CSV with route length data
#     # route length is the number of blocks the route has within our clipped grid
#     # note that this information depends on the route_assoc, and grid_df...
#     # not the algorithm selected

#     df = pd.DataFrame(columns=['route_id', 'length'])

#     for route_id in route_assoc:
#         df.loc[len(df)] = [route_id, len(route_assoc[route_id])]

#     df = df.sort_values(by=['length'], ascending=False, ignore_index=True)
#     df.to_csv(f'./data/{prefix}_route_data.csv', sep='|', index=False)

#     x = sorted(df['length'].to_list())
#     y = np.arange(len(x)) / (len(x)-1)
#     # import pdb; pdb.set_trace()
#     assert np.isclose(y[-1], 1.0), "CDF doesn't end at 1.0"
#     plt.title("CDF of route length (# blocks in grid)")
#     plt.xlabel("route length")
#     plt.ylabel("p")
#     plt.plot(x, y)
#     plt.savefig(f"./data/{prefix}_cdf.png")
#     plt.clf()
#     create_six_number_summary(x, 'route length', prefix)

# # must omit any overlap with prev_routes
# def value_from_eoc_and_voc(route_assoc, df, prev_grid_ids):
#     value_dict = dict()
#     for route_id in route_assoc:
#         value_dict[route_id] = 0
#         for grid_id in route_assoc[route_id]:
#             my_slice = df.index[df['id'] == grid_id]
#             if len(my_slice) == 0:
#                 # out of bounds grid_id
#                 continue
#             else:
#                 my_index = my_slice[0]
#             if grid_id not in prev_grid_ids and \
#                np.isclose(df['verizon_eoc'][my_index], 0.5) and \
#                df['POP10'][my_index] > 0.0:
#                 value_dict[route_id] += 1
#     return value_dict

# def value_from_voc(route_assoc, df):
#     value_dict = dict()
#     for route_id in route_assoc:
#         value_dict[route_id] = 0
#         for grid_id in route_assoc[route_id]:
#             my_slice = df.index[df['id'] == grid_id]
#             if len(my_slice) == 0:
#                 # out of bounds grid_id
#                 continue
#             else:
#                 my_index = my_slice[0]
#             if df['POP10'][my_index] > 0.0:
#                 value_dict[route_id] += 1
#     return value_dict

# # refer to the wikipedia page:
# # https://en.wikipedia.org/wiki/Maximum_coverage_problem
# # we have a weighted solution

# # df contains all data
# # route_assoc is a dict with route_assoc[route_id] = <set-of-gridids>
# # max_routes is the maximum number of routes we may select
# def eoc_solution(df, route_assoc, max_routes, prefix):
#     result_fp = open(f"./data/{prefix}_tamf.txt", "w")

#     # value_dict is a dict with value[route_id] = <scalar based on EoC/VoC>
#     value_dict = value_from_eoc_and_voc(route_assoc, df, set())
#     count = 0
#     selected_route_ids = []
#     selected_grid_ids = set()
#     while count < max_routes:
#         best_route_id = max(value_dict, key=value_dict.get)
#         selected_route_ids.append(best_route_id)
#         selected_grid_ids |= route_assoc[best_route_id]
#         del route_assoc[best_route_id]
#         value_dict = value_from_eoc_and_voc(route_assoc, df, selected_grid_ids)
#         count += 1
#         result_fp.write(f"{count}: {best_route_id}\n")
#     result_fp.close()
#     return selected_route_ids

# def simple_solution(df, route_assoc, max_routes, prefix):
#     result_fp = open(f"./data/{prefix}_tamf.txt", "w")
#     # value_dict is a dict with value[route_id] = <scalar based on EoC/VoC>
#     value_dict = value_from_voc(route_assoc, df)
#     count = 0
#     selected_route_ids = []
#     selected_grid_ids = set()
#     while count < max_routes:
#         best_route_id = max(value_dict, key=value_dict.get)
#         selected_route_ids.append(best_route_id)
#         selected_grid_ids |= route_assoc[best_route_id]
#         del route_assoc[best_route_id]
#         del value_dict[best_route_id]
#         count += 1
#         result_fp.write(f"{count}: {best_route_id}\n")
#     result_fp.close()        
#     return selected_route_ids

# if __name__ == '__main__':
#     # was using ./new_selection.csv before
#     df = pd.read_csv(sys.argv[1], sep='|')
#     route_assoc = pickle.load(open('./route_assoc.pkl', 'rb'))
#     route_assoc = filter_routes(df, route_assoc)
#     n = int(sys.argv[3])
#     prefix = sys.argv[4]
#     create_cdf(df,
#                route_assoc,
#                "CDF of route length (# blocks in grid)",
#                prefix,
#     )    
#     if sys.argv[2] == 'e':
#         # eoc solution
#         result = eoc_solution(df, copy.deepcopy(route_assoc), n, prefix)
#         print(f"result = {result}")
#     elif sys.argv[2] == 's':
#         # simple solution
#         result = simple_solution(df, copy.deepcopy(route_assoc), n, prefix)
#         print(f"result = {result}")
#     elif sys.argv[2] == 'a':
#         # don't solve we just want the box plots
#         result = list(route_assoc.keys())
#     elif sys.argv[2] == 'k':
#         # get kendall tau, exit
#         route_ids = list(route_assoc.keys())
#         result1 = eoc_solution(df, copy.deepcopy(route_assoc), n, prefix)
#         result2 = simple_solution(df, copy.deepcopy(route_assoc), n, prefix)
#         ranks1 = []
#         ranks2 = []
#         for route_id in route_ids:
#             ranks1.append(result1.index(route_id) + 1)
#             ranks2.append(result2.index(route_id) + 1)
#         print_kendall_tau(ranks1, ranks2)        
            
#     else:
#         print(f"unknown parameter {sys.argv[2]}")
#         sys.exit(1)

#     create_grid_file_of_selected_routes(df, result, prefix)
#     create_box_plots(df, route_assoc, result, prefix)
#     print_block_counts(df, route_assoc, result)
