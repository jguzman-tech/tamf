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
    def evaluate_utility_naive(self, route_assoc, prev_grid_ids):
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
                   self.grid_df['POP10'][my_index] > 0.0:
                    value_dict[route_id] += 1
        return value_dict

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
            func = self.evaluate_utility_naive
        elif algorithm == tamf_algorithm.EOC_BINARY_DECISIONS:
            func = self.evaluate_utility_w_eoc
        count = 0
        route_assoc = copy.deepcopy(self.route_assoc)
        print(f"before ---- {len(route_assoc)}:{len(self.route_assoc)}")
        route_assoc = self.filter_route_assoc(route_assoc)
        print(f"after ---- {len(route_assoc)}:{len(self.route_assoc)}")
        value_dict = func(route_assoc, set())
        selected_route_ids = []
        selected_grid_ids = set()
        while count < n:
            selected_route_id = max(value_dict, key=value_dict.get)
            selected_route_ids.append(selected_route_id)
            selected_grid_ids |= route_assoc[selected_route_id]
            del route_assoc[selected_route_id]
            value_dict = func(route_assoc,
                              selected_grid_ids)
            count += 1
            print(f"{count}: {selected_route_id}")
        return selected_route_ids
