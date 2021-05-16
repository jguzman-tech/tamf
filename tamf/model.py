import pandas as pd
import shapely.wkt
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import figure as fig
import pandas as pd
import statistics
import copy
import scipy.stats
from enum import Enum
import geopandas as gpd
import shapely
from shapely.ops import cascaded_union
import os

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

class Utility(Enum):
    UNIQUE_BLOCKS = 1
    POPULATED_BLOCKS = 2
    POPULATED_BLOCKS_W_CONFLICT = 3
    POPULATED_BLOCKS_W_SCALED_CONFLICT = 4
    SCALING_BY_LEVEL = 5
    SCALING_BY_LEVEL_W_CONFLICT = 6
    SCALING_BY_LEVEL_W_SCALED_CONFLICT = 7

class Model():
    def __init__(self, grid_file, route_file, state_wide=False):
        # DataFrame containing all grid data
        # IMPORTANT: We assume the grid is "clipped" already
        self.grid_df = pd.read_csv(grid_file, sep='|')
        self.grid_df['POP10'] = self.grid_df['POP10'].fillna(0.0)
        # dictionary formatted as: route_assoc[route_id] = <list-of-grid-ids>
        self.route_assoc = pickle.load(open(route_file, 'rb'))
        if state_wide:
            self.filtered_route_assoc = copy.deepcopy(self.route_assoc)
        else:
            print(f"Model.__init__: filtering routes...")
            self.filtered_route_assoc = self.filter_route_assoc(self.route_assoc)
        print(f"using {len(self.filtered_route_assoc)} out of {len(self.route_assoc)} routes")
    def get_grid_category_counts(self):
        result = [len(self.grid_df),
                  len(self.grid_df[self.grid_df['verizon_eoc'] != 0.5]),
                  len(self.grid_df[self.grid_df['verizon_eoc'] == 0.5]),
                  len(self.grid_df[self.grid_df['POP10'] == 0]),
                  len(self.grid_df[self.grid_df['POP10'] > 0]),
                  len(self.grid_df[(self.grid_df['verizon_eoc'] == 0.5) & \
                                   (self.grid_df['POP10'] > 0)])]
        return result
    def print_grid_category_counts(self):
        counts = self.get_grid_category_counts()
        print(f"Total: {counts[0]}")
        print(f"EoC != 0.5: {counts[1]}")
        print(f"EoC == 0.5: {counts[2]}")
        print(f"VoC == 0: {counts[3]}")
        print(f"VoC == 1: {counts[4]}")
        print(f"EoC == 0.5 & VoC == 1: {counts[5]}")
    def create_grid_category_bar_chart(self, out_file):
        # bar chart will show the block counts of the following:
        # 1. Total
        # 2. EoC != 0.5
        # 3. EoC == 0.5
        # 4. VoC == 0
        # 5. VoC == 1
        # 6. EoC == 0.5 & VoC == 1
        bar_names = ['Total',
                     'EoC != 0.5',
                     'EoC == 0.5',
                     'VoC == 0',
                     'VoC == 1',
                     'EoC == 0.5 &\nVoC == 1']
        y = self.get_grid_category_counts()
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
    def get_utility_dict(self, route_assoc, prev_grid_ids, func):
        utility_dict = dict()
        for route_id in route_assoc:
            utility_dict[route_id] = 0
            for inner_dict in route_assoc[route_id]:
                grid_id = inner_dict['id']
                if grid_id not in prev_grid_ids:
                    utility_dict[route_id] += func(inner_dict)
        return utility_dict
    def utility_unique_blocks(self, inner_dict):
        return 1
    def utility_populated_blocks(self, inner_dict):
        if inner_dict['POP10'] > 0.0:
            return 1
        else:
            return 0
    def utility_populated_blocks_w_conflict(self, inner_dict):
        if inner_dict['verizon_eoc'] == 0.5 or \
           inner_dict['tmobile_eoc'] == 0.5 or \
           inner_dict['att_eoc'] == 0.5 or \
           inner_dict['sprint_eoc'] == 0.5:
            return self.utility_populated_blocks(inner_dict)
        else:
            return 0
    def utility_populated_blocks_w_scaled_conflict(self, inner_dict):
        conflicts = 0
        if inner_dict['verizon_eoc'] == 0.5:
            conflicts += 1
        if inner_dict['tmobile_eoc'] == 0.5:
            conflicts += 1
        if inner_dict['att_eoc'] == 0.5:
            conflicts += 1
        if inner_dict['sprint_eoc'] == 0.5:
            conflicts += 1
        return conflicts * self.utility_populated_blocks(inner_dict)
    def utility_scaling_by_level(self, inner_dict):
        if inner_dict['POP10'] == 0.0:
            return 0
        elif 0.0 < inner_dict['POP10'] < 3.0:
            return 1
        elif 3.0 <= inner_dict['POP10'] < 5.0:
            return 2
        elif 5.0 <= inner_dict['POP10'] <= 10.0:
            return 3
        elif 10.0 <= inner_dict['POP10'] <= 25:
            return 4
        elif 25.0 < inner_dict['POP10']:
            return 5
    def utility_scaling_by_level_w_conflict(self, inner_dict):
        if inner_dict['verizon_eoc'] == 0.5:
            return self.utility_scaling_by_level(inner_dict)
        else:
            return 0
    def utility_scaling_by_level_w_scaled_conflict(self, inner_dict):
        conflicts = 0
        if inner_dict['verizon_eoc'] == 0.5:
            conflicts += 1
        if inner_dict['tmobile_eoc'] == 0.5:
            conflicts += 1
        if inner_dict['att_eoc'] == 0.5:
            conflicts += 1
        if inner_dict['sprint_eoc'] == 0.5:
            conflicts += 1
        return conflicts * self.utility_scaling_by_level(inner_dict)
    def filter_route_assoc(self, route_assoc):
        oob_route_ids = []
        for route_id in route_assoc:
            # count of grid_ids within the region
            count = 0
            for inner_dict in route_assoc[route_id]:
                grid_id = inner_dict['id']
                my_slice = self.grid_df.index[self.grid_df['id'] == grid_id]
                if len(my_slice) == 1:
                    # in bounds grid_id
                    count += 1
            if count == 0:
                oob_route_ids.append(route_id)
        route_assoc = dict([(key, val) for key, val in route_assoc.items() \
                            if key not in oob_route_ids])
        return route_assoc
    def filter_grid_by_route_ids(self, selected_route_ids, out_file):
        # This may be made simpler. Filter the main dataset by the set of grid ids.
        # After getting the filtered df add the route_ids column by iterating through.
        # This function is not generic enough, if we were to add more columns.
        region_grid_ids = set(self.grid_df['id'].to_list())
        # grid_dict[grid_id] = row, necessary so we can get a route_ids column
        grid_dict = dict()
        for route_id in selected_route_ids:
            route_grid_ids = set([inner_dict['id'] \
                            for inner_dict in self.route_assoc[route_id]])
            if len(region_grid_ids.intersection(route_grid_ids)) > 0:
                # keep the route_id
                for inner_dict in self.route_assoc[route_id]:
                    keyorder = self.grid_df.columns.to_list()
                    row = copy.deepcopy(inner_dict)
                    row = sorted(row.items(), key=lambda i:keyorder.index(i[0]))
                    row = [elem[1] for elem in row]
                    grid_id = inner_dict['id']
                    if grid_id not in grid_dict:
                        grid_dict[grid_id] = row + [route_id]
                    else:
                        grid_dict[grid_id][-1] += f",{route_id}"
        my_cols = ["id","grid_polygon","verizon_sh","tmobile_sh","att_sh","sprint_sh","verizon_fcc","tmobile_fcc","att_fcc","sprint_fcc","verizon_eoc","tmobile_eoc","att_eoc","sprint_eoc","UR10","b_type","HOUSING10","POP10","route_ids"]
        new_df = pd.DataFrame(columns=my_cols)
        for grid_id in grid_dict:
            new_df.loc[len(new_df)] = grid_dict[grid_id]
        new_df.to_csv(out_file, index=False, sep="|")
        print(f"wrote: {out_file}")
    def solve(self, n, func, prefix, debug=True):
        if func == Utility.UNIQUE_BLOCKS:
            func = self.utility_unique_blocks
        elif func == Utility.POPULATED_BLOCKS:
            func = self.utility_populated_blocks
        elif func == Utility.POPULATED_BLOCKS_W_CONFLICT:
            func = self.utility_populated_blocks_w_conflict
        elif func == Utility.POPULATED_BLOCKS_W_SCALED_CONFLICT:
            func = self.utility_populated_blocks_w_scaled_conflict            
        elif func == Utility.SCALING_BY_LEVEL:
            func = self.utility_scaling_by_level
        elif func == Utility.SCALING_BY_LEVEL_W_CONFLICT:
            func = self.utility_scaling_by_level_w_conflict
        elif func == Utility.SCALING_BY_LEVEL_W_SCALED_CONFLICT:
            func = self.utility_scaling_by_level_w_scaled_conflict
        utility_fp = open(f"{prefix}_utility_scores.txt", 'w')
        route_fp = open(f"{prefix}_route_ids.txt", "w")
        count = 0
        route_assoc = copy.deepcopy(self.filtered_route_assoc)
        if n == -1:
            n = len(route_assoc)
        elif n > len(route_assoc):
            n = len(route_assoc)
        elif n < -1:
            raise Exception(f"Received invalid n argument of {n}")
        utility_dict = self.get_utility_dict(route_assoc, set(), func)
        utility_fp.write(f"{utility_dict}\n\n")
        selected_route_ids = []
        selected_grid_ids = set()
        while count < n and len(route_assoc) > 0:
            selected_route_id = max(utility_dict, key=utility_dict.get)
            route_fp.write(f"{selected_route_id}\n")
            selected_route_ids.append(selected_route_id)
            # selected_grid_ids |= route_assoc[selected_route_id]
            selected_grid_ids |= set([inner_dict['id'] \
                                      for inner_dict in \
                                      route_assoc[selected_route_id]])
            del route_assoc[selected_route_id]
            utility_dict = self.get_utility_dict(route_assoc,
                                               selected_grid_ids,
                                               func)
            utility_fp.write(f"{utility_dict}\n\n")
            count += 1
            print(f"{count}: {selected_route_id}")
        utility_fp.close()
        route_fp.close()
        print(f"wrote: {prefix}_utility_scores.txt")
        print(f"wrote: {prefix}_route_ids.txt")
        return selected_route_ids
