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

# the nm_grid_file needs to be that which covers all of new mexico
# this grid will be truncated based on the counties' shape file
def create_county_grid_file(nm_grid_file, \
                            counties_shp_file, \
                            county_name, \
                            out_file):
    pass

class TamfAlgorithm(Enum):
    NAIVE = 1
    EOC_BINARY_DECISIONS = 2

class Model():
    def __init__(self, grid_file, route_file):
        # DataFrame containing all grid data
        # IMPORTANT: We assume the grid is "clipped" already
        self.grid_df = pd.read_csv(grid_file, sep='|')
        # dictionary formatted as: route_assoc[route_id] = <list-of-grid-ids>
        self.route_assoc = pickle.load(open(route_file, 'rb'))
        # filtered by the grid_df
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
    def evaluate_utility_naive(self, route_assoc, prev_grid_ids):
        value_dict = dict()
        for route_id in route_assoc:
            value_dict[route_id] = 0
            for inner_dict in route_assoc[route_id]:
                grid_id = inner_dict['id']
                if grid_id not in prev_grid_ids and \
                   inner_dict['POP10'] > 0.0:
                    value_dict[route_id] += 1
        return value_dict
    def evaluate_utility_w_eoc(self, route_assoc, prev_grid_ids):
        value_dict = dict()
        for route_id in route_assoc:
            value_dict[route_id] = 0
            for inner_dict in route_assoc[route_id]:
                grid_id = inner_dict['id']
                if grid_id not in prev_grid_ids and \
                   inner_dict['verizon_eoc'] == 0.5 and \
                   inner_dict['POP10'] > 0.0:
                    value_dict[route_id] += 1
        return value_dict
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
        new_df = pd.DataFrame(columns=["id", "grid_polygon", "route_ids",
                                       "contains_route", "verizon_eoc",
                                       "tmobile_eoc", "att_eoc", "sprint_eoc",
                                       "UR10", "b_type", "HOUSING10", "POP10"])
        grid_df = copy.deepcopy(self.grid_df)
        grid_df = grid_df.dropna(subset=['route_ids']).reset_index(drop=True)
        for i in range(len(grid_df)):
            route_ids = set(grid_df['route_ids'][i].split(','))
            if len(route_ids.intersection(selected_route_ids)) > 0:
                # keep the grid_id
                new_df.loc[len(new_df)] = grid_df.loc[i]
            if i % 1000 == 0:
                percentage = int(round(i/len(grid_df), 2) * 100)
                print(f"filtering grid_df... {percentage:2}% Complete")
        print(f"filtering grid_df... {100:2}% Complete")
        new_df.to_csv(out_file, index=False, sep="|")
        print(f"wrote: {out_file}")
    def solve(self, n, algorithm):
        if algorithm == TamfAlgorithm.NAIVE:
            func = self.evaluate_utility_naive
        elif algorithm == TamfAlgorithm.EOC_BINARY_DECISIONS:
            func = self.evaluate_utility_w_eoc
        count = 0
        route_assoc = copy.deepcopy(self.filtered_route_assoc)
        if n == -1:
            n = len(route_assoc)
        elif n > len(route_assoc):
            n = len(route_assoc)
        elif n < -1:
            raise Exception(f"Received invalid n argument of {n}")
        value_dict = func(route_assoc, set())
        selected_route_ids = []
        selected_grid_ids = set()
        while count < n:
            selected_route_id = max(value_dict, key=value_dict.get)
            selected_route_ids.append(selected_route_id)
            # selected_grid_ids |= route_assoc[selected_route_id]
            selected_grid_ids |= set([inner_dict['id'] \
                                      for inner_dict in \
                                      route_assoc[selected_route_id]])
            del route_assoc[selected_route_id]
            value_dict = func(route_assoc,
                              selected_grid_ids)
            count += 1
            print(f"{count}: {selected_route_id}")
        return selected_route_ids
    # should solve twice, compare the n ranks, -1 signifies all routes
    def perform_kendall_tau(self, out_file, n=-1):
        pass
