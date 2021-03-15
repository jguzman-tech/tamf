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

def create_subplot(result_files, title, ax):
    labels = ['EoC != 0.5 &\nPOP == 0',
              'EoC == 0.5 &\nPOP == 0',
              'EoC != 0.5 &\nPOP > 0',
              'EoC == 0.5 &\nPOP > 0']
    denom = None
    proportions = dict()
    for h in result_files:
        result_file = result_files[h]
        df = pd.read_csv(result_file, sep='|')
        df['POP10'] = df['POP10'].fillna(0.0)
        if denom is None and h == Heuristic.ALL:
            denom = float(len(df))
        prop0 = int(100 * len(df[(df['verizon_eoc'] != 0.5) & (df['POP10'] == 0.0)]) / denom)
        prop1 = int(100 * len(df[(df['verizon_eoc'] == 0.5) & (df['POP10'] == 0.0)]) / denom)
        prop2 = int(100 * len(df[(df['verizon_eoc'] != 0.5) & (df['POP10'] > 0.0)]) / denom)
        prop3 = int(100 * len(df[(df['verizon_eoc'] == 0.5) & (df['POP10'] > 0.0)]) / denom)
        proportions[str(h).replace('Heuristic.', '')] = [prop0, prop1, prop2, prop3]
    x = np.arange(len(labels))
    width= 0.1 # width of the bars
    ax.bar(x - 2*width, proportions['ALL'], width, label='ALL')
    ax.bar(x - width, proportions['NAIVE'], width, label='NAIVE')
    ax.bar(x, proportions['EOC_BINARY_DECISIONS'], width, label='EOC_BINARY_DECISIONS')
    ax.bar(x + width, proportions['SCALING_BY_LEVEL'], width, label='SCALING_BY_LEVEL')
    ax.bar(x + 2*width, proportions['SCALING_BY_LEVEL_W_EOC'], width, label='SCALING_BY_LEVEL_W_EOC')
    ax.set_ylabel('Percentage')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

def create_aggregate_bar_chart(n, out_file):
    nm_county_fips = ['011', '035', '003', '059', '047', '055', '017', '007', '043', '006', '013', '021', '023', '053', '028', '033', '015', '009', '041', '045', '027', '019', '057', '029', '031', '039', '025', '005', '049', '037', '001', '051', '061']
    c = 7
    r = 5
    fig1, axes = plt.subplots(ncols=c, nrows=r, constrained_layout=True)
    axes = list(axes.flatten())
    fig1.set_size_inches(35, 20)
    for i in range(c*r):
        if i != 0 and i < len(nm_county_fips) + 1:
            fips = nm_county_fips[i-1]
            result_files = dict()
            for h in Heuristic:
                result_file = f"./results/by_county/naive_{fips}_{h}_n=1.csv"
                result_files[h] = result_file
            create_subplot(result_files, f"FIPS:{fips}", axes[i])
        else:
            axes[i].axis('off')
    handles, labels = axes[1].get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='upper left', prop={'size': 18})
    plt.suptitle("Block Category Percentages For Each County (N=1)")
    plt.savefig(out_file)
    print(f"wrote: {out_file}")

# result_files should be a dictionary
# result_files[heuristic] = result_file
def create_plot(result_files, out_file, title):
    labels = ['EoC != 0.5 &\nPOP == 0',
              'EoC == 0.5 &\nPOP == 0',
              'EoC != 0.5 &\nPOP > 0',
              'EoC == 0.5 &\nPOP > 0']
    denom = None
    proportions = dict()
    for h in result_files:
        result_file = result_files[h]
        df = pd.read_csv(result_file, sep='|')
        df['POP10'] = df['POP10'].fillna(0.0)
        if denom is None and h == Heuristic.ALL:
            denom = float(len(df))
        prop0 = int(100 * len(df[(df['verizon_eoc'] != 0.5) & (df['POP10'] == 0.0)]) / denom)
        prop1 = int(100 * len(df[(df['verizon_eoc'] == 0.5) & (df['POP10'] == 0.0)]) / denom)
        prop2 = int(100 * len(df[(df['verizon_eoc'] != 0.5) & (df['POP10'] > 0.0)]) / denom)
        prop3 = int(100 * len(df[(df['verizon_eoc'] == 0.5) & (df['POP10'] > 0.0)]) / denom)
        proportions[str(h).replace('Heuristic.', '')] = [prop0, prop1, prop2, prop3]
    x = np.arange(len(labels))
    width= 0.1 # width of the bars
    fig, ax = plt.subplots()
    # import pdb; pdb.set_trace()
    ax.bar(x - 2*width, proportions['ALL'], width, label='ALL')
    ax.bar(x - width, proportions['NAIVE'], width, label='NAIVE')
    ax.bar(x, proportions['EOC_BINARY_DECISIONS'], width, label='EOC_BINARY_DECISIONS')
    ax.bar(x + width, proportions['SCALING_BY_LEVEL'], width, label='SCALING_BY_LEVEL')
    ax.bar(x + 2*width, proportions['SCALING_BY_LEVEL_W_EOC'], width, label='SCALING_BY_LEVEL_W_EOC')
    ax.set_ylabel('Percentage')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"wrote: {out_file}")
    plt.clf()

def perform_kendalltau(result1, result2):
    ranks1 = []
    ranks2 = []
    for route_id in set(result1):
        ranks1.append(result1.index(route_id) + 1)
        ranks2.append(result2.index(route_id) + 1)
    tau, p = scipy.stats.kendalltau(ranks1, ranks2)
    return tau, p

def create_population_cdf(grid_file, plot_file, summary_file):
    grid_df = pd.read_csv(grid_file, sep='|')
    # fill na with 0.0
    grid_df['POP10'] = grid_df['POP10'].fillna(0.0)
    # get rid of all 0s
    x = sorted(grid_df['POP10'][grid_df['POP10'] != 0.0].to_list())
    # import pdb; pdb.set_trace()
    y = np.arange(len(x)) / (len(x)-1)
    # import pdb; pdb.set_trace()
    assert np.isclose(y[-1], 1.0), "CDF doesn't end at 1.0"
    plt.title("CDF of population-per-block")
    plt.xlabel("population")
    plt.ylabel("p")
    plt.plot(x, y)
    plt.savefig(plot_file)
    print(f"wrote: {plot_file}")
    plt.clf()
    # six-number summary: mean, median, mode, stdev, min, max
    mean = statistics.mean(x)
    median = statistics.median(x)
    try:
        mode = statistics.mode(x)
    except:
        mode = "N/A"
    stdev = statistics.stdev(x)
    summary_fp = open(summary_file, 'w')
    summary_fp.write(f"---- population per block ----\n")
    summary_fp.write(f"mean: {mean}\n")
    summary_fp.write(f"median: {median}\n")
    summary_fp.write(f"mode: {mode}\n")
    summary_fp.write(f"stdev: {stdev}\n")
    summary_fp.write(f"min: {min(x)}\n")
    summary_fp.write(f"max: {max(x)}\n")
    summary_fp.write("\n")
    summary_fp.write("potential boundaries for 10-point scale:\n")
    for i in range(11):
        percentile = int(i * (100/10))
        summary_fp.write(f"{percentile}th percentile: {np.percentile(x, percentile)}\n")
    summary_fp.close()
    print(f"wrote: {summary_file}")

# the nm_grid_file needs to be that which covers all of new mexico
# this grid will be truncated based on the counties' shape file
def create_county_grid_file(nm_grid_file, \
                            counties_shp_file, \
                            county_fips, \
                            out_file):
    grid_df = pd.read_csv(nm_grid_file, sep='|')
    counties_df = gpd.read_file(counties_shp_file)
    my_slice = counties_df[(counties_df['STATEFP'] == '35') & \
                           (counties_df['COUNTYFP'] == county_fips)]
    clipped_df = pd.DataFrame(columns=grid_df.columns.to_list())
    count = 0
    if len(my_slice) == 1:
        index = my_slice.index[0]
        county_polygon = counties_df.loc[index, 'geometry']
        for i in range(len(grid_df)):
            block_polygon = shapely.wkt.loads(grid_df['grid_polygon'][i])
            if block_polygon.intersects(county_polygon):
                clipped_df.loc[len(clipped_df)] = grid_df.loc[i].to_list()
                count += 1
            if i % 2000 == 0:
                percentage = int(round(i/len(grid_df), 2) * 100)
                print(f"filtering out county... {percentage:2}% Complete")
        print(f"filtering out county... {100:2}% Complete")
    else:
        raise Exception(f"Could not lookup county fips of {county_fips}")
    clipped_df.to_csv(out_file, sep='|', index=False)
    print(f"wrote: {out_file}")

class Heuristic(Enum):
    ALL = 0 # selects all available routes, ignoring n, used for comparison
    NAIVE = 1
    EOC_BINARY_DECISIONS = 2
    SCALING_BY_LEVEL = 3
    SCALING_BY_LEVEL_W_EOC = 4

class Model():
    def __init__(self, grid_file, route_file):
        # DataFrame containing all grid data
        # IMPORTANT: We assume the grid is "clipped" already
        self.grid_df = pd.read_csv(grid_file, sep='|')
        self.grid_df['POP10'] = self.grid_df['POP10'].fillna(0.0)
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
    def evaluate_utility_scaling_by_level(self, route_assoc, prev_grid_ids):
        value_dict = dict()
        for route_id in route_assoc:
            value_dict[route_id] = 0
            for inner_dict in route_assoc[route_id]:
                grid_id = inner_dict['id']
                if grid_id not in prev_grid_ids:
                    if inner_dict['POP10'] == 0.0:
                        # same as adding 0
                        pass
                    elif 0.0 < inner_dict['POP10'] < 3.0:
                        value_dict[route_id] += 1
                    elif 3.0 <= inner_dict['POP10'] < 5.0:
                        value_dict[route_id] += 2
                    elif 5.0 <= inner_dict['POP10'] <= 10.0:
                        value_dict[route_id] += 3
                    elif 10.0 <= inner_dict['POP10'] <= 25:
                        value_dict[route_id] += 4
                    elif 25.0 < inner_dict['POP10']:
                        value_dict[route_id] += 5
        return value_dict
    def evaluate_utility_scaling_by_level_w_eoc(self, route_assoc, prev_grid_ids):
        value_dict = dict()
        for route_id in route_assoc:
            value_dict[route_id] = 0
            for inner_dict in route_assoc[route_id]:
                grid_id = inner_dict['id']
                if grid_id not in prev_grid_ids:
                    if inner_dict['POP10'] == 0.0:
                        # same as adding 0
                        pass
                    elif 0.0 < inner_dict['POP10'] < 3.0:
                        value_dict[route_id] += 1
                    elif 3.0 <= inner_dict['POP10'] < 5.0:
                        value_dict[route_id] += 2
                    elif 5.0 <= inner_dict['POP10'] <= 10.0:
                        value_dict[route_id] += 3
                    elif 10.0 <= inner_dict['POP10'] <= 25:
                        value_dict[route_id] += 4
                    elif 25.0 < inner_dict['POP10']:
                        value_dict[route_id] += 5
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
    def solve(self, n, heuristic, debug=True):
        if heuristic == Heuristic.ALL:
            return list(self.filtered_route_assoc.keys())
        elif heuristic == Heuristic.NAIVE:
            func = self.evaluate_utility_naive
        elif heuristic == Heuristic.EOC_BINARY_DECISIONS:
            func = self.evaluate_utility_w_eoc
        elif heuristic == Heuristic.SCALING_BY_LEVEL:
            func = self.evaluate_utility_scaling_by_level
        elif heuristic == Heuristic.SCALING_BY_LEVEL_W_EOC:
            func = self.evaluate_utility_scaling_by_level_w_eoc
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
