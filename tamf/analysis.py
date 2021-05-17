from tamf.model import *
import scipy.stats
from pathlib import Path
import os
import seaborn as sns
from ast import literal_eval
import numpy as np

def create_routes_per_county_cdf():
    # will add these counts to the summary
    route_count_dict = dict()
    route_counts = []
    directory = './data/by_county/'
    count = 0
    for fname in os.listdir(directory):
        m = model.Model(os.path.join(directory, fname), './data/new_route_assoc.pkl')
        routes = len(m.filtered_route_assoc)
        route_counts.append(routes)
        route_count_dict[fname.replace('.csv', '')] = routes
        count += 1
        print(f"count = {count}")
    x = sorted(route_counts)
    y = np.arange(len(x)) / (len(x)-1)
    assert np.isclose(y[-1], 1.0), "CDF doesn't end at 1.0"
    plt.title("CDF of routes-per-county")
    plt.xlabel("routes-per-county")
    plt.ylabel("p")
    plt.plot(x, y)
    plt.savefig("./plots/routes_cdf.png")
    print("wrote: ./plots/routes_cdf.png")
    plt.clf()
    # six-number summary: mean, median, mode, stdev, min, max
    mean = statistics.mean(x)
    median = statistics.median(x)
    try:
        mode = statistics.mode(x)
    except:
        mode = "N/A"
    stdev = statistics.stdev(x)
    summary_fp = open('./results/route_summary.txt', 'w')
    summary_fp.write(f"---- routes per county ----\n")
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
    summary_fp.write("\n")
    d_view = [ (k,v) for k,v in route_count_dict.items() ]
    d_view.sort(reverse=True) # natively sort tuples by first element
    for route, count in d_view:
        summary_fp.write(f"{route}: {count}\n")

def create_subplot(result_files, title, ax):
    labels = ['EoC != 0.5 &\nPOP == 0',
              'EoC == 0.5 &\nPOP == 0',
              'EoC != 0.5 &\nPOP > 0',
              'EoC == 0.5 &\nPOP > 0']
    proportions = dict()
    for h in result_files:
        result_file = result_files[h]
        df = pd.read_csv(result_file, sep='|')
        df['POP10'] = df['POP10'].fillna(0.0)
        denom = float(len(df))
        prop0 = int(100 * len(df[(df['verizon_eoc'] != 0.5) & (df['POP10'] == 0.0)]) / denom)
        prop1 = int(100 * len(df[(df['verizon_eoc'] == 0.5) & (df['POP10'] == 0.0)]) / denom)
        prop2 = int(100 * len(df[(df['verizon_eoc'] != 0.5) & (df['POP10'] > 0.0)]) / denom)
        prop3 = int(100 * len(df[(df['verizon_eoc'] == 0.5) & (df['POP10'] > 0.0)]) / denom)
        proportions[str(h).replace('Utility.', '')] = [prop0, prop1, prop2, prop3]
    x = np.arange(len(labels))
    width= 0.1 # width of the bars
    ax.bar(x - 2*width, proportions['UNIQUE_BLOCKS'], width, label='UNIQUE_BLOCKS')
    ax.bar(x - width, proportions['POPULATED_BLOCKS'], width, label='POPULATED_BLOCKS')
    ax.bar(x, proportions['POPULATED_BLOCKS_W_CONFLICT'], width, label='POPULATED_BLOCKS_W_CONFLICT')
    ax.bar(x + width, proportions['SCALING_BY_LEVEL'], width, label='SCALING_BY_LEVEL')
    ax.bar(x + 2*width, proportions['SCALING_BY_LEVEL_W_CONFLICT'], width, label='SCALING_BY_LEVEL_W_CONFLICT')
    ax.set_ylabel('Percentage')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    return proportions

def create_aggregate_bar_chart(n, use_percent=False):
    nm_county_fips = ['011', '035', '003', '059', '047', '055', '017', '007', '043', '006', '013', '021', '023', '053', '028', '033', '015', '009', '041', '045', '027', '019', '057', '029', '031', '039', '025', '005', '049', '037', '001', '051', '061']
    c = 7
    r = 5
    fig1, axes = plt.subplots(ncols=c, nrows=r, constrained_layout=True)
    axes = list(axes.flatten())
    fig1.set_size_inches(35, 20)
    prop_dict_list = []
    df = pd.DataFrame()
    for i in range(c*r):
        if i != 0 and i < len(nm_county_fips) + 1:
            fips = nm_county_fips[i-1]
            result_files = dict()
            for h in Utility:
                # ex file: ./results/by_county/5/011_Utility.UNIQUE_BLOCKS_grid.csv
                result_file = f"./results/by_county/{n}/{fips}_{h}_grid.csv"
                result_files[h] = result_file
            prop_dict = create_subplot(result_files, f"FIPS:{fips}", axes[i])
            prop_dict_list.append(prop_dict)
        else:
            axes[i].axis('off')
    handles, labels = axes[1].get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='upper left', prop={'size': 18})
    plt.suptitle("Block Category Percentages For Each County (N=1)")
    if use_percent:
        out_file1 = f"./plots/by_county/bar_charts_n={n}_percentage.png"
    else:
        out_file1 = f"./plots/by_county/bar_charts_n={n}_number.png"
    plt.savefig(out_file1)
    plt.clf()
    print(f"wrote: {out_file1}")
    # now instead of making a subplot for each county, aggregate
    final_props = dict()
    for h in Utility:
        avg0 = avg1 = avg2 = avg3 = 0.0
        for i in range(len(prop_dict_list)):
            avg0 += prop_dict_list[i][str(h).replace('Utility.', '')][0]
            avg1 += prop_dict_list[i][str(h).replace('Utility.', '')][1]
            avg2 += prop_dict_list[i][str(h).replace('Utility.', '')][2]
            avg3 += prop_dict_list[i][str(h).replace('Utility.', '')][3]
        avg0 /= len(prop_dict_list)
        avg1 /= len(prop_dict_list)
        avg2 /= len(prop_dict_list)
        avg3 /= len(prop_dict_list)
        final_props[str(h).replace('Utility.', '')] = [avg0, avg1, avg2, avg3]
    width = 0.05 # width of the bars
    labels = ['EoC != 0.5 &\nPOP == 0',
              'EoC == 0.5 &\nPOP == 0',
              'EoC != 0.5 &\nPOP > 0',
              'EoC == 0.5 &\nPOP > 0']
    x = np.arange(len(labels))
    plt.figure(figsize=(8,4))
    plt.bar(x - 2*width, final_props['UNIQUE_BLOCKS'], width, label='UNIQUE_BLOCKS')
    plt.bar(x - width, final_props['POPULATED_BLOCKS'], width, label='POPULATED_BLOCKS')
    plt.bar(x, final_props['POPULATED_BLOCKS_W_CONFLICT'], width, label='POPULATED_BLOCKS_W_CONFLICT')
    plt.bar(x + width, final_props['SCALING_BY_LEVEL'], width, label='SCALING_BY_LEVEL')
    plt.bar(x + 2*width, final_props['SCALING_BY_LEVEL_W_CONFLICT'], width, label='SCALING_BY_LEVEL_W_CONFLICT')
    plt.ylabel('Percentage')
    plt.title("Block Category Percentages For Each County (N=1)")
    plt.xticks(x, labels)
    plt.tight_layout()
    plt.legend()
    plt.savefig(out_file2)
    plt.clf()
    print(f"wrote: {out_file2}")

# result_files should be a dictionary
# result_files[func] = result_file
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
        if denom is None:
            denom = float(len(df))
        prop0 = int(100 * len(df[(df['verizon_eoc'] != 0.5) & (df['POP10'] == 0.0)]) / denom)
        prop1 = int(100 * len(df[(df['verizon_eoc'] == 0.5) & (df['POP10'] == 0.0)]) / denom)
        prop2 = int(100 * len(df[(df['verizon_eoc'] != 0.5) & (df['POP10'] > 0.0)]) / denom)
        prop3 = int(100 * len(df[(df['verizon_eoc'] == 0.5) & (df['POP10'] > 0.0)]) / denom)
        proportions[str(h).replace('Utility.', '')] = [prop0, prop1, prop2, prop3]
    x = np.arange(len(labels))
    width= 0.1 # width of the bars
    fig, ax = plt.subplots()
    ax.bar(x - 2*width, proportions['UNIQUE_BLOCKS'], width, label='UNIQUE_BLOCKS')
    ax.bar(x - width, proportions['POPULATED_BLOCKS'], width, label='POPULATED_BLOCKS')
    ax.bar(x, proportions['POPULATED_BLOCKS_W_CONFLICT'], width, label='POPULATED_BLOCKS_W_CONFLICT')
    ax.bar(x + width, proportions['SCALING_BY_LEVEL'], width, label='SCALING_BY_LEVEL')
    ax.bar(x + 2*width, proportions['SCALING_BY_LEVEL_W_CONFLICT'], width, label='SCALING_BY_LEVEL_W_CONFLICT')
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
    y = np.arange(len(x)) / (len(x)-1)
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

def create_kendall_tau_matrix():
    nm_county_fips = ['011', '035', '003', '059', '047', '055', '017', '007', '043', '006', '013', '021', '023', '053', '028', '033', '015', '009', '041', '045', '027', '019', '057', '029', '031', '039', '025', '005', '049', '037', '001', '051', '061']
    cellText = []
    df = pd.DataFrame(columns=['County FIPS', 'U1 x U2', 'U1 x U3', 'U1 x U4',
                               'U1 x U5', 'U1 x U6', 'U1 x U7', 'U2 x U3',
                               'U2 x U4', 'U5 x U6', 'U5 x U7'])
    for fips in nm_county_fips:
        row = []
        # format: util_dict['UX'] = [selected_routes]
        util_dict = dict()
        count = 0
        for util in Utility:
            lines = open(f'./results/by_county/-1/{fips}_{util}_route_ids.txt', 'r').readlines()
            lines = [line.strip() for line in lines]
            count += 1
            util_dict[f"U{count}"] = lines
        row.append(fips)
        row.append(perform_kendalltau(util_dict['U1'], util_dict['U2'])[0])
        row.append(perform_kendalltau(util_dict['U1'], util_dict['U3'])[0])
        row.append(perform_kendalltau(util_dict['U1'], util_dict['U4'])[0])
        row.append(perform_kendalltau(util_dict['U1'], util_dict['U5'])[0])
        row.append(perform_kendalltau(util_dict['U1'], util_dict['U6'])[0])
        row.append(perform_kendalltau(util_dict['U1'], util_dict['U7'])[0])
        row.append(perform_kendalltau(util_dict['U2'], util_dict['U3'])[0])
        row.append(perform_kendalltau(util_dict['U2'], util_dict['U4'])[0])
        row.append(perform_kendalltau(util_dict['U5'], util_dict['U6'])[0])
        row.append(perform_kendalltau(util_dict['U5'], util_dict['U7'])[0])
        df.loc[len(df)] = row
    # include the state
    for util in Utility:
        # file ex: ./results/state_wide/132/Utility.UNIQUE_BLOCKS_route_ids.txt
        lines = open(f'./results/state_wide/-1/{util}_route_ids.txt', 'r').readlines()
        lines = [line.strip() for line in lines]
        count += 1
        util_dict[f"U{count}"] = lines
    row = []
    row.append("STATE-WIDE")
    row.append(perform_kendalltau(util_dict['U1'], util_dict['U2'])[0])
    row.append(perform_kendalltau(util_dict['U1'], util_dict['U3'])[0])
    row.append(perform_kendalltau(util_dict['U1'], util_dict['U4'])[0])
    row.append(perform_kendalltau(util_dict['U1'], util_dict['U5'])[0])
    row.append(perform_kendalltau(util_dict['U1'], util_dict['U6'])[0])
    row.append(perform_kendalltau(util_dict['U1'], util_dict['U7'])[0])
    row.append(perform_kendalltau(util_dict['U2'], util_dict['U3'])[0])
    row.append(perform_kendalltau(util_dict['U2'], util_dict['U4'])[0])
    row.append(perform_kendalltau(util_dict['U5'], util_dict['U6'])[0])
    row.append(perform_kendalltau(util_dict['U5'], util_dict['U7'])[0])
    df.loc[len(df)] = row
    # df = pd.DataFrame(cellText, columns=colLabels)
    df = df.set_index('County FIPS')
    fig(figsize=(10, 10))
    sns.heatmap(df, annot=True, cmap='Greys')
    plt.title('Kendall Tau Heatmap')
    plt.savefig('./plots/by_county/table.png')
    plt.clf()

def jaccard(list1, list2):
    numerator = float(len(set(list1).intersection(list2)))
    denominator = float(len(set(list1).union(list2)))
    return numerator / denominator

# only considering U1-U4 for now
def create_jaccard_similarity_matrix():
    nm_county_fips = ['011', '035', '003', '059', '047', '055', '017', '007', '043', '006', '013', '021', '023', '053', '028', '033', '015', '009', '041', '045', '027', '019', '057', '029', '031', '039', '025', '005', '049', '037', '001', '051', '061']
    cellText = []
    n = "10%"
    df = pd.DataFrame(columns=['County FIPS', 'U1 x U2', 'U1 x U3', 'U1 x U4', 'U2 x U3', 'U2 x U4', 'U3 x U4'])
    for fips in nm_county_fips:
        row = []
        # format: util_dict['UX'] = [selected_routes]
        util_dict = dict()
        count = 0
        for util in Utility:
            if util not in {Utility.NAIVE, Utility.POPULATED_BLOCKS, Utility.POPULATED_BLOCKS_W_CONFLICT, Utility.POPULATED_BLOCKS_W_SCALED_CONFLICT}:
                continue
            lines = open(f'./results/by_county/{n}/{fips}_{util}_route_ids.txt', 'r').readlines()
            lines = [line.strip() for line in lines]
            count += 1
            util_dict[f"U{count}"] = lines
        row.append(fips)
        row.append(jaccard(util_dict['U1'], util_dict['U2']))
        row.append(jaccard(util_dict['U1'], util_dict['U3']))
        row.append(jaccard(util_dict['U1'], util_dict['U4']))
        row.append(jaccard(util_dict['U2'], util_dict['U3']))
        row.append(jaccard(util_dict['U2'], util_dict['U4']))
        row.append(jaccard(util_dict['U3'], util_dict['U4']))
        df.loc[len(df)] = row
    # include the state
    for util in Utility:
        # file ex: ./results/state_wide/132/Utility.UNIQUE_BLOCKS_route_ids.txt
        if util not in {Utility.NAIVE, Utility.POPULATED_BLOCKS, Utility.POPULATED_BLOCKS_W_CONFLICT, Utility.POPULATED_BLOCKS_W_SCALED_CONFLICT}:
            continue
        lines = open(f'./results/state_wide/{n}/{util}_route_ids.txt', 'r').readlines()
        lines = [line.strip() for line in lines]
        count += 1
        util_dict[f"U{count}"] = lines
    row = []
    row.append("STATE-WIDE")
    row.append(jaccard(util_dict['U1'], util_dict['U2']))
    row.append(jaccard(util_dict['U1'], util_dict['U3']))
    row.append(jaccard(util_dict['U1'], util_dict['U4']))
    row.append(jaccard(util_dict['U2'], util_dict['U3']))
    row.append(jaccard(util_dict['U2'], util_dict['U4']))
    row.append(jaccard(util_dict['U3'], util_dict['U4']))
    df.loc[len(df)] = row
    # df = pd.DataFrame(cellText, columns=colLabels)
    df = df.set_index('County FIPS')
    fig(figsize=(12, 10))
    sns.heatmap(df, annot=True, cmap='Greys')
    plt.title(f'Jaccard Heatmap (n={n})')
    plot_file = f'./plots/jaccard_heatmap.png'
    plt.savefig(plot_file)
    print(f"wrote: {plot_file}")
    plt.clf()

def create_by_county_box_and_whiskers_plots(n, use_percent, use_scaled_conflict):
    my_dict = dict()
    nm_county_fips = ['011', '035', '003', '059', '047', '055', '017', '007', '043', '006', '013', '021', '023', '053', '028', '033', '015', '009', '041', '045', '027', '019', '057', '029', '031', '039', '025', '005', '049', '037', '001', '051', '061']
    for util in Utility:
        util_name = str(util).replace('Utility.', '')
        values = []
        for fips in nm_county_fips:
            # ex file: ./results/by_county/5/011_Utility.UNIQUE_BLOCKS_grid.csv
            df = pd.read_csv(f'./results/by_county/{n}/{fips}_{util}_grid.csv', sep='|')
            if use_scaled_conflict:
                values.append(len(df[
                    (df['POP10'] > 0.0) & \
                    ((df['verizon_eoc'] == 0.5) | \
                     (df['att_eoc'] == 0.5) | \
                     (df['tmobile_eoc'] == 0.5) | \
                     (df['sprint_eoc'] == 0.5))
                ]))
            else:
                values.append(len(df[(df['POP10'] > 0.0) & (df['verizon_eoc'] == 0.5)]))
        my_dict[f"U{util.value}"] = values
    if use_percent:
        index = 0
        for fips in nm_county_fips:
            df = pd.read_csv(f'./results/by_county/-1/{fips}_Utility.UNIQUE_BLOCKS_grid.csv', sep='|')
            if use_scaled_conflict:
                denom = len(df[
                    (df['POP10'] > 0.0) & \
                    ((df['verizon_eoc'] == 0.5) | \
                     (df['att_eoc'] == 0.5) | \
                     (df['tmobile_eoc'] == 0.5) | \
                     (df['sprint_eoc'] == 0.5))
                ])
            else:
                denom = len(df[(df['POP10'] > 0.0) & (df['verizon_eoc'] == 0.5)])
            for key in my_dict:
                if my_dict[key][index] == 0 and denom == 0:
                    my_dict[key][index] = 0
                else:
                    my_dict[key][index] /= denom
            index += 1
    fig, ax = plt.subplots()
    ax.boxplot(my_dict.values())
    ax.set_xticklabels(my_dict.keys())
    if use_percent:
        if use_scaled_conflict:
            plt.title(f'Percentage of Populated Conflict blocks by County (n = {n}, all carriers)')
            fname = f'./plots/by_county/box_and_whiskers_n={n}_percentage_all.png'
        else:
            plt.title(f'Percentage of Populated Conflict blocks by County (n = {n}, verizon only)')
            fname = f'./plots/by_county/box_and_whiskers_n={n}_percentage_verizon.png'
        ax.set_ylabel('Percentage')
    plt.savefig(fname)
    print(f"wrote: {fname}")
    plt.clf()

# will create four figures:
# box plot showing the ROUTE %-age of the four main categories (by-county)
# pie chart showing the ROUTE %-age of the four main categories (state-wide)
# box plot showing the REGION %-age of the four main categories (by-county)
# pie chart showing the REGION %-age of the four main categories (state-wide)
def create_category_plots():
    nm_county_fips = ['011', '035', '003', '059', '047', '055', '017', '007', '043', '006', '013', '021', '023', '053', '028', '033', '015', '009', '041', '045', '027', '019', '057', '029', '031', '039', '025', '005', '049', '037', '001', '051', '061']
    util = Utility.UNIQUE_BLOCKS
    n = -1
    # plot 1/4
    my_dict = dict()
    my_dict['EoC != 0.5 &\nPOP == 0'] = []
    my_dict['EoC == 0.5 &\nPOP == 0'] = []
    my_dict['EoC != 0.5 &\nPOP > 0'] = []
    my_dict['EoC == 0.5 &\nPOP > 0'] = []
    for fips in nm_county_fips:
        df = pd.read_csv(f'./results/by_county/{n}/{fips}_{util}_grid.csv', sep='|')
        my_dict['EoC != 0.5 &\nPOP == 0'].append(
            100 * len(df[(
                (df['verizon_eoc'] != 0.5) & \
                (df['tmobile_eoc'] != 0.5) & \
                (df['att_eoc'] != 0.5) & \
                (df['sprint_eoc'] != 0.5)) & (df['POP10'] == 0.0)]) / len(df)
        )
        my_dict['EoC == 0.5 &\nPOP == 0'].append(
            100 * len(df[(
                (df['verizon_eoc'] == 0.5) | \
                (df['tmobile_eoc'] == 0.5) | \
                (df['att_eoc'] == 0.5) | \
                (df['sprint_eoc'] == 0.5)) & (df['POP10'] == 0.0)]) / len(df)
        )
        my_dict['EoC != 0.5 &\nPOP > 0'].append(
            100 * len(df[(
                (df['verizon_eoc'] != 0.5) & \
                (df['tmobile_eoc'] != 0.5) & \
                (df['att_eoc'] != 0.5) & \
                (df['sprint_eoc'] != 0.5)) & (df['POP10'] > 0.0)]) / len(df)
        )
        my_dict['EoC == 0.5 &\nPOP > 0'].append(
            100 * len(df[(
                (df['verizon_eoc'] == 0.5) | \
                (df['tmobile_eoc'] == 0.5) | \
                (df['att_eoc'] == 0.5) | \
                (df['sprint_eoc'] == 0.5)) & (df['POP10'] > 0.0)]) / len(df)
        )
    fig, ax = plt.subplots()
    ax.boxplot(my_dict.values())
    ax.set_xticklabels(my_dict.keys())
    plt.title(f'Route Category Percentages (by county)')
    plt.ylabel("Percentage")
    plot_file = "./plots/route_category_percentages_by_county.png"
    plt.savefig(plot_file)
    print(f"wrote: {plot_file}")
    plt.clf()
    # plot 2/4
    my_dict = dict()
    df = pd.read_csv(f'./results/state_wide/{n}/{util}_grid.csv', sep='|')
    my_dict['EoC != 0.5 & POP == 0'] = len(df[(
        (df['verizon_eoc'] != 0.5) & \
        (df['tmobile_eoc'] != 0.5) & \
        (df['att_eoc'] != 0.5) & \
        (df['sprint_eoc'] != 0.5)) & (df['POP10'] == 0.0)])
    my_dict['EoC == 0.5 & POP == 0'] = len(df[(
        (df['verizon_eoc'] == 0.5) | \
        (df['tmobile_eoc'] == 0.5) | \
        (df['att_eoc'] == 0.5) | \
        (df['sprint_eoc'] == 0.5)) & (df['POP10'] == 0.0)])
    my_dict['EoC != 0.5 & POP > 0'] = len(df[(
        (df['verizon_eoc'] != 0.5) & \
        (df['tmobile_eoc'] != 0.5) & \
        (df['att_eoc'] != 0.5) & \
        (df['sprint_eoc'] != 0.5)) & (df['POP10'] > 0.0)])
    my_dict['EoC == 0.5 & POP > 0'] = len(df[(
        (df['verizon_eoc'] == 0.5) | \
        (df['tmobile_eoc'] == 0.5) | \
        (df['att_eoc'] == 0.5) | \
        (df['sprint_eoc'] == 0.5)) & (df['POP10'] > 0.0)])
    plt.pie(my_dict.values(), labels=my_dict.keys(), autopct='%1.1f%%')
    plot_file = "./plots/route_category_percentages_state_wide.png"
    plt.title(f'Route Category Percentages (state wide)')
    plt.savefig(plot_file)
    print(f"wrote: {plot_file}")
    plt.clf()
    # plot 3/4
    columns = ["id", "grid_polygon", "verizon_sh", "tmobile_sh", "att_sh", "sprint_sh", "verizon_fcc", "tmobile_fcc", "att_fcc", "sprint_fcc", "verizon_eoc", "tmobile_eoc", "att_eoc", "sprint_eoc", "UR10", "b_type", "HOUSING10", "POP10"]
    my_dict = dict()
    my_dict['EoC != 0.5 &\nPOP == 0'] = []
    my_dict['EoC == 0.5 &\nPOP == 0'] = []
    my_dict['EoC != 0.5 &\nPOP > 0'] = []
    my_dict['EoC == 0.5 &\nPOP > 0'] = []
    for fips in nm_county_fips:
        df1 = pd.read_csv(f'./results/by_county/{n}/{fips}_{util}_grid.csv', sep='|')
        df1 = df1[columns]
        df2 = pd.read_csv(f'./data/by_county/{fips}.csv', sep='|')
        df2 = df2[columns]
        df = pd.concat([df1, df2], ignore_index=True)
        df = df.drop_duplicates(subset=['id'], ignore_index=True)
        my_dict['EoC != 0.5 &\nPOP == 0'].append(
            100 * len(df[(
                (df['verizon_eoc'] != 0.5) & \
                (df['tmobile_eoc'] != 0.5) & \
                (df['att_eoc'] != 0.5) & \
                (df['sprint_eoc'] != 0.5)) & (df['POP10'] == 0.0)]) / len(df)
        )
        my_dict['EoC == 0.5 &\nPOP == 0'].append(
            100 * len(df[(
                (df['verizon_eoc'] == 0.5) | \
                (df['tmobile_eoc'] == 0.5) | \
                (df['att_eoc'] == 0.5) | \
                (df['sprint_eoc'] == 0.5)) & (df['POP10'] == 0.0)]) / len(df)
        )
        my_dict['EoC != 0.5 &\nPOP > 0'].append(
            100 * len(df[(
                (df['verizon_eoc'] != 0.5) & \
                (df['tmobile_eoc'] != 0.5) & \
                (df['att_eoc'] != 0.5) & \
                (df['sprint_eoc'] != 0.5)) & (df['POP10'] > 0.0)]) / len(df)
        )
        my_dict['EoC == 0.5 &\nPOP > 0'].append(
            100 * len(df[(
                (df['verizon_eoc'] == 0.5) | \
                (df['tmobile_eoc'] == 0.5) | \
                (df['att_eoc'] == 0.5) | \
                (df['sprint_eoc'] == 0.5)) & (df['POP10'] > 0.0)]) / len(df)
        )    
    fig, ax = plt.subplots()
    ax.boxplot(my_dict.values())
    ax.set_xticklabels(my_dict.keys())
    plt.title(f'Region Category Percentages (by county)')
    plt.ylabel("Percentage")
    plot_file = "./plots/region_category_percentages_by_county.png"
    plt.savefig(plot_file)
    print(f"wrote: {plot_file}")
    plt.clf()
    # plot 4/4
    my_dict = dict()
    df1 = pd.read_csv(f'./results/state_wide/{n}/{util}_grid.csv', sep='|')
    df1 = df1[columns]
    df2 = pd.read_csv(f'./data/grid_1km_sh_fcc_eoc_voc.csv', sep='|')
    df2 = df2[columns]
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.drop_duplicates(subset=['id'], ignore_index=True)
    my_dict['EoC != 0.5 & POP == 0'] = len(df[(
        (df['verizon_eoc'] != 0.5) & \
        (df['tmobile_eoc'] != 0.5) & \
        (df['att_eoc'] != 0.5) & \
        (df['sprint_eoc'] != 0.5)) & (df['POP10'] == 0.0)])
    my_dict['EoC == 0.5 & POP == 0'] = len(df[(
        (df['verizon_eoc'] == 0.5) | \
        (df['tmobile_eoc'] == 0.5) | \
        (df['att_eoc'] == 0.5) | \
        (df['sprint_eoc'] == 0.5)) & (df['POP10'] == 0.0)])
    my_dict['EoC != 0.5 & POP > 0'] = len(df[(
        (df['verizon_eoc'] != 0.5) & \
        (df['tmobile_eoc'] != 0.5) & \
        (df['att_eoc'] != 0.5) & \
        (df['sprint_eoc'] != 0.5)) & (df['POP10'] > 0.0)])
    my_dict['EoC == 0.5 & POP > 0'] = len(df[(
        (df['verizon_eoc'] == 0.5) | \
        (df['tmobile_eoc'] == 0.5) | \
        (df['att_eoc'] == 0.5) | \
        (df['sprint_eoc'] == 0.5)) & (df['POP10'] > 0.0)])
    plt.pie(my_dict.values(), labels=my_dict.keys(), autopct='%1.1f%%')
    plot_file = "./plots/region_category_percentages_state_wide.png"
    plt.title(f'Region Category Percentages (state wide)')
    plt.savefig(plot_file)
    print(f"wrote: {plot_file}")
    plt.clf()

# these are state-wide plots
def create_line_plots():
    n = -1
    my_utilities = [Utility.UNIQUE_BLOCKS,
                    Utility.POPULATED_BLOCKS,
                    Utility.POPULATED_BLOCKS_W_CONFLICT]
    my_dict = {Utility.UNIQUE_BLOCKS: [],
               Utility.POPULATED_BLOCKS: [],
               Utility.POPULATED_BLOCKS_W_CONFLICT: []}
    for util in my_utilities:
        grid_df = pd.read_csv(f'./results/state_wide/{n}/{util}_grid.csv', sep='|')
        route_ids = open(f'./results/state_wide/{n}/{util}_route_ids.txt', 'r').readlines()
        route_ids = [route_id.strip() for route_id in route_ids]
        cumulative_df = pd.DataFrame(columns=grid_df.columns)
        count = 0
        denom = len(grid_df[(
            (grid_df['verizon_eoc'] == 0.5) | \
            (grid_df['tmobile_eoc'] == 0.5) | \
            (grid_df['att_eoc'] == 0.5) | \
            (grid_df['sprint_eoc'] == 0.5)) & (grid_df['POP10'] > 0.0)])
        for route_id in route_ids:
            count += 1
            my_slice = grid_df[grid_df['route_ids'].str.contains(route_id)]
            cumulative_df = pd.concat([cumulative_df, my_slice], ignore_index=True)
            cumulative_df = cumulative_df.drop_duplicates(subset=['id'], ignore_index=True)
            my_dict[util].append(100 * len(cumulative_df[(
                (cumulative_df['verizon_eoc'] == 0.5) | \
                (cumulative_df['tmobile_eoc'] == 0.5) | \
                (cumulative_df['att_eoc'] == 0.5) | \
                (cumulative_df['sprint_eoc'] == 0.5)) & \
                (cumulative_df['POP10'] > 0.0)]) / denom
            )
            if count % 100 == 0:
                print(f"{util}: iteration {count}")
            if count == 100:
                break
        print(f"done with {util}")
    x_axis = [i for i in range(1, 101)]
    print("plotting...")
    plt.plot(x_axis, my_dict[Utility.UNIQUE_BLOCKS], label="UNIQUE_BLOCKS")
    plt.plot(x_axis, my_dict[Utility.POPULATED_BLOCKS], label="POPULATED_BLOCKS")
    plt.plot(x_axis, my_dict[Utility.POPULATED_BLOCKS_W_CONFLICT], label="POPULATED_BLOCKS_W_CONFLICT")
    plt.xlabel('Number of Selected Routes')
    plt.ylabel('Percentage of Covered Populated Conflict Blocks')
    plot_file = "./plots/line_plot_routes_vs_percentage.png"
    plt.title(f'Number of Routes vs Populated Conflict Percentage (state wide)')
    plt.legend()
    plt.savefig(plot_file)
    print(f"wrote: {plot_file}")

# plots histograms across utilities (U1-U4) for the state-wide case
def plot_utility_distributions():
    distributions = dict()

    # U1
    fp = open('./results/state_wide/-1/Utility.NAIVE_utility_scores.txt', 'r')
    utility_dict = literal_eval(fp.readline())
    utility_scores = list(utility_dict.values())
    distributions['U1'] = utility_scores

    # U2
    fp = open('./results/state_wide/-1/Utility.POPULATED_BLOCKS_utility_scores.txt', 'r')
    utility_dict = literal_eval(fp.readline())
    utility_scores = list(utility_dict.values())
    distributions['U2'] = utility_scores

    # U3
    fp = open('./results/state_wide/-1/Utility.POPULATED_BLOCKS_W_CONFLICT_utility_scores.txt', 'r')
    utility_dict = literal_eval(fp.readline())
    utility_scores = list(utility_dict.values())
    distributions['U3'] = utility_scores    

    # U4
    fp = open('./results/state_wide/-1/Utility.POPULATED_BLOCKS_W_SCALED_CONFLICT_utility_scores.txt', 'r')
    utility_dict = literal_eval(fp.readline())
    utility_scores = list(utility_dict.values())
    distributions['U4'] = utility_scores
    
    plt.hist(distributions['U1'], 50, alpha=0.6, log=True, label="U1", histtype='stepfilled')
    plt.hist(distributions['U2'], 50, alpha=0.6, log=True, label="U2", histtype='stepfilled')
    plt.hist(distributions['U3'], 50, alpha=0.6, log=True, label="U3", histtype='stepfilled')
    plt.hist(distributions['U4'], 50, alpha=0.6, log=True, label="U4", histtype='stepfilled')
    plt.legend()
    plot_file = "./plots/utility_distributions.png"
    plt.savefig(plot_file)
    print(f"wrote: {plot_file}")

def pearson1(mylist):
    return (np.mean(mylist) - scipy.stats.mode(mylist).mode[0]) / np.std(mylist)

def pearson2(mylist):
    return 3.0*(np.mean(mylist) - np.median(mylist)) / np.std(mylist)

# plots U4 histograms across low/med/high JS values
# also prints skewness metrics
def plot_js_distributions():
    nm_county_fips = ['011', '035', '003', '059', '047', '055', '017', '007', '043', '006', '013', '021', '023', '053', '028', '033', '015', '009', '041', '045', '027', '019', '057', '029', '031', '039', '025', '005', '049', '037', '001', '051', '061']
    n = "10%"
    county_js_dict = dict()
    county_util_scores = dict()
    for fips in nm_county_fips:
        row = []
        # format: util_dict['UX'] = [selected_routes]
        util_dict = dict()
        count = 0
        for util in Utility:
            if util not in {Utility.NAIVE, Utility.POPULATED_BLOCKS, Utility.POPULATED_BLOCKS_W_CONFLICT, Utility.POPULATED_BLOCKS_W_SCALED_CONFLICT}:
                continue
            lines = open(f'./results/by_county/{n}/{fips}_{util}_route_ids.txt', 'r').readlines()
            lines = [line.strip() for line in lines]
            count += 1
            util_dict[f"U{count}"] = lines
        county_js_dict[fips] = 0
        county_js_dict[fips] += jaccard(util_dict['U1'], util_dict['U2'])
        county_js_dict[fips] += jaccard(util_dict['U1'], util_dict['U3'])
        county_js_dict[fips] += jaccard(util_dict['U1'], util_dict['U4'])
        county_js_dict[fips] += jaccard(util_dict['U2'], util_dict['U3'])
        county_js_dict[fips] += jaccard(util_dict['U2'], util_dict['U4'])
        county_js_dict[fips] += jaccard(util_dict['U3'], util_dict['U4'])

        fp = open(f'./results/by_county/-1/{fips}_Utility.POPULATED_BLOCKS_W_SCALED_CONFLICT_utility_scores.txt', 'r')
        utility_dict = literal_eval(fp.readline())
        utility_scores = list(utility_dict.values())
        county_util_scores[fips] = utility_scores

    mysorted = sorted(county_js_dict.items(), key=lambda x: x[1])
    low_js_fips = set([fips for (fips, js_sum) in mysorted][0:11])
    med_js_fips = set([fips for (fips, js_sum) in mysorted][11:22])
    high_js_fips = set([fips for (fips, js_sum) in mysorted][22:])

    low_js_util_scores = []
    med_js_util_scores = []
    high_js_util_scores = []
    for (fips, util_scores) in county_util_scores.items():
        if fips in low_js_fips:
            low_js_util_scores += util_scores
        elif fips in med_js_fips:
            med_js_util_scores += util_scores
        elif fips in high_js_fips:
            high_js_util_scores += util_scores

    # display skewness coefficients
    print(f"low js Fisher-Pearson skew: {scipy.stats.skew(low_js_util_scores)}")
    print(f"medium js Fisher-Pearson skew: {scipy.stats.skew(med_js_util_scores)}")
    print(f"high js Fisher-Pearson skew: {scipy.stats.skew(high_js_util_scores)}")
    print("")
    print(f"low js Pearson's First skewness coeff: {pearson1(low_js_util_scores)}")
    print(f"med js Pearson's First skewness coeff: {pearson1(med_js_util_scores)}")
    print(f"high js Pearson's First skewness coeff: {pearson1(high_js_util_scores)}")
    print("")
    print(f"low js Pearson's Second skewness coeff: {pearson2(low_js_util_scores)}")
    print(f"med js Pearson's Second skewness coeff: {pearson2(med_js_util_scores)}")
    print(f"high js Pearson's Second skewness coeff: {pearson2(high_js_util_scores)}")
    print("")

    plt.hist(low_js_util_scores, 20, alpha=0.7, log=True, label="low JS", histtype='stepfilled', color="#e41a1c")
    plt.hist(med_js_util_scores, 20, alpha=0.7, log=True, label="medium JS", histtype='stepfilled', color="#377eb8")
    plt.hist(high_js_util_scores, 20, alpha=0.7, log=True, label="high JS", histtype='stepfilled', color="#4daf4a")
    plt.legend()
    plot_file = "./plots/js_distributions.png"
    plt.savefig(plot_file)
    print(f"wrote: {plot_file}")

# plots histograms of route length across low/med/high JS values
# also prints skewness metrics
def plot_js_length_distributions():
    nm_county_fips = ['011', '035', '003', '059', '047', '055', '017', '007', '043', '006', '013', '021', '023', '053', '028', '033', '015', '009', '041', '045', '027', '019', '057', '029', '031', '039', '025', '005', '049', '037', '001', '051', '061']
    n = "10%"
    county_js_dict = dict()
    county_util_scores = dict()
    for fips in nm_county_fips:
        row = []
        # format: util_dict['UX'] = [selected_routes]
        util_dict = dict()
        count = 0
        for util in Utility:
            if util not in {Utility.NAIVE, Utility.POPULATED_BLOCKS, Utility.POPULATED_BLOCKS_W_CONFLICT, Utility.POPULATED_BLOCKS_W_SCALED_CONFLICT}:
                continue
            lines = open(f'./results/by_county/{n}/{fips}_{util}_route_ids.txt', 'r').readlines()
            lines = [line.strip() for line in lines]
            count += 1
            util_dict[f"U{count}"] = lines
        county_js_dict[fips] = 0
        county_js_dict[fips] += jaccard(util_dict['U1'], util_dict['U2'])
        county_js_dict[fips] += jaccard(util_dict['U1'], util_dict['U3'])
        county_js_dict[fips] += jaccard(util_dict['U1'], util_dict['U4'])
        county_js_dict[fips] += jaccard(util_dict['U2'], util_dict['U3'])
        county_js_dict[fips] += jaccard(util_dict['U2'], util_dict['U4'])
        county_js_dict[fips] += jaccard(util_dict['U3'], util_dict['U4'])

        fp = open(f'./results/by_county/-1/{fips}_Utility.POPULATED_BLOCKS_W_SCALED_CONFLICT_utility_scores.txt', 'r')
        utility_dict = literal_eval(fp.readline())
        utility_scores = list(utility_dict.values())
        county_util_scores[fips] = utility_scores

    mysorted = sorted(county_js_dict.items(), key=lambda x: x[1])
    low_js_fips = set([fips for (fips, js_sum) in mysorted][0:11])
    med_js_fips = set([fips for (fips, js_sum) in mysorted][11:22])
    high_js_fips = set([fips for (fips, js_sum) in mysorted][22:])

    low_js_util_scores = []
    med_js_util_scores = []
    high_js_util_scores = []
    for (fips, util_scores) in county_util_scores.items():
        if fips in low_js_fips:
            low_js_util_scores += util_scores
        elif fips in med_js_fips:
            med_js_util_scores += util_scores
        elif fips in high_js_fips:
            high_js_util_scores += util_scores

    # display skewness coefficients
    print(f"low js Fisher-Pearson skew: {scipy.stats.skew(low_js_util_scores)}")
    print(f"medium js Fisher-Pearson skew: {scipy.stats.skew(med_js_util_scores)}")
    print(f"high js Fisher-Pearson skew: {scipy.stats.skew(high_js_util_scores)}")
    print("")
    print(f"low js Pearson's First skewness coeff: {pearson1(low_js_util_scores)}")
    print(f"med js Pearson's First skewness coeff: {pearson1(med_js_util_scores)}")
    print(f"high js Pearson's First skewness coeff: {pearson1(high_js_util_scores)}")
    print("")
    print(f"low js Pearson's Second skewness coeff: {pearson2(low_js_util_scores)}")
    print(f"med js Pearson's Second skewness coeff: {pearson2(med_js_util_scores)}")
    print(f"high js Pearson's Second skewness coeff: {pearson2(high_js_util_scores)}")
    print("")

    plt.hist(low_js_util_scores, 20, alpha=0.7, log=True, label="low JS", histtype='stepfilled', color="#e41a1c")
    plt.hist(med_js_util_scores, 20, alpha=0.7, log=True, label="medium JS", histtype='stepfilled', color="#377eb8")
    plt.hist(high_js_util_scores, 20, alpha=0.7, log=True, label="high JS", histtype='stepfilled', color="#4daf4a")
    plt.legend()
    plot_file = "./plots/js_distributions.png"
    plt.savefig(plot_file)
    print(f"wrote: {plot_file}")
