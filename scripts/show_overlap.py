import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
import pickle

if __name__ == '__main__':
    summary_file = './results/percentage_overlap.txt'
    plot_file = './plots/percentage_overlap.png'
    nm_county_fips = ['011', '035', '003', '059', '047', '055', '017', '007', '043', '006', '013', '021', '023', '053', '028', '033', '015', '009', '041', '045', '027', '019', '057', '029', '031', '039', '025', '005', '049', '037', '001', '051', '061']
    county_percentage_overlap = []
    for fips in nm_county_fips:
        # example fp: ./results/by_county/-1/061_Utility.UNIQUE_BLOCKS_grid.csv
        df = pd.read_csv(f"./results/by_county/-1/{fips}_Utility.NAIVE_grid.csv", sep='|')
        percentage = 100 * len(df[df['route_ids'].str.contains(',')])/(len(df))
        if percentage > 38:
            print("max: " + fips)
        county_percentage_overlap.append(percentage)
    x = sorted(county_percentage_overlap)
    y = np.arange(len(x)) / (len(x)-1)
    assert np.isclose(y[-1], 1.0), "CDF doesn't end at 1.0"
    plt.title("CDF of percentage overlap")
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
    summary_fp.write(f"---- percentage ----\n")
    summary_fp.write(f"mean: {mean}\n")
    summary_fp.write(f"median: {median}\n")
    summary_fp.write(f"mode: {mode}\n")
    summary_fp.write(f"stdev: {stdev}\n")
    summary_fp.write(f"min: {min(x)}\n")
    summary_fp.write(f"max: {max(x)}\n")
    x = pickle.load(open('./data/new_route_assoc.pkl', 'rb'))
    ids = []
    for route in x:
        for grid_row in x[route]:
            ids.append(grid_row['id'])
    state_percentage = 100 * ((len(ids) - len(set(ids)))/len(ids))
    summary_fp.write(f"state-wide: {state_percentage}\n")
    summary_fp.close()
    print(f"wrote: {summary_file}")
