from tamf.model import *
from pathlib import Path

def run_by_county(selected_util=None):
    Path("./results").mkdir(parents=True, exist_ok=True)
    Path("./results/by_county").mkdir(parents=True, exist_ok=True)
    # -1 means all, needed for kendall tau later
    n_values = [-1, "10%"]
    nm_county_fips = ['011', '035', '003', '059', '047', '055', '017', '007', '043', '006', '013', '021', '023', '053', '028', '033', '015', '009', '041', '045', '027', '019', '057', '029', '031', '039', '025', '005', '049', '037', '001', '051', '061']
    count = 0
    total = len(nm_county_fips) * len(n_values) * len(Utility)
    for fips in nm_county_fips:
        new_grid_file = f"./data/by_county/{fips}.csv"
        if not os.path.exists(new_grid_file):
            create_county_grid_file('./data/grid_1km_sh_fcc_eoc_voc.csv',
                                    './data/tl_2019_us_county.shp',
                                    fips,
                                    new_grid_file)
        my_model = Model(new_grid_file, './data/new_route_assoc.pkl')
        for n in n_values:
            Path(f"./results/by_county/{n}").mkdir(parents=True, exist_ok=True)
            for util in Utility: # use each heuristic
                if selected_util is not None and selected_util != util:
                    continue
                routes = my_model.solve(n, util, f"./results/by_county/{n}/{fips}_{util}")
                out_file = f"./results/by_county/{n}/{fips}_{util}_grid.csv"
                my_model.filter_grid_by_route_ids(routes, out_file)
                count += 1
                print(f"run_by_county: completed {count} out of {total} runs")

def run_state_wide(selected_util=None):
    # we have 1321 routes in all
    # -1 means all, needed for kendall tau later
    n_values = [-1, "10%"]
    Path("./results").mkdir(parents=True, exist_ok=True)
    Path("./results/state_wide").mkdir(parents=True, exist_ok=True)
    count = 0
    total = len(n_values) * len(Utility)
    my_model = Model('./data/grid_1km_sh_fcc_eoc_voc.csv', './data/new_route_assoc.pkl', True)
    for n in n_values:
        Path(f"./results/state_wide/{n}").mkdir(parents=True, exist_ok=True)
        for util in Utility: # use each heuristic
            if selected_util is not None and selected_util != util:
                    continue
            out_file = f"./results/state_wide/{n}/{util}_grid.csv"
            # you should remove this if later
            if Path(out_file).is_file():
                continue
            routes = my_model.solve(n, util, f"./results/state_wide/{n}/{util}")
            # comment the next line to avoid the costly grid creation step
            my_model.filter_grid_by_route_ids(routes, out_file)
            count += 1
            print(f"run_by_state: completed {count} out of {total} runs")

def test():
    print("Hey it worked! :)")
