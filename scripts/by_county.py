from tamf.utility import *
import os.path

if __name__ == '__main__':
    nm_county_fips = ['011', '035', '003', '059', '047', '055', '017', '007', '043', '006', '013', '021', '023', '053', '028', '033', '015', '009', '041', '045', '027', '019', '057', '029', '031', '039', '025', '005', '049', '037', '001', '051', '061']
    # nm_county_fips = ['021', '059']
    for fips in nm_county_fips:
        new_grid_file = f"./data/by_county/{fips}.csv"
        if not os.path.exists(new_grid_file):
            create_county_grid_file('./data/grid_1km_sh_fcc_eoc_voc.csv',
                                    './data/tl_2019_us_county.shp',
                                    fips,
                                    new_grid_file)
        my_model = Model(new_grid_file, './data/new_route_assoc.pkl')

        for h in Heuristic: # use each heuristic
            routes = my_model.solve(1, h)
            out_file = f"./results/by_county/naive_{fips}_{h}_n=1.csv"
            my_model.filter_grid_by_route_ids(routes, out_file)
