from tamf.utility import *
import os.path

if __name__ == '__main__':
    create_aggregate_bar_chart(1, './plots/test.png')
    # route_assoc = pickle.load(open('./data/new_route_assoc.pkl', 'rb'))
    # # nm_county_fips = ['011', '035', '003', '059', '047', '055', '017', '007', '043', '006', '013', '021', '023', '053', '028', '033', '015', '009', '041', '045', '027', '019', '057', '029', '031', '039', '025', '005', '049', '037', '001', '051', '061']
    # nm_county_fips = ['021']
    # for fips in nm_county_fips:
    #     # before filtering by any heuristic
    #     result_files = dict()
    #     for h in Heuristic: # use each heuristic
    #         # after filtering by a hueristic
    #         result_file = f"./results/by_county/naive_{fips}_{h}_n=1.csv"
    #         result_files[h] = result_file
    #     create_plot(result_files, './plots/test.png', f"FIPS:{fips}")
