from tamf.utility import *

if __name__ == '__main__':
    nm_county_fips = ['011', '035', '003', '059', '047', '055', '017', '007', '043', '006', '013', '021', '023', '053', '028', '033', '015', '009', '041', '045', '027', '019', '057', '029', '031', '039', '025', '005', '049', '037', '001', '051', '061']
    for fips in nm_county_fips:
        create_county_grid_file('./data/tl_2019_us_county.shp')
        my_model = Model()
