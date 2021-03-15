from tamf.utility import *

if __name__ == '__main__':
    create_population_cdf("./data/grid_1km_sh_fcc_eoc_voc.csv",
                          "./plots/population_cdf.png",
                          "./results/population_summary.txt")
    print("Done!")
