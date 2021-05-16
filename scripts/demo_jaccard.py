from tamf.analysis import *

if __name__ == "__main__":
    U1 = open(f'./results/state_wide/132/Utility.UNIQUE_BLOCKS_route_ids.txt', 'r').readlines()
    U1 = [line.strip() for line in U1]
    U2 = open(f'./results/state_wide/132/Utility.POPULATED_BLOCKS_route_ids.txt', 'r').readlines()
    U2 = [line.strip() for line in U2]
    U3 = open(f'./results/state_wide/132/Utility.POPULATED_BLOCKS_W_CONFLICT_route_ids.txt', 'r').readlines()
    U3 = [line.strip() for line in U3]
    print(f"jaccard(U1, U2) = {jaccard(U1, U2)}")
    print(f"jaccard(U1, U3) = {jaccard(U1, U3)}")
    print(f"jaccard(U2, U3) = {jaccard(U2, U3)}")
