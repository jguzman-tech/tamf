import sys
import os
print(os.getcwd())
print(os.path.dirname(__file__))
sys.path.append('./tamf/')
from driver import *

if __name__ == '__main__':
    print("in main: ")
    # run_state_wide()
