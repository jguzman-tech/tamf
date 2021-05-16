#!/bin/bash
echo "from tamf.driver import *; run_by_county(Utility.POPULATED_BLOCKS_W_CONFLICT); run_state_wide(Utility.POPULATED_BLOCKS_W_CONFLICT)" | python3 -u
