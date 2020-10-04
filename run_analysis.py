import sys
from analysis.variance.analysis import runAll as var_runAll
from analysis.rolling_mean.analysis import runAll as roll_runAll

if len(sys.argv) != 2 or sys.argv[1] not in ["variance", "rolling_mean"]:
    print("Usage: python3 run_analysis.py [variance | rolling_mean]")
elif sys.argv[1] == "variance":
    var_runAll()
else:
    roll_runAll()
