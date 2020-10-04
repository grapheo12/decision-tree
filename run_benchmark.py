import sys
try:
    from analysis.variance.benchmark import runAll as var_runAll
    from analysis.rolling_mean.benchmark import runAll as roll_runAll
except ImportError:
    print("Please install Sklearn to run benchmarks")
    sys.exit(1)

if len(sys.argv) != 2 or sys.argv[1] not in ["variance", "rolling_mean"]:
    print("Usage: python3 run_benchmark.py [variance | rolling_mean]")
elif sys.argv[1] == "variance":
    var_runAll()
else:
    roll_runAll()
