# Decision Tree from Scratch

## Configuration

1. Download the dataset and keep in a folder called `data`.
2. Extract all the files inside the data folder. (In particular, `data/csv` must contain all the csv files, so create a `csv` subfolder and copy them there).
3. Create a top-level folder called `outputs`. This is where all the model outputs will go. Create 2 subfolders, `variance` and `rolling_mean` inside `outputs`. Ignore if already created.
4. Create a virtual environment called `venv`. Run: `virtualenv venv`.
5. Install Dependencies: `source venv/bin/activate && pip install -r requirements.txt`.
6. Install the `dot` software. This is required by `python-graphviz` to render decision tree diagrams.
7. To run benchmarks, install `sklearn` (not included in the dependencies)
8. In case the pip install fails due to network reasons, install `pandas, numpy, matplotlib, tqdm, graphviz` packages separately.

## Running

There are 2 modes of running, corresponding to the different modes of data aggregation, `variance` and `rolling_mean`.
For more information on the modes, please see the [report](report.pdf).

1. First generate the aggregate dataset: `python3 run_data_processor.py mode`.
2. Then run the analysis: `python3 run_analysis.py mode`.
3. [Optional] Run the sklearn benchmarks for the dataset: `python3 run_benchmarks.py mode`.

All the outputs will be available in `outputs/mode`.

Here `mode` is one of `variance` and `rolling_mean`.

