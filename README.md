# Decision Tree from Scratch

## Configuration

1. Download the dataset and keep in a folder called `data`.
2. Extract all the files inside the data folder. (In particular, `data/csv` must contain all the csv files.
3. Create a top-level folder called `outputs`. This is where all the model outputs will go.
4. Create a virtual environment called `venv`. Run: `virtualenv venv`.
5. Install Dependencies: `source venv/bin/activate && pip install -r requirements.txt`.
6. Install the `dot` software. This is required by `python-graphviz` to render decision tree diagrams.

## Running

1. Run the `data_processor.py` to generate the target dataset from all the csv files.
This data set will remain in `outputs/data.csv`.
2. Run `analysis.py`. It will create a 80%-20% train-test split of the dataset.
Then it will create and train decision trees of heights from 2 to 10 and give out the train and test accuracies.
3. Final aggregated results of the run will be available in `outputs` folder.

## Stuff to do

1. Make it work! There are a number of hyperparameters in `dtree/learner.py`. Tweak them to a working value.
2. Debug the code. `dtree/learner.py` contains all the `id3` and related algorithms implemented.
`dtree/tree.py` contains the data structure.
3. Write the model post-pruning code.

