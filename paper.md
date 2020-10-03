# Introduction

This report is based on building a decision tree from scratch without using explicit Python libraries like sklearn for building decision tree. This is part of our Assignment in Machine Learning course taught by Professor Jayanta Mukhopadhyay at IIT Kharagpur. The dataset used in this assignment is the **[EEG Steady-State Visual Evoked Potential Signals Data Set](https://archive.ics.uci.edu/ml/datasets/EEG+Steady-State+Visual+Evoked+Potential+Signals)**

# Data Preparation

## Dataset Information

There are 5 different types of tests performed on 30 subjects


1. SB1- Five Box Visual Test 1
2. SB2- Five Box Visual Test 2
3. SB3- Five Box Visual Test 3
4. SV1- Visual Image Search
5. SM1- Motor Images(Handshake experiment)

There are a total 142 tests performed on these 30 subjects. Each subject undergoes different tests which are provided in .csv format as follows: suppose you have a .csv whose name is A001SB1_1 This means the data corresponds to group A (only Group A is provided at present), subject 001, Test SB1 (Five Box Visual Test), and first experiment.

## Attribute Information

The time series .csv files contain 16 attributes, of which last 14 are the signals coming from the electrodes. We use these 14 signals as attributes of our decision tree. The attributes are namely- AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4. Also note that the attributes are continuous-valued.

## Data Preprocessing

Since for each test, we have information of the electrodes signals at several times, we need to preprocess the data such that each data is 14 dimensional corresponding to the 14 attributes along with the label, which is the name of the test performed. To prepare the data to be fed into our algorithm, we have used two approaches.

1. Variance of the signals from each electrode
   This has been intuitively used as a good heuristic measure for filtering the dataset.
2. Rolling mean or moving average of the signals from each electrode

Data preprocessing tasks have been implemented in the file data_processor.py

## Learning Task

The learning task for our decision tree is to predict the type of test from given 14 dimensional data of the electrodes signals.

# Pruning

Pruning is a major task in decision-tree implementation to reduce overfitting of the tree. In our assignment, we have implemented a method of post-pruning called as Reduced-Error Pruning.

Steps:

1. The training data is partitioned into "grow" and "validation" set in 80:20 ratio
2. Complete tree is grown using the "grow" set using max-depth as the one that gave maximum accuracy earlier (__ in our case)
3. Until validation accuracy increases, we do bottom-up pruning

*Bottom-up Pruning Algorithm*

1. For each non-leaf node recursively call prune function
2. Once we reach leaf-node, return to to prune call in its parent node
3. Temporarily prune the node, assigning its decision as the majority vote
* If the validation accuracy increases, permanently cut/prune the node
* Then go upwards in the tree (returns to parent's prune call)

# Cites

Zotero + Better BibTex. All cites are on the file bibliography.bib. This is
a cite[@djangoproject_models_2016].

# Conclusion

Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod
tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At
vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren,
no sea takimata sanctus est Lorem ipsum dolor sit amet.

# References

