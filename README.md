## Pre-processing:
  - The scripts should be run in the order described in the paper
  - The scripts use the same parameters for morphology, noise etc as described in the paper
  - You should run them in order and feed output of one script to the next

## Model:
 - This directory contains the script used for training and validating the model
 - The script has an inner DataLoader class that automatically loads the batches
   for you.

## Evaluation:
 - Contains all the code that we have used for benchmarking. It uses the same 
   approach as described in "An open approach towards the benchmarking of table structure recognition systems" by Shahab et al.
 - The file eval.py contains the core logic and calculates all the results.
 - You should execute the files in this directory in the following order:
    1) adjust_predictions.py
    2) prepare_predictions.py
    3) rescale.py
    4) eval.py
