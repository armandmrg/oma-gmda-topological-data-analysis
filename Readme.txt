To run the experiments, run the the execution.py script with the following options:
-w=n to choose the size of the sliding window (by default n=50)
-ls=0 or 1 to choose which experiment you want to run (ls=1 to use persistence landscapes, ls=0 to use Wasserstein distance between consecutive diagrams)

The code is organised in 2 python files:
financialTDA.py with all the useful functions to reproduce the experiments of the paper
execution.py to run the experiments with a command

The answers are detailed in the report.