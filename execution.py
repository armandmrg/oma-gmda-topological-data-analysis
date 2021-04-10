from optparse import OptionParser
from financialTDA import runExperiment
from financialTDA import runExperimentWasserstein

#parse command
parser = OptionParser()

parser.add_option("-w", dest="windowSize", help="choose a window size",default=40)
parser.add_option("-ls", dest="landscape", help="choose to run the experiment with persistent landscapes or Wasserstein distance",default=1)

(options, args) = parser.parse_args()

w=options.windowSize
landscapeMode=options.landscape

#run experiment
if landscapeMode == 1:
    runExperiment(w)
elif landscapeMode == 0:
    runExperimentWasserstein(w)