from experiments.scp_experiment import SCP_Experiment
from configs.resnet18_configs import resnet18_config

datafolder = '../data/ptbxl/'
outputfolder = '../output/'

models = [resnet18_config]

e = SCP_Experiment('your_custom_experiment', 'diagnostic',
                   datafolder, outputfolder, models)
e.prepare()
e.perform()
e.evaluate()
