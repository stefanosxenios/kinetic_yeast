from skimpy.analysis.oracle.minimum_fluxes import MinFLux, \
    MinFLuxVariable

from pytfa.io.json import load_json_model, save_json_model
from pytfa.optim.variables import LogConcentration, \
    ThermoDisplacement,DeltaG

from pytfa.analysis.sampling import sample
from pytfa.optim.utils import strip_from_integer_variables
from pytfa.analysis.variability import variability_analysis
from pytfa.analysis import apply_reaction_variability, apply_generic_variability

import pandas as pd
import numpy as np

EPSILON = 1e-9
NUM_OF_SAMPLES = 5000

make_plots = False

# Path
path_to_tmodel = './../models/redYeast_ST8943_fdp1.json'
tmodel = load_json_model(path_to_tmodel)


# Set and configure solver
CPLEX = 'optlang-cplex'
#GUROBI = 'optlang-gurobi'
tmodel.solver = CPLEX

tmodel.solver.configuration.tolerances.feasibility = 1e-9
tmodel.solver.configuration.tolerances.optimality  = 1e-9
tmodel.solver.configuration.tolerances.integrality = 1e-9


thermo_vars = [LogConcentration]
tva_thermo = variability_analysis(tmodel, kind=thermo_vars)
apply_generic_variability(tmodel, tva_thermo, inplace=True)



tva_fluxes = variability_analysis(tmodel, kind='reaction')


# Check for bi-directional reactions
bidirectional_rxns = ~ (( (tva_fluxes['maximum'] > 0) & (tva_fluxes['minimum'] > -EPSILON) ) \
                         | ((tva_fluxes['maximum'] < EPSILON) & (tva_fluxes['minimum'] < 0) ))

if tva_fluxes.loc[bidirectional_rxns].shape[0] > 0:
    raise ValueError('Model is not fixed to one FDP, cannot be sampled')

zero = (tva_fluxes['minimum'].abs() <= EPSILON) | (tva_fluxes['maximum'].abs() <= EPSILON)

# check for zero flux
if tva_fluxes[zero].shape[0]> 0 :
    raise ValueError('Minimum flux not enforced')

thermo_vars = [ThermoDisplacement, DeltaG, LogConcentration]
tva_thermo = variability_analysis(tmodel, kind=thermo_vars)

tight_model = apply_reaction_variability(tmodel, tva_fluxes, inplace=False)
tight_model = apply_generic_variability(tight_model, tva_thermo, inplace=False)



# Sample space
continuous_model = strip_from_integer_variables(tight_model)
continuous_model.repair()

continuous_model.optimize()

save_json_model(continuous_model, path_to_tmodel.replace('.json', '_continuous.json'))


samples = sample(continuous_model, NUM_OF_SAMPLES, method='achr',thinning=100)
#implementing rules from the mm algorithms
#rule Gamma_FBA<=0.52711
pruned_samples=samples[samples['LnGamma_FBA']<=-0.640340142]
pruned_samples=pruned_samples[pruned_samples['LnGamma_GLUDC']<=-18.8867]
#pruned_samples=pruned_samples[pruned_samples['LnGamma_HCO3E']>-0.15567]


 
pruned_samples.to_csv('output/new_pruned_samples_fdp1_1000.csv')

if make_plots:
    import matplotlib.pyplot as plt
    # Make violin plots of the samples and the data