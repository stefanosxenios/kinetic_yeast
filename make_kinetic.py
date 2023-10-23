from skimpy.analysis.oracle.minimum_fluxes import MinFLux, \
    MinFLuxVariable
from pytfa.io.json import load_json_model
from pytfa.analysis import variability_analysis
from skimpy.mechanisms  import make_irrev_massaction
from skimpy.core import Reaction
from skimpy.core.compartments import Compartment
from skimpy.utils.general import sanitize_cobra_vars

from skimpy.io.generate_from_pytfa import FromPyTFA
from skimpy.io.yaml import export_to_yaml

# Path
# Select pytfa model constrainted to one FDP
path_to_tmodel = './../models/redYeast_ST8943_fdp1_continuous.json'

# Ouput path
path_to_kmodel = './../models/redYeast_ST8943_fdp1_curated.yml'


# Import and set solver
tmodel = load_json_model(path_to_tmodel)

# Set and configure solver
CPLEX = 'optlang-cplex'
GUROBI = 'optlang-gurobi'
tmodel.solver = CPLEX

tmodel.solver.configuration.tolerances.feasibility = 1e-9
tmodel.solver.configuration.tolerances.optimality  = 1e-9
tmodel.solver.configuration.tolerances.integrality = 1e-9

sol_fdp = tmodel.optimize()
tmodel.reactions.GROWTH.lower_bound = sol_fdp.objective_value
# Round biomass to 1e-5*1e-4 = 1e-9
# for rxn in tmodel.reactions:
#     if rxn.id.startswith('LMPD_'):
#         rxn.add_metabolites({x: round(v, 8) - v for x,v in rxn.metabolites.items()})

sol_fdp = tmodel.optimize()

small_molecules = ['h_c', 'h_e','h_m', 'h_ce', 'h_er', 'h_erm', 'h_g', 'h_gm', 'h_i', 'h_v']

reactants_to_exclude = ['mn2_c', 'mn2_e',
                        'so4_c', 'so4_e',
                        'na1_c', 'na1_e',
                        'zn2_e', 'zn2_c',
                        'fe2_e', 'fe2_c',
                        'cu2_e', 'cu2_c',
                        'cl_e', 'cl_c',
                        'mg2_c', 'mg2_e',
                        'k_c', 'k_e',
                        'biomass_c']


model_gen = FromPyTFA(small_molecules=small_molecules,
                      reactants_to_exclude=reactants_to_exclude,
                      max_revesible_deltag_0=100,)


kmodel = model_gen.import_model(tmodel,
                                sol_fdp.raw,
                                concentration_scaling_factor=1e6)



"""
Add and map compartements 
"""
for c in tmodel.compartments:
    comp = Compartment(name=c)
    kmodel.add_compartment(comp)

for met in tmodel.metabolites:
    comp = kmodel.compartments[met.compartment]
    kin_met = sanitize_cobra_vars(met.id)
    if kin_met in kmodel.reactants:
        kmodel.reactants[kin_met].compartment=comp

"""
Add volume parameters
"""

reference_cell_volume = {'cell_volume_c': 42.,
                         'cell_volume_ce': 42.,
                         'cell_volume_e': 42.,
                         'cell_volume_er': 42.,
                         'cell_volume_erm': 42.,
                         'cell_volume_g': 42.,
                         'cell_volume_gm': 42.,
                         'cell_volume_i': 42.,
                         'cell_volume_m': 42.,
                         'cell_volume_v': 42.,
                         'volume_c': 0.7*42,
                         'volume_ce': 1e-0*42,
                         'volume_e': 42.,
                         'volume_er': 0.02*42,
                         'volume_erm': 1e-0*42,
                         'volume_g': 0.01*42,
                         'volume_gm': 1e-0*42,
                         'volume_i': 0.01*0.1*42, # 10 % of mitochondria
                         'volume_m': 0.01*0.9*42, # 90 % of mitochondria
                         'volume_v': 0.07*42,
}

kmodel.parameters = reference_cell_volume

# Export the kinetic model
export_to_yaml(kmodel, path_to_kmodel)







