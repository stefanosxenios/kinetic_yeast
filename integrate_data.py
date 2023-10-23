from pytfa.io.json import load_json_model, save_json_model
from pytfa.analysis.variability import variability_analysis
from pytfa.optim.variables import LogConcentration, ThermoDisplacement
from skimpy.analysis.oracle import *
from pytfa.optim.relaxation import relax_dgo
from cobra import Reaction, Metabolite

import pandas as pd
import numpy as np

plot_output = True

EPSILON = 1e-9
MIN_FLUX = 1e-5#

# Path
path_to_tmodel = './../../../models/tfa/mini_redYeast8_26Oct2020_151326.json'

save_model_as = './../models/redYeast_ST8943_fdp1.json'
"""
Load wildtype model
"""
# Load model
tmodel = load_json_model(path_to_tmodel)

# GLPK = 'optlang-glpk'
CPLEX = 'optlang-cplex'
# GUROBI = 'optlang-gurobi'
tmodel.solver = CPLEX

tmodel.solver.configuration.tolerances.feasibility = 1e-9
tmodel.solver.configuration.tolerances.optimality  = 1e-9
tmodel.solver.configuration.tolerances.integrality = 1e-9

# Remove rhe revesre reaction of ALCD2x > ALCD2ir we model as a reversible reaction!
tmodel.remove_reactions([tmodel.reactions.ALCD2ir,])
tmodel.reactions.ALCD2x.bounds = (-100, 100)

# NADH

tmodel.log_concentration.nadp_c.variable.ub = np.log(150e-6)
tmodel.log_concentration.nadp_c.variable.lb = np.log(20e-6)
tmodel.log_concentration.nadph_c.variable.ub = np.log(150e-6)
tmodel.log_concentration.nadph_c.variable.lb = np.log(50e-6)

tmodel.log_concentration.nadp_m.variable.ub = np.log(150e-6)
tmodel.log_concentration.nadp_m.variable.lb = np.log(20e-6)
tmodel.log_concentration.nadph_m.variable.ub = np.log(150e-6)
tmodel.log_concentration.nadph_m.variable.lb = np.log(50e-6)


tmodel.log_concentration.nadh_c.variable.ub = np.log(160e-6)
tmodel.log_concentration.nadh_c.variable.lb = np.log(100e-6)
tmodel.log_concentration.nad_c.variable.ub = np.log(1600e-6)
tmodel.log_concentration.nad_c.variable.lb = np.log(1000e-6)

tmodel.log_concentration.nadh_m.variable.ub = np.log(160e-6)
tmodel.log_concentration.nadh_m.variable.lb = np.log(100e-6)
tmodel.log_concentration.nad_m.variable.ub = np.log(1600e-6)
tmodel.log_concentration.nad_m.variable.lb = np.log(1000e-6)

sol = tmodel.optimize()


"""
Heterologoous pathways
"""
# Add new metabolites
_3dhs = tmodel.metabolites.get_by_id('3dhsk_c')

pca_c = Metabolite(
    'pca_c',
    formula='',
    name='3,4-Dihydroxybenzoate',
    compartment='c')

pca_e = Metabolite(
    'pca_e',
    formula='',
    name='3,4-Dihydroxybenzoate',
    compartment='e')


catechol_c = Metabolite(
    'catechol_c',
    formula='',
    name='catechol',
    compartment='c')

ccm_c = Metabolite(
    'ccm_c',
    formula='',
    name='cis-cis-muconate',
    compartment='c')

ccm_e = Metabolite(
    'ccm_e',
    formula='',
    name='cis-cis-muconate',
    compartment='e')

#EC: 4.2.1.118 3-dehydroshikimate dehydratase PaAroZ,
PaAroZ = Reaction('PaAroZ')
PaAroZ.add_metabolites({_3dhs:-1,
                        pca_c:1,
                        tmodel.metabolites.h2o_c:1
                        })

KpAroY = Reaction('KpAroY')
KpAroY.add_metabolites({pca_c:-1,
                        tmodel.metabolites.co2_c:1,
                        catechol_c:1
                        })


CaCatA = Reaction('CaCatA')
CaCatA.add_metabolites({catechol_c:-1,
                        tmodel.metabolites.o2_c:-1,
                        ccm_c:1
                        })

pca2tp = Reaction('pca2tp')
pca2tp.add_metabolites({pca_c:-1,
                        pca_e:1
                        })

ccm2tp = Reaction('ccm2tp')
ccm2tp.add_metabolites({ccm_c:-1,
                        ccm_e:1
                        })

ex_ccm = Reaction('EX_ccm_e')
ex_ccm.add_metabolites({ccm_e:-1,})

ex_pca = Reaction('EX_pca_e')
ex_pca.add_metabolites({pca_e:-1,})

tmodel.add_reactions( [PaAroZ, KpAroY, CaCatA, ccm2tp,pca2tp, ex_ccm, ex_pca] )

tmodel.optimize()

"""
Add Gene KOS
"""

# Metabolic genes KOs from UV mutations
# DUG1 Cys-Gly metallodipeptidase (Gense not in red model)
# MET22 3'(2'),5'-bisphosphate nucleotidase (Gene not in red model)
# BUB1 heckpoint serine/threonine-protein kinase BUB (Gene not in red model)

# Cant model the introducion of URA3 as this part of the metabolism is not included in the red model


"""
Add fermentation data
"""


ratio_gdw_gww = 1-0.68
density = 1200


for c in tmodel.compartments.values():
    c['c_min'] = 1e-10

# Exometabolomics:
tmodel.compartments['e']['pH'] = 6.0

# Integrate Metabolomics from fitting the fermentation data
tmodel.log_concentration.glc__D_e.variable.ub = np.log(1e-5*1.02)
tmodel.log_concentration.glc__D_e.variable.lb = np.log(1e-5*0.98)


# Integrate Extracellular fluxes
tmodel.reactions.EX_glc__D_e.bounds = (-0.55,0)
tmodel.reactions.EX_ccm_e.bounds = (0.028,0.028*1.1)
tmodel.reactions.EX_pca_e.bounds = (0.008,0.008*1.1)

sol = tmodel.optimize()

MIN_GROWTH = 0.03

# Constraint non-medium concentrations to be lower then muM
LC_SECRETION = np.log(1e-6)
secretions = [r for r in tmodel.boundary if r.upper_bound <= 0]
for sec in secretions:
    for met in sec.metabolites:
        if met.id in ['h2o_2', 'h_e']:
            continue
        try:
            tmodel.log_concentration.get_by_id(met.id).variable.upper_bound = LC_SECRETION
        except KeyError:
            pass

"""
Minimalfluxes 
"""

# Scaling to avoid numerical errors with bad lumps
for rxn in tmodel.reactions:
    if rxn.id.startswith('LMPD_'):
        rxn.add_metabolites({x:v*(1e-5 - 1) for x,v in rxn.metabolites.items()})


print(tmodel.reactions.LMPD_s_0450_c_1_256.reaction)
sol = tmodel.optimize()

tva_fluxes = variability_analysis(tmodel,kind='reactions')

small_fluxes = [rxn for rxn in tva_fluxes.index
                if (tva_fluxes.loc[rxn,'minimum'] > -MIN_FLUX) and
                (tva_fluxes.loc[rxn,'maximum'] < MIN_FLUX)]

# List of lumps and biomass
lmps = [rxn.id for rxn in tmodel.reactions if rxn.id.startswith('LMPD')]
exclude_reactions = lmps



# Removed blocked reactions
blocked = (tva_fluxes['maximum'] < EPSILON) & (tva_fluxes['minimum'] > -EPSILON)
# Print blocked reactions
tva_fluxes[blocked].index

tmodel.remove_reactions( [tmodel.reactions.get_by_id(i) for i in tva_fluxes[blocked].index ])

# Test GROWTH
sol = tmodel.optimize()
tmodel.reactions.GROWTH.lower_bound = MIN_GROWTH
print("Test growth after removing blocked reactions {:0.3f}".format(sol.objective_value))


#Add Minimal flux requirements
tmodel_basal = add_min_flux_requirements(tmodel,
                          MIN_FLUX,
                          inplace=False,
                          exclude=small_fluxes)

tmodel_basal.reactions.GROWTH.lower_bound = MIN_GROWTH

# Relax the minimal flux requirements
try:
    sol_basal = tmodel_basal.optimize()
    print("Test growth of model with min flux requierements {:0.3f}"
          .format(sol_basal.objective_value))

except:
    print('Model not feasible with minimal fluxes, relaxing lower bounds of FW and BW fluxes'
          'except biomass')


# Run a TVA and check the BDRS commented here
tmodel_basal.reactions.GROWTH.lower_bound = MIN_GROWTH

tva_fluxes_basal0 = variability_analysis(tmodel_basal,kind='reactions')
BDRS = ~ (( (tva_fluxes_basal0['maximum'] > 0) & (tva_fluxes_basal0['minimum'] > -EPSILON) ) \
       | ((tva_fluxes_basal0['maximum'] < EPSILON) & (tva_fluxes_basal0['minimum'] < 0) ))
print(tva_fluxes_basal0[BDRS])


# Hydrogran uptake from the environment (pH out << ph in)
to_backward = ['EX_h_e']
for rxn_id in to_backward:
    tmodel_basal.reactions.get_by_id(rxn_id).upper_bound = 0
tmodel_basal.optimize()


# 1) Assume diffusion of akg and mal__L out of the cells (passive transporters)
# as not in medium ['AKGt', 'MALt'] to backward
to_backward = ['AKGt', 'MALt']
for rxn_id in to_backward:
    tmodel_basal.reactions.get_by_id(rxn_id).upper_bound = 0.0
tmodel_basal.optimize()


# 2) Glycolysis down
to_forward = ['FBA','PGI','PGM']
for rxn_id in to_forward:
    tmodel_basal.reactions.get_by_id(rxn_id).lower_bound = 0.0
tmodel_basal.optimize()


# Constraint FDP based cyclic TCA
# 2) BDRs in TCA Cycle 'SUCOAACTm',  'ICDHyr',
to_forward = ['ACN_b_m','ACONTb' , 'ACONTm','MDHm' ,'FUM', 'SUCOAACTm',  'ICDHyr', 'MDH']
#'MDHm','MDH','FRD2m', 'FUM']

for rxn_id in to_forward:
    tmodel_basal.reactions.get_by_id(rxn_id).lower_bound = 0.0
for rxn_id in to_backward:
    tmodel_basal.reactions.get_by_id(rxn_id).upper_bound = 0.0
tmodel_basal.optimize()

# 2) BDRs in Pentosphosphate pathway (FDP reported in Park et al 2016)
#  - Conversion of s7p to e4p ['TALA'] to forward
#  - Conversion of s17bp_c to dhap_c + e4p_c' ['FBA3'] to forward (Maybe cases FDP1 and FDP2 ?)
to_forward = ['TALA', 'TKT1' , 'TKT2','PPM','TPI',
              'FBA3','ENO']
for rxn_id in to_forward:
    tmodel_basal.reactions.get_by_id(rxn_id).lower_bound = 0.0
tmodel_basal.optimize()


# 3) Glutamate production and interconversion (most abundant metabolite)
#    - Transport of mitochondrial akg to cytosolic akg [AKGCITtm] to backward
#    - Conversion of  akg_c + phe__L_c <=> glu__L_c + phpyr_c [PHETA1,] to forward
#      doi: 10.1128/AEM.69.8.4534-4541.2003
#    - Conversion of glu  isoleucine and aspartate ['ILETA','ASPTA'] (Park et al 2016)
#    - Tyrosin metabolism production of tyrosine from 3-(4-hydroxyphenyl)pyruvate [TYRTA,TYRTRA] to fwd
#      https://pathway.yeastgenome.org/YEAST/NEW-IMAGE?object=PWY3O-4120

to_forward = ['PHETA1','TYRTA','TYRTRA','ALATA_L']
to_backward= ['AKGCITtm',
              'ILETA','ASPTA']
for rxn_id in to_forward:
    tmodel_basal.reactions.get_by_id(rxn_id).lower_bound = 0.0
for rxn_id in to_backward:
    tmodel_basal.reactions.get_by_id(rxn_id).upper_bound = 0.0

tmodel_basal.optimize()

# 11) Mitochondrial citrate transporter fully reversibile {Again could be used to make cases here!}
#      - ['CITtcm'] to backward to transport some citrate to the cytosol for FA syn

to_forward = ['PPItm', 'CITtam','CITtcm', 'D_LACt2m','ALCD2x',]
to_backward= ['AKGMAL','PHETRA']
for rxn_id in to_forward:
    tmodel_basal.reactions.get_by_id(rxn_id).lower_bound = 0.0
for rxn_id in to_backward:
    tmodel_basal.reactions.get_by_id(rxn_id).upper_bound = 0.0

tmodel_basal.optimize()

tva_fluxes_basal0 = variability_analysis(tmodel_basal,kind='reactions')

BDRS = ~ (( (tva_fluxes_basal0['maximum'] > 0) & (tva_fluxes_basal0['minimum'] > -EPSILON) ) \
       | ((tva_fluxes_basal0['maximum'] < EPSILON) & (tva_fluxes_basal0['minimum'] < 0) ))
print(tva_fluxes_basal0[BDRS])

# Removed blocked reactions
blocked = (tva_fluxes_basal0['maximum'] < EPSILON) & (tva_fluxes_basal0['minimum'] > -EPSILON)
# Print blocked reactions
tva_fluxes_basal0[blocked].index

tmodel_basal.remove_reactions( [tmodel_basal.reactions.get_by_id(i)
                                for i in tva_fluxes_basal0[blocked].index ])

tva_fluxes_basal0 = variability_analysis(tmodel_basal,kind='reactions')
blocked = (tva_fluxes_basal0['maximum'] < EPSILON) & (tva_fluxes_basal0['minimum'] > -EPSILON)
print(tva_fluxes_basal0[blocked])

small_fluxes = [rxn for rxn in tva_fluxes_basal0.index
                if (tva_fluxes_basal0.loc[rxn,'minimum'] > -MIN_FLUX) and
                (tva_fluxes_basal0.loc[rxn,'maximum'] < MIN_FLUX)]

# Check small fluxes
print(tva_fluxes_basal0.loc[small_fluxes])


if tva_fluxes_basal0[blocked].shape[0]> 0 :
    print('Remove blocked reactions: {}'.format(tva_fluxes_basal0[blocked].index))
    tmodel_basal.remove_reactions( [tmodel_basal.reactions.get_by_id(i)
                                    for i in tva_fluxes_basal0[blocked].index ])
    print("Test growth of model with removed reactions {:0.3f}"
          .format(tmodel_basal.optimize().objective_value))


# Constratint small fluxes to be large than 0
#                  maximum       minimum
# ADCS        2.716508e-06  1.902000e-06
# CA2t3ec     9.297828e-06  6.510000e-06
# DHFS        2.716508e-06  1.902000e-06
# DHPTtm     -1.902000e-06 -2.716508e-06
# EX_ca2_e   -6.510000e-06 -9.297828e-06
# EX_fe2_e   -9.420000e-07 -1.345400e-06
# EX_gcald_e  2.767925e-06  1.938000e-06
# FE2t        1.345400e-06  9.420000e-07
# FRDPtm      4.284714e-08  3.000000e-08
# GCALDt     -1.938000e-06 -2.767925e-06
# GLYtm       3.427771e-07  2.400000e-07
# HEMEOSm     4.284714e-08  3.000000e-08
# THZPSN1_SC  5.141656e-08  0.000000e+00
# THZPSN2_SC  5.141656e-08  0.000000e+00


tmodel_basal.reactions.ADCS.lower_bound = 1e-6
tmodel_basal.reactions.CA2t3ec.lower_bound = 1e-6
tmodel_basal.reactions.DHFS.lower_bound = 1e-6

tmodel_basal.reactions.DHPTtm.upper_bound = -1e-6
tmodel_basal.reactions.EX_ca2_e.upper_bound = -1e-6
tmodel_basal.reactions.EX_fe2_e.upper_bound = -1e-6

tmodel_basal.reactions.EX_gcald_e.lower_bound = 1e-6
tmodel_basal.reactions.FE2t.lower_bound = 1e-7
tmodel_basal.reactions.FRDPtm.lower_bound = 1e-8

tmodel_basal.reactions.GCALDt.upper_bound = -1e-6

tmodel_basal.reactions.GLYtm.lower_bound = 1e-7
tmodel_basal.reactions.HEMEOSm.lower_bound = 1e-8
#tmodel_basal.reactions.THZPSN1_SC.lower_bound = 1e-8
tmodel_basal.reactions.THZPSN2_SC.lower_bound = 1e-8


# Solve with in the defined FDP
sol_fdp = tmodel_basal.optimize()

tmodel_basal.reactions.GROWTH.lower_bound = MIN_GROWTH
exclude_reactions.append('FERCOXOXI', )

# Add missing delta Gs for all except lumps
tmodel_basal_dg0 = add_undefined_delta_g(tmodel_basal,
                                     sol_fdp,
                                     delta_g_std=0,
                                     delta_g_std_err=40,
                                     add_displacement=True,
                                     inplace=False,
                                     exclude_reactions=exclude_reactions)

tmodel_basal_dg0.optimize()

# Add remaining deltaGs for lumps and biomass
tmodel_basal_dg0 = add_undefined_delta_g(tmodel_basal_dg0,
                        sol_fdp,
                        delta_g_std=0,
                        delta_g_std_err=1e6,
                        add_displacement=True,
                        inplace=True,)

sol_basal_dg0 = tmodel_basal_dg0.optimize()

tmodel_basal.reactions.GROWTH.lower_bound = MIN_GROWTH


tva_disp = variability_analysis(tmodel_basal_dg0, kind=[ThermoDisplacement])

MIN_DISP = 1e-2
zero_disp = (tva_disp['minimum'].abs() <= MIN_DISP) & (tva_disp['maximum'].abs() <= MIN_DISP)

if tva_disp[zero_disp].shape[0] > 0:
    print(tva_disp[zero_disp])
    reactions_to_relax = [disp.id for disp in tmodel_basal_dg0.thermo_displacement if
                          disp.variable.name in tva_disp[zero_disp].index]
    for rxn_id in reactions_to_relax:
        dg0_var = tmodel_basal_dg0.delta_gstd.get_by_id(rxn_id).variable
        dg0_var.lb = dg0_var.lb - 1e1 * MIN_DISP*tmodel_basal_dg0.RT
        dg0_var.ub = dg0_var.ub + 1e1 * MIN_DISP*tmodel_basal_dg0.RT

    #raise ValueError()

# Force minimum displacement
tmodel_basal_min_disp = add_min_log_displacement(tmodel_basal_dg0, 1e-3,
                                                 tva_fluxes=tva_fluxes_basal0,
                                                 inplace=False)

tva_disp = variability_analysis(tmodel_basal_dg0, kind=[ThermoDisplacement])

MAX_LOG_TRANSPORT_DISPLACEMENT = np.log(0.1)
ETC_REACTIONS = ['NADH2_u6cm', 'NADH2_u6m', 'FERCOXOXI', 'UBICRED']


transport_reactions =  [r.id for r in tmodel_basal_min_disp.reactions
                        if (len(r.compartments) > 1 )
                        and not ('i' in r.compartments)
                        and not r.id.startswith('LMPD_')
                        and not r.id in ETC_REACTIONS]

for r_id in transport_reactions:
    variable = tmodel_basal_min_disp.thermo_displacement.get_by_id(r_id).variable
    max_disp, min_disp = tva_disp.loc[variable.name, :]
    print(max_disp, min_disp)
    if max_disp > -MAX_LOG_TRANSPORT_DISPLACEMENT \
            and min_disp > 0:

        if not min_disp > -MAX_LOG_TRANSPORT_DISPLACEMENT:
            variable.ub = -MAX_LOG_TRANSPORT_DISPLACEMENT
        else:
            print('Reaction needs to be displaced from eqiulibrium {}'.format(r_id))


    elif max_disp < 0 \
            and min_disp < MAX_LOG_TRANSPORT_DISPLACEMENT:

        if not max_disp < MAX_LOG_TRANSPORT_DISPLACEMENT:
            variable.lb = MAX_LOG_TRANSPORT_DISPLACEMENT
        else:
            print('Reaction needs to be displaced from eqiulibrium {}'.format(r_id))
    else:
        pass
    tmodel_basal_min_disp.optimize()

    print(variable)

tmodel_basal_min_disp.optimize()

# Save the model
#del tmodel_basal_min_disp.relaxation
save_json_model(tmodel_basal_min_disp, save_model_as)

tva_fluxes_fdp = variability_analysis(tmodel_basal_min_disp, kind='reactions')
tva_conc_fdp = variability_analysis(tmodel_basal_min_disp, kind=[LogConcentration])

