import numpy as np
from pymatgen.electronic_structure.boltztrap2 import BztInterpolator, BztTransportProperties
from boltztrap2_aims_utils.boltztrap2_aims_utils import AimsBSLoader, parse_aims

aimsrun = parse_aims("./")
data_aims = AimsBSLoader(aimsrun)
bztInterp_aims = BztInterpolator(data_aims,lpfac=10,energy_range=10.0,curvature=True)
sbs_aims = bztInterp_aims.get_band_structure()
bztTransp_aims = BztTransportProperties(bztInterp_aims,temp_r = np.arange(300,1300,300), doping=10.**np.arange(10,23))

print(" n doping: ", np.average(bztTransp_aims.Effective_mass_doping['n'][0,8].diagonal()), end= " ")
print(" p doping: ", np.average(bztTransp_aims.Effective_mass_doping['p'][0,8].diagonal()), end= " ")
print()
