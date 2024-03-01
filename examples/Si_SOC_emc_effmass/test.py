import numpy as np
from pymatgen.electronic_structure.boltztrap2 import BztInterpolator, BztTransportProperties
from boltztrap2_aims_utils.boltztrap2_aims_utils import AimsBSLoader, parse_aims
from boltztrap2_aims_utils.myemc import EffectMass
import ase.units
from BoltzTraP2 import fite

flag = "VBM"

def calc_effmass(flag):
    aimsrun = parse_aims("./")
    EMC = EffectMass()
    data_aims = AimsBSLoader(aimsrun)
    bztInterp_aims = BztInterpolator(data_aims,lpfac=10,energy_range=10000000.0,curvature=True)
    
    if flag == 'VBM':
        band_idx = data_aims.vbm_idx
        vbm_kpt_idx = np.argmax(data_aims.ebands_all[band_idx])
        kpt_idx = vbm_kpt_idx
    elif flag == 'CBM':
        band_idx = data_aims.cbm_idx
        cbm_kpt_idx = np.argmin(data_aims.ebands_all[band_idx])
        kpt_idx = cbm_kpt_idx
    kpt = (data_aims.kpoints[kpt_idx]).tolist()
    stepsize = 0.01
    prg = 'V'
    basis = (data_aims.get_lattvec() * ase.units.Bohr).tolist()
    kpoints = EMC.get_kpointsfile(kpt, stepsize, prg, basis)
    kpoints = np.asarray(kpoints)
    lattvec = data_aims.get_lattvec()
    egrid, _vgrid = fite.getBands(kpoints, bztInterp_aims.equivalences, lattvec, bztInterp_aims.coeffs)
    EMC.cal_effmass_aims(kpt, stepsize, band_idx, basis, egrid)
    print(EMC.condeffmass)
#effmass = lambda z, T: np.abs(EMC.condeffmass) * (1 + 2 * Beta(T) * z)
#
calc_effmass("VBM")
calc_effmass("CBM")
