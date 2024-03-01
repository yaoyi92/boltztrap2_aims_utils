#### These are the tools orginally written by Roman Kempt, provided as modified versions of BoltzTrap2 and pymatgen.
#### The difference comparing to the BoltzTrap2 version is more support for the SOC eigenvalues.
#### I just refined them and made it a separate tool. Yi Yao

import logging
import glob
import os
import numpy as np
import ase.io
from pymatgen.electronic_structure.bandstructure import (
    BandStructure,
    Spin,
)
from pymatgen.io.ase import AseAtomsAdaptor
try:
    from BoltzTraP2 import units
    import BoltzTraP2.misc
    from BoltzTraP2.units import *
    from BoltzTraP2.sphere import calc_reciprocal_iksubset
    #from BoltzTraP2.io import AIMSReader
except ImportError:
    raise BoltztrapError("BoltzTraP2 has to be installed and working")

class AIMSReader:
    """Reader for AIMS calculations.

    This class simply wraps all reading functions for AIMS calculations,
    because not all parameters can be obtained from a single file.

    Args:
        directory (str): Directory for processing.

    Attributes:
        parameters (dict): Collection of parameters.
        atoms (atoms): ASE atoms object of the structure.
        kpoints (ndarray): A (nkpoints, 3) numpy array of the fractional
            k-coordinates folded into the 1st Brillouin Zone.
    """
    def __init__(self, directory):
        outputfile = self._get_AIMS_output(directory)
        self.parameters = self.read_AIMS_output(outputfile)
        self.atoms = self.read_AIMS_struct(
            os.path.join(directory, "geometry.in")
        )
        from pathlib import Path
        if Path(directory).joinpath("SOC_eigenvalues.dat").exists():
            filename = "SOC_eigenvalues.dat"
        elif Path(directory).joinpath("Final_KS_eigenvalues.dat").exists():
            filename = "Final_KS_eigenvalues.dat"            
        else:
            raise Exception(f"Could not find either Final_KS_eigenvalues.dat or SOC_eigenvalues.dat in directory {directory}.")
        logging.info(f"Reading eigenvalues from {filename} ...")
        self.kpoints = self.read_AIMS_kpoints(
            os.path.join(directory, filename)
        )
        self.fermi_level, self.dosweight, self.eigenvalues = self.read_AIMS_energies(
            os.path.join(directory, filename)
        )
        toreduce = True
        if toreduce:
            logging.info("Reducing {} kpoints...".format(len(self.kpoints)))
            subset = calc_reciprocal_iksubset(self.atoms, None, self.kpoints)
            # currently I do not see a way to extract magnetic
            # moments correctly.
            if self.parameters["spin"] == "collinear":
                logging.warning(
                    """Post-SCF magnetic moments are not printed to the outputfile.
                Symmetry reduction is performed without taking these into account.
                Hence, results might be wrong."""
                )
            self.kpoints = np.ascontiguousarray(self.kpoints[subset, :])
            self.eigenvalues = np.ascontiguousarray(
                self.eigenvalues[:, subset, :]
            )
            logging.info("Reduced to {} kpoints.".format(len(self.kpoints)))

    def read_AIMS_struct(self, filename="geometry.in"):
        """Read in a geometry.in file in AIMS format.

        The file contains information about lattice vectors (in Angström),
            elements and positions (in angstrom), as well as initial magnetic
            moments.

        Args:
            filename (str): path to the geometry.in file

        Returns:
            atoms: An ASE Atoms object describing the structure contained in
                the file.
        """
        return ase.io.read(filename)

    def read_AIMS_kpoints(self, filename="Final_KS_eigenvalues.dat"):
        """Read in a Final_KS_eigenvalues.dat file in AIMS format.

        The file contains blocks of k-points with eigenvalues and occupations.
        FHI-AIMS writes out the eigenvalues for the entire first Billouin Zone.
        By default, it runs on a Gamma-centered grid.
        If SOC is enabled, an additional file called
        Final_KS_eigenvalues.dat.no_soc is present containing unperturbed
        eigenvalues.

        Args:
            filename (str): path to the Final_KS_eigenvalues.dat file

        Returns:
            An (nkpoints, 3) array with the coordinates with the k points in
                the file.
        """
        import re
        if "Final_KS_eigenvalues.dat" in filename:
            with open(filename, "r") as file:
                content = file.readlines()

            kpoints = [
                line.split()[-3:]
                for line in content if "k-point in recip. lattice units:" in line
            ]
            kpoints = np.array(kpoints, dtype=float)
            for row in range(kpoints.shape[0]):  #folding back into first BZ
                if kpoints[row][0] > 0.5:
                    kpoints[row][0] -= 1
                if kpoints[row][1] > 0.5:
                    kpoints[row][1] -= 1
                if kpoints[row][2] > 0.5:
                    kpoints[row][2] -= 1

            self.parameters["nkpoints"] = len(kpoints)
            return kpoints
        elif "SOC_eigenvalues.dat" in filename:
            floatpattern = re.compile(r"[-+]?(\d+\.\d+)")
            with open(filename, "r") as file:
                content = file.readlines()
            
            kpoints = [
                re.findall(floatpattern, line)
                for line in content if "K-point:" in line
            ]
            kpoints = np.array(kpoints, dtype=float)
            kpoints[kpoints > 0.5] -= 1
            
            self.parameters["nkpoints"] = len(kpoints)
            return kpoints      
        else:
            raise Exception("File not recognized.")      

    def read_AIMS_energies(self, filename="Final_KS_eigenvalues.dat"):
        """Read in a Final_KS_eigenvalues.dat file in AIMS format.

        The file contains blocks of k-points with eigenvalues and occupations.
        If SOC is enabled, every eigenvalue is split and singly occupied.
        If collinear spin is enabled, there is an additional block for spin 
        up / spin down.

        Args:
            filename (str): path to the Final_KS_eigenvalues.dat file

        Returns:
            A 3-tuple. The first element is the Fermi level in Hartree, the 
            second is the spin degeneracy of each energy (1.0 or 2.0), and the
            third is an array with shape (nspins, nkpoints, nbands) containing the
            band energies in Ha.
        """
        from itertools import groupby
        import re

        nspins = (1 if (self.parameters["spin"] is None) else 2)
        if (not self.parameters["SOC"]) and (nspins == 1):
            dosweight = 2.0
        else:
            dosweight = 1.0

        if "Final_KS_eigenvalues.dat" in filename:
            with open(filename, "r") as file:
                content = [
                    line.strip().split() for line in file.readlines()
                    if "k-point" not in line and "#" not in line and
                    "occupation number (dn), eigenvalue (dn)" not in line
                ]
                content = [
                    list(group) for k,
                    group in groupby(content, lambda x: x == []) if not k
                ]  # this splits by empty lines
                if (not self.parameters["SOC"]) and (nspins == 2):
                    spinup = np.array(
                        content, dtype=float
                    )[:, :, 2] * eV  # (nkpoints, bands)
                    spindown = np.array(
                        content, dtype=float
                    )[:, :, 4] * eV  # (nkpoints, bands)
                else:
                    content = np.array(
                        content, dtype=float
                    )[:, :, 2] * eV  # (nkpoints, bands)

            if (not self.parameters["SOC"]) and (nspins == 2):
                nbands = spinup.shape[1]
                nks = spinup.shape[0]
                nbands2 = spindown.shape[1]
                assert nbands == nbands2, "Number of spin-up and spin-down bands is not the same."
            else:
                nbands = int(content.shape[1])
                nks = content.shape[0]

            ebands = np.empty((nspins, nks, nbands))

            if (not self.parameters["SOC"]) and (nspins == 2):
                ebands[0, :, :] = spinup
                ebands[1, :, :] = spindown
            else:
                ebands[0, :, :] = content
            return (self.parameters["fermi_level"], dosweight, ebands)
        elif "SOC_eigenvalues.dat" in filename:
            nspins = 1 # for soc always 1
            dosweight = 1.0
            tablepattern = re.compile(r"(\d+)\s+(\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)")
            # State    Occupation    Unperturbed Eigenvalue [eV]    Eigenvalue [eV]    Level Spacing [eV]
            #arr = np.fromregex(str(filename), tablepattern, dtype=(float,float))
            arr1 = np.fromregex(str(filename), tablepattern, dtype=[('f1', float),('f2',float), ('f3',float), ('f4',float), ('f5',float)])
            arr = np.stack((arr1['f1'], arr1['f2'], arr1['f3'], arr1['f4'], arr1['f5'])).T
            nbands = int(np.max(arr[:, 0]))
            #occs = arr[:, 1].reshape(-1, nbands)
            eigvals = arr[:, 3].reshape(-1, nbands)
            nkpoints = eigvals.shape[0]
            ebands = np.empty((nspins, nkpoints, nbands))
            ebands[0, :, :] = eigvals * eV
            #print(ebands.shape)
            return (self.parameters["fermi_level"], dosweight, ebands)        

    def read_AIMS_output(self, outputfile):
        """Read in a .out file in AIMS format.

        The file contains information about the Fermi level, the number of spins, and the number of electrons.

        Args:
            filename (str): path to the Final_KS_eigenvalues.dat file 

        Returns:
            dict: Parameters dictionary.
        """
        parameters = {"SOC": False, "spin": None}
        with open(outputfile, "r") as file:
            for line in file.readlines():
                if "include_spin_orbit" in line:
                    parameters["SOC"] = True
                if ("spin" in line) & ("collinear" in line):
                    parameters["spin"] = "collinear"
                if "Chemical potential" in line:
                    if parameters["spin"] is not None:
                        if "spin up" in line:
                            up_fermi_level = float(line.split()[-2])
                        elif "spin dn" in line:
                            down_fermi_level = float(line.split()[-2])
                            parameters["fermi_level"] = max(
                                [up_fermi_level, down_fermi_level]
                            )
                if "Chemical potential (Fermi level)" in line:
                    fermi_level = line.replace("eV", "")
                    parameters["fermi_level"] = float(fermi_level.split()[-1])
                if "Chemical potential is" in line:
                    parameters["fermi_level"] = float(line.split()[-2])
                if "number of electrons (from input files)" in line:
                    parameters["nelectrons"] = float(line.split()[-1])
                if "Number of k-points" in line:
                    parameters["k_points"] = int(line.split()[-1])
        parameters["fermi_level"] = parameters[
            "fermi_level"] * eV  # reverting to atomic units
        return parameters

    def _get_AIMS_output(self, dirname):
        """Automatically looks for the correct AIMS .out file in dirname.

        Lists all .out file in directory and looks for "Have a nice day" line.

        Args:
            dirname (str): path to directory.

        Returns:
            str : path to outputfile.
        """
        with BoltzTraP2.misc.dir_context(dirname):
            # Find all the .out files in the directory, check for Have a nice day.
            filenames = sorted(
                [i for i in glob.glob("*.out") if os.path.isfile(i)]
            )
            if not filenames:
                return None
            if len(filenames) == 1:
                return os.path.join(dirname, filenames[0])
            if len(filenames) > 1:
                logging.warning(
                    "There is more than one .out file in the directory "
                    "- looking for 'Have a nice day.' in outputfiles."
                )
                for outfile in filenames:
                    check = os.popen(
                        "tail -n 10 {}".format(os.path.join(dirname, outfile))
                    ).read()
                    if "Have a nice day" in check:
                        return os.path.join(dirname, outfile)
                else:
                    raise LoaderError("Could not find outputfile.")


def parse_aims(directory):
    """Wrapper function to parse the AIMSReader class.

    Args:
        directory (str): Path to directory.

    Returns:
        <class> : AIMSreader class.
    """
    logging.info("Parsing AIMS ...")
    #print(directory)
    return AIMSReader(directory)

class AimsBSLoader:
    """Loader for Bandstructure from aims calculation objects"""

    def __init__(self, obj, structure=None, nelect=None):
        """
        Args:
            obj: Either a pmg Vasprun or a BandStructure object.
            structure: Structure object in case is not included in the BandStructure object.
            nelect: number of electrons in case a BandStructure obj is provided.

        Example:
            vrun = Vasprun('vasprun.xml')
            data = VasprunBSLoader(vrun)
        """
        if isinstance(obj, AIMSReader):
            structure = AseAtomsAdaptor.get_structure(obj.atoms)
            atoms = obj.atoms
            nelect = obj.parameters["nelectrons"]
            bs_obj = BandStructure(obj.kpoints, 
                         {Spin.up: obj.eigenvalues[0].T / units.eV}, 
                         AseAtomsAdaptor.get_structure(obj.atoms).lattice, 
                         obj.fermi_level / units.eV)
            kpoints = obj.kpoints
            e = obj.eigenvalues[0].T
            efermi = obj.fermi_level
            dosweight = obj.dosweight
        else:
            raise BoltztrapError("The object provided is not an AIMSReader.")

        self.kpoints = kpoints
        self.structure = structure
        self.atoms = atoms

        self.proj_all = None


        self.ebands_all = e

        self.is_spin_polarized = False

        self.dosweight = dosweight

        self.lattvec = self.atoms.get_cell().T * units.Angstrom
        self.mommat_all = None  # not implemented yet
        self.mommat = None  # not implemented yet
        self.magmom = None  # not implemented yet
        self.fermi = efermi
        self.UCvol = self.structure.volume * units.Angstrom**3

        if not bs_obj.is_metal():
            self.vbm_idx = max(bs_obj.get_vbm()["band_index"][Spin.up] + bs_obj.get_vbm()["band_index"][Spin.down])
            self.cbm_idx = min(bs_obj.get_cbm()["band_index"][Spin.up] + bs_obj.get_cbm()["band_index"][Spin.down])
            self.vbm = bs_obj.get_vbm()["energy"]
            self.cbm = bs_obj.get_cbm()["energy"]
        else:
            self.vbm_idx = None
            self.cbm_idx = None
            self.vbm = self.fermi / units.eV
            self.cbm = self.fermi / units.eV

        if nelect:
            self.nelect_all = nelect
        else:
            raise BoltztrapError("nelect must be given.")

    def get_lattvec(self):
        """
        :return: The lattice vectors.
        """
        try:
            self.lattvec
        except AttributeError:
            self.lattvec = self.atoms.get_cell().T * units.Angstrom
        return self.lattvec

    def get_volume(self):
        """
        :return: Volume
        """
        try:
            self.UCvol
        except AttributeError:
            lattvec = self.get_lattvec()
            self.UCvol = np.abs(np.linalg.det(lattvec))
        return self.UCvol

    def bandana(self, emin=-np.inf, emax=np.inf):
        """Cut out bands outside the range (emin,emax)"""
        bandmin = np.min(self.ebands_all, axis=1)
        bandmax = np.max(self.ebands_all, axis=1)
        ntoolow = np.count_nonzero(bandmax <= emin)
        accepted = np.logical_and(bandmin < emax, bandmax > emin)
        # self.data_bkp = np.copy(self.data.ebands)
        self.ebands = self.ebands_all[accepted]

        self.proj = {}
        if self.proj_all:
            if len(self.proj_all) == 2:
                h = int(len(accepted) / 2)
                self.proj[Spin.up] = self.proj_all[Spin.up][:, accepted[:h], :, :]
                self.proj[Spin.down] = self.proj_all[Spin.down][:, accepted[h:], :, :]
            elif len(self.proj_all) == 1:
                self.proj[Spin.up] = self.proj_all[Spin.up][:, accepted, :, :]

        if self.mommat_all:
            self.mommat = self.mommat[:, accepted, :]
        # Removing bands may change the number of valence electrons
        if self.nelect_all:
            self.nelect = self.nelect_all - self.dosweight * ntoolow

        return accepted
