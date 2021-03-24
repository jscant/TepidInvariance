"""
Convert pdb, sdf and mol2 coordinates files into pandas-readable parquet files,
which can be used by point cloud/GNN models in this repo. Usage:

pdb_to_parquet.py <base_path> <output_path>

<base_path> should be structured like so:

<base_path>
├── ligands
│   ├── receptor_a
│   │   └── ligands.sdf
│   └── receptor_b
│       └── ligands.sdf
└── receptors
    ├── receptor_a
    │   └── receptor.pdb
    └── receptor_a
        └── receptor.pdb
"""
import multiprocessing as mp
import urllib
from collections import defaultdict, namedtuple
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pandas as pd
import yaml
from Bio import PDB as PDB
from openbabel import openbabel
from plip.basic.supplemental import extract_pdbid

try:
    from openbabel import pybel
except (ModuleNotFoundError, ImportError):
    import pybel

import argparse

from plip.structure.preparation import PDBComplex

RESIDUE_IDS = {'MET', 'ARG', 'SER', 'TRP', 'HIS', 'CYS', 'LYS', 'GLU', 'THR',
               'LEU', 'TYR', 'PRO', 'ASN', 'ASP', 'PHE', 'GLY', 'VAL', 'ALA',
               'ILE', 'GLN'}


def fetch_pdb(pdbid):
    """Modified plip function."""
    pdbid = pdbid.lower()
    pdburl = f'https://files.rcsb.org/download/{pdbid}.pdb'
    try:
        pdbfile = urlopen(pdburl).read().decode()
        if 'sorry' in pdbfile:
            print('No file in PDB format available from wwPDB for', pdbid)
            return None, None
    except HTTPError:
        print('No file in PDB format available from wwPDB for', pdbid)
        return None, None
    return [pdbfile, pdbid]


class Info:
    """Data structure to hold atom type data"""

    def __init__(
            self,
            sm,
            smina_name,
            adname,
            anum,
            ad_radius,
            ad_depth,
            ad_solvation,
            ad_volume,
            covalent_radius,
            xs_radius,
            xs_hydrophobe,
            xs_donor,
            xs_acceptor,
            ad_heteroatom,
    ):
        self.sm = sm
        self.smina_name = smina_name
        self.adname = adname
        self.anum = anum
        self.ad_radius = ad_radius
        self.ad_depth = ad_depth
        self.ad_solvation = ad_solvation
        self.ad_volume = ad_volume
        self.covalent_radius = covalent_radius
        self.xs_radius = xs_radius
        self.xs_hydrophobe = xs_hydrophobe
        self.xs_donor = xs_donor
        self.xs_acceptor = xs_acceptor
        self.ad_heteroatom = ad_heteroatom


class DistanceCalculator:
    """Python reimplementation of the gninatyper function,
    as per https://pubs.acs.org/doi/10.1021/acs.jcim.6b00740
    (some code modified from Constantin Schneider, OPIG)
    """

    def __init__(self):
        # self.etab = openbabel.OBElementTable()
        self.non_ad_metal_names = [
            "Cu",
            "Fe",
            "Na",
            "K",
            "Hg",
            "Co",
            "U",
            "Cd",
            "Ni",
            "Si",
        ]
        self.atom_equivalence_data = [("Se", "S")]
        self.atom_type_data = [
            Info(
                "Hydrogen",
                "Hydrogen",
                "H",
                1,
                1.000000,
                0.020000,
                0.000510,
                0.000000,
                0.370000,
                0.000000,
                False,
                False,
                False,
                False,
            ),
            Info(
                "PolarHydrogen",
                "PolarHydrogen",
                "HD",
                1,
                1.000000,
                0.020000,
                0.000510,
                0.000000,
                0.370000,
                0.000000,
                False,
                False,
                False,
                False,
            ),
            Info(
                "AliphaticCarbonXSHydrophobe",
                "AliphaticCarbonXSHydrophobe",
                "C",
                6,
                2.000000,
                0.150000,
                -0.001430,
                33.510300,
                0.770000,
                1.900000,
                True,
                False,
                False,
                False,
            ),
            Info(
                "AliphaticCarbonXSNonHydrophobe",
                "AliphaticCarbonXSNonHydrophobe",
                "C",
                6,
                2.000000,
                0.150000,
                -0.001430,
                33.510300,
                0.770000,
                1.900000,
                False,
                False,
                False,
                False,
            ),
            Info(
                "AromaticCarbonXSHydrophobe",
                "AromaticCarbonXSHydrophobe",
                "A",
                6,
                2.000000,
                0.150000,
                -0.000520,
                33.510300,
                0.770000,
                1.900000,
                True,
                False,
                False,
                False,
            ),
            Info(
                "AromaticCarbonXSNonHydrophobe",
                "AromaticCarbonXSNonHydrophobe",
                "A",
                6,
                2.000000,
                0.150000,
                -0.000520,
                33.510300,
                0.770000,
                1.900000,
                False,
                False,
                False,
                False,
            ),
            Info(
                "Nitrogen",
                "Nitrogen",
                "N",
                7,
                1.750000,
                0.160000,
                -0.001620,
                22.449300,
                0.750000,
                1.800000,
                False,
                False,
                False,
                True,
            ),
            Info(
                "NitrogenXSDonor",
                "NitrogenXSDonor",
                "N",
                7,
                1.750000,
                0.160000,
                -0.001620,
                22.449300,
                0.750000,
                1.800000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "NitrogenXSDonorAcceptor",
                "NitrogenXSDonorAcceptor",
                "NA",
                7,
                1.750000,
                0.160000,
                -0.001620,
                22.449300,
                0.750000,
                1.800000,
                False,
                True,
                True,
                True,
            ),
            Info(
                "NitrogenXSAcceptor",
                "NitrogenXSAcceptor",
                "NA",
                7,
                1.750000,
                0.160000,
                -0.001620,
                22.449300,
                0.750000,
                1.800000,
                False,
                False,
                True,
                True,
            ),
            Info(
                "Oxygen",
                "Oxygen",
                "O",
                8,
                1.600000,
                0.200000,
                -0.002510,
                17.157300,
                0.730000,
                1.700000,
                False,
                False,
                False,
                True,
            ),
            Info(
                "OxygenXSDonor",
                "OxygenXSDonor",
                "O",
                8,
                1.600000,
                0.200000,
                -0.002510,
                17.157300,
                0.730000,
                1.700000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "OxygenXSDonorAcceptor",
                "OxygenXSDonorAcceptor",
                "OA",
                8,
                1.600000,
                0.200000,
                -0.002510,
                17.157300,
                0.730000,
                1.700000,
                False,
                True,
                True,
                True,
            ),
            Info(
                "OxygenXSAcceptor",
                "OxygenXSAcceptor",
                "OA",
                8,
                1.600000,
                0.200000,
                -0.002510,
                17.157300,
                0.730000,
                1.700000,
                False,
                False,
                True,
                True,
            ),
            Info(
                "Sulfur",
                "Sulfur",
                "S",
                16,
                2.000000,
                0.200000,
                -0.002140,
                33.510300,
                1.020000,
                2.000000,
                False,
                False,
                False,
                True,
            ),
            Info(
                "SulfurAcceptor",
                "SulfurAcceptor",
                "SA",
                16,
                2.000000,
                0.200000,
                -0.002140,
                33.510300,
                1.020000,
                2.000000,
                False,
                False,
                False,
                True,
            ),
            Info(
                "Phosphorus",
                "Phosphorus",
                "P",
                15,
                2.100000,
                0.200000,
                -0.001100,
                38.792400,
                1.060000,
                2.100000,
                False,
                False,
                False,
                True,
            ),
            Info(
                "Fluorine",
                "Fluorine",
                "F",
                9,
                1.545000,
                0.080000,
                -0.001100,
                15.448000,
                0.710000,
                1.500000,
                True,
                False,
                False,
                True,
            ),
            Info(
                "Chlorine",
                "Chlorine",
                "Cl",
                17,
                2.045000,
                0.276000,
                -0.001100,
                35.823500,
                0.990000,
                1.800000,
                True,
                False,
                False,
                True,
            ),
            Info(
                "Bromine",
                "Bromine",
                "Br",
                35,
                2.165000,
                0.389000,
                -0.001100,
                42.566100,
                1.140000,
                2.000000,
                True,
                False,
                False,
                True,
            ),
            Info(
                "Iodine",
                "Iodine",
                "I",
                53,
                2.360000,
                0.550000,
                -0.001100,
                55.058500,
                1.330000,
                2.200000,
                True,
                False,
                False,
                True,
            ),
            Info(
                "Magnesium",
                "Magnesium",
                "Mg",
                12,
                0.650000,
                0.875000,
                -0.001100,
                1.560000,
                1.300000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "Manganese",
                "Manganese",
                "Mn",
                25,
                0.650000,
                0.875000,
                -0.001100,
                2.140000,
                1.390000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "Zinc",
                "Zinc",
                "Zn",
                30,
                0.740000,
                0.550000,
                -0.001100,
                1.700000,
                1.310000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "Calcium",
                "Calcium",
                "Ca",
                20,
                0.990000,
                0.550000,
                -0.001100,
                2.770000,
                1.740000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "Iron",
                "Iron",
                "Fe",
                26,
                0.650000,
                0.010000,
                -0.001100,
                1.840000,
                1.250000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "GenericMetal",
                "GenericMetal",
                "M",
                0,
                1.200000,
                0.000000,
                -0.001100,
                22.449300,
                1.750000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            # note AD4 doesn't have boron, so copying from carbon
            Info(
                "Boron",
                "Boron",
                "B",
                5,
                2.04,
                0.180000,
                -0.0011,
                12.052,
                0.90,
                1.920000,
                True,
                False,
                False,
                False,
            ),
        ]
        self.atom_types = [info.sm for info in self.atom_type_data]
        self.type_map = self.get_type_map()

    def get_type_map(self):
        """Original author: Constantin Schneider"""
        types = [
            ['AliphaticCarbonXSHydrophobe'],
            ['AliphaticCarbonXSNonHydrophobe'],
            ['AromaticCarbonXSHydrophobe'],
            ['AromaticCarbonXSNonHydrophobe'],
            ['Nitrogen', 'NitrogenXSAcceptor'],
            ['NitrogenXSDonor', 'NitrogenXSDonorAcceptor'],
            ['Oxygen', 'OxygenXSAcceptor'],
            ['OxygenXSDonor', 'OxygenXSDonorAcceptor'],
            ['Sulfur', 'SulfurAcceptor'],
            ['Phosphorus']
        ]
        out_dict = {}
        generic = []
        for i, element_name in enumerate(self.atom_types):
            for types_list in types:
                if element_name in types_list:
                    out_dict[i] = types.index(types_list)
                    break
            if i not in out_dict.keys():
                generic.append(i)

        generic_type = len(types)
        for other_type in generic:
            out_dict[other_type] = generic_type
        return out_dict

    @staticmethod
    def read_file(infile, add_hydrogens, read_type='openbabel'):
        """Use openbabel to read in a pdb file.

        Original author: Constantin Schneider

        Args:
            infile (str): Path to input file
            add_hydrogens (bool): Add hydrogens to the openbabel OBMol object
            read_type: either biopython or openbabel

        Returns:
            pybel.Molecule
        """
        if read_type == 'biopython':
            parser = PDB.PDBParser()
            return parser.get_structure('receptor', infile)
        elif read_type == 'plip':
            mol = PDBComplex()
            mol.load_pdb(str(infile), as_string=False)
            if add_hydrogens:
                mol.protcomplex.OBMol.AddPolarHydrogens()
                mol.protcomplex.write('pdb', str(infile), overwrite=True)

            for ligand in mol.ligands:
                mol.characterize_complex(ligand)
            return mol
        else:
            molecules = []

            suffix = Path(infile).suffix[1:]
            file_read = pybel.readfile(suffix, str(infile))

            for mol in file_read:
                molecules.append(mol)

            if len(molecules) != 1:
                raise RuntimeError(
                    'More than one molecule detected in PDB file.')

            mol = molecules[0]

            if add_hydrogens:
                mol.OBMol.AddHydrogens()
            return mol

    @staticmethod
    def adjust_smina_type(t, h_bonded, hetero_bonded):
        """Original author: Constantin schneider"""
        if t in ('AliphaticCarbonXSNonHydrophobe',
                 'AliphaticCarbonXSHydrophobe'):  # C_C_C_P,
            if hetero_bonded:
                return 'AliphaticCarbonXSNonHydrophobe'
            else:
                return 'AliphaticCarbonXSHydrophobe'
        elif t in ('AromaticCarbonXSNonHydrophobe',
                   'AromaticCarbonXSHydrophobe'):  # C_A_C_P,
            if hetero_bonded:
                return 'AromaticCarbonXSNonHydrophobe'
            else:
                return 'AromaticCarbonXSHydrophobe'
        elif t in ('Nitrogen', 'NitogenXSDonor'):
            # N_N_N_P, no hydrogen bonding
            if h_bonded:
                return 'NitrogenXSDonor'
            else:
                return 'Nitrogen'
        elif t in ('NitrogenXSAcceptor', 'NitrogenXSDonorAcceptor'):
            # N_NA_N_A, also considered an acceptor by autodock
            if h_bonded:
                return 'NitrogenXSDonorAcceptor'
            else:
                return 'NitrogenXSAcceptor'
        elif t in ('Oxygen' or t == 'OxygenXSDonor'):  # O_O_O_P,
            if h_bonded:
                return 'OxygenXSDonor'
            else:
                return 'Oxygen'
        elif t in ('OxygenXSAcceptor' or t == 'OxygenXSDonorAcceptor'):
            # O_OA_O_A, also an autodock acceptor
            if h_bonded:
                return 'OxygenXSDonorAcceptor'
            else:
                return 'OxygenXSAcceptor'
        else:
            return t

    def obatom_to_smina_type(self, ob_atom):
        """Original author: Constantin schneider"""
        atomic_number = ob_atom.atomicnum
        num_to_name = {1: 'HD', 6: 'A', 7: 'NA', 8: 'OA', 16: 'SA'}

        # Default fn returns True, otherwise inspect atom properties
        condition_fns = defaultdict(lambda: lambda: True)
        condition_fns.update({
            6: ob_atom.OBAtom.IsAromatic,
            7: ob_atom.OBAtom.IsHbondAcceptor,
            16: ob_atom.OBAtom.IsHbondAcceptor
        })

        # Get symbol
        ename = openbabel.GetSymbol(atomic_number)

        # Do we need to adjust symbol?
        if condition_fns[atomic_number]():
            ename = num_to_name.get(atomic_number, ename)

        atype = self.string_to_smina_type(ename)

        h_bonded = False
        hetero_bonded = False
        for neighbour in openbabel.OBAtomAtomIter(ob_atom.OBAtom):
            if neighbour.GetAtomicNum() == 1:
                h_bonded = True
            elif neighbour.GetAtomicNum() != 6:
                hetero_bonded = True

        return self.adjust_smina_type(atype, h_bonded, hetero_bonded)

    def string_to_smina_type(self, string: str):
        """Convert string type to smina type.

        Original author: Constantin schneider

        Args:
            string (str): string type
        Returns:
            string: smina type
        """
        if len(string) <= 2:
            for type_info in self.atom_type_data:
                # convert ad names to smina types
                if string == type_info.adname:
                    return type_info.sm
            # find equivalent atoms
            for i in self.atom_equivalence_data:
                if string == i[0]:
                    return self.string_to_smina_type(i[1])
            # generic metal
            if string in self.non_ad_metal_names:
                return "GenericMetal"
            # if nothing else found --> generic metal
            return "GenericMetal"

        else:
            # assume it's smina name
            for type_info in self.atom_type_data:
                if string == type_info.smina_name:
                    return type_info.sm
            # if nothing else found, return numtypes
            # technically not necessary to call this numtypes,
            # but including this here to make it equivalent to the cpp code
            return "NumTypes"

    def download_pdbs_from_csv(self, csv, output_dir):
        output_dir = Path(output_dir).expanduser()
        pdbids = set()
        with open(csv, 'r') as f:
            for line in f.readlines():
                pdbids.add(line.split(',')[0].lower())
        cpus = mp.cpu_count()
        inputs = [(pdbid, output_dir / pdbid) for pdbid in pdbids
                  if not Path(output_dir, pdbid, 'receptor.pdb').is_file()]
        with mp.Pool(processes=cpus) as pool:
            pool.starmap(self.download_pdb_file, inputs)

    @staticmethod
    def download_pdb_file(pdbid, output_dir):
        """Given a PDB ID, downloads the corresponding PDB structure.
        Checks for validity of ID and handles error while downloading.
        Returns the path of the downloaded file (From PLIP)"""
        output_dir = Path(output_dir).expanduser()
        pdbpath = output_dir / 'receptor.pdb'
        if pdbpath.is_file():
            print(pdbpath, 'already exists.')
            return
        if len(pdbid) != 4 or extract_pdbid(
                pdbid.lower()) == 'UnknownProtein':
            raise RuntimeError('Unknown protein ' + pdbid)
        while True:
            try:
                pdbfile, pdbid = fetch_pdb(pdbid.lower())
            except urllib.error.URLError:
                print('Fetching pdb {} failed, retrying...'.format(
                    pdbid))
            else:
                break
        if pdbfile is None:
            print('None')
            return 'none'
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(pdbpath, 'w') as g:
            g.write(pdbfile)
        print('File downloaded as', pdbpath)
        return pdbpath

    def mol_calculate_interactions(self, mol, pl_interaction):
        """Return dataframe with interactions from plip mol object"""
        interaction_info = {}

        # Process interactions
        hbonds_rec_acceptors = pl_interaction.hbonds_ldon
        hbonds_rec_donators = pl_interaction.hbonds_pdon
        interaction_info['rec_acceptors'] = np.array([
            h.a_orig_idx for h in hbonds_rec_acceptors], dtype=np.int32)
        interaction_info['rec_donors'] = np.array([
            h.d_orig_idx for h in hbonds_rec_donators], dtype=np.int32)

        pi_lists = [interaction.proteinring.atoms_orig_idx for interaction
                    in pl_interaction.pistacking]
        interaction_info['pi_stacking'] = np.array([
            idx for idx_list in pi_lists for idx in idx_list],
            dtype=np.int32)

        hydrophobic_indices = np.array([
            interaction.bsatom_orig_idx for interaction
            in pl_interaction.hydrophobic_contacts], dtype=np.int32)
        interaction_info['hydrophobic'] = hydrophobic_indices

        all_ligand_indices = [list(ligand.can_to_pdb.values()) for ligand in
                              mol.ligands]
        all_ligand_indices = [idx for idx_list in all_ligand_indices for idx in
                              idx_list]

        return self.featurise_interaction(
            mol, interaction_info, all_ligand_indices)

    def featurise_interaction(self, mol, interaction_dict, all_ligand_indices):
        """Return dataframe with interactions from one particular plip site."""
        xs, ys, zs, types, atomic_nums, atomids = [], [], [], [], [], []
        keep_atoms = []
        for atomid, atom in mol.atoms.items():
            if atom.OBAtom.GetResidue().GetName().upper() in RESIDUE_IDS \
                    and atomid not in all_ligand_indices and \
                    atom.atomicnum > 1:
                keep_atoms.append(atomid)
            atomids.append(atomid)
            smina_type = self.obatom_to_smina_type(atom)
            if smina_type == "NumTypes":
                smina_type_int = len(self.atom_type_data)
            else:
                smina_type_int = self.atom_types.index(smina_type)
            type_int = self.type_map[smina_type_int]

            x, y, z = [float('{:.3f}'.format(i)) for i in atom.coords]
            xs.append(x)
            ys.append(y)
            zs.append(z)
            types.append(type_int)
            atomic_nums.append(atom.atomicnum)

        xs = np.array(xs, dtype=int)
        ys = np.array(ys, dtype=int)
        zs = np.array(zs, dtype=int)
        types = np.array(types, dtype=int)
        atomids = np.array(atomids, dtype=int)
        atomic_nums = np.array(atomic_nums, dtype=int)

        pistacking = np.zeros((len(types),), dtype=np.int32)
        hydrophobic = np.zeros_like(pistacking)
        hba = np.zeros_like(pistacking)
        hbd = np.zeros_like(pistacking)

        pistacking[interaction_dict['pi_stacking'] - 1] = 1
        hydrophobic[interaction_dict['hydrophobic'] - 1] = 1
        hba[interaction_dict['rec_acceptors'] - 1] = 1
        hbd[interaction_dict['rec_donors'] - 1] = 1

        mask = np.zeros_like(pistacking)
        mask[keep_atoms] = 1

        pistacking = pistacking[np.where(keep_atoms)]
        hydrophobic = hydrophobic[np.where(keep_atoms)]
        hba = hba[np.where(keep_atoms)]
        hbd = hbd[np.where(keep_atoms)]
        xs = xs[np.where(keep_atoms)]
        ys = ys[np.where(keep_atoms)]
        zs = zs[np.where(keep_atoms)]
        types = types[np.where(keep_atoms)]
        atomic_nums = atomic_nums[np.where(keep_atoms)]
        atomids = atomids[np.where(keep_atoms)]

        df = pd.DataFrame()
        df['atom_id'] = atomids
        df['x'] = xs
        df['y'] = ys
        df['z'] = zs
        df['atomic_number'] = atomic_nums

        df['types'] = types
        df['pistacking'] = pistacking
        df['hydrophobic'] = hydrophobic
        df['hba'] = hba
        df['hbd'] = hbd
        return df

    def calculate_interactions(self, pdbfile, hets, output_path,
                               remove_suspected_duplicates=True):
        """Write parquet files with interactions for each interaction site."""
        pdbfile = Path(pdbfile).expanduser()
        output_path = Path(output_path).expanduser()
        if Path(output_path / 'ligand_centres.yaml').is_file():
            print(pdbfile, 'has already been processed.')
            return

        output_path.mkdir(parents=True, exist_ok=True)
        mol = self.read_file(pdbfile, True, read_type='plip')
        interaction_info = defaultdict(dict)
        already_processed = set()
        data = namedtuple('interaction_set', 'pdbid molname df ligand_centre')

        for mol_name, pl_interaction in mol.interaction_sets.items():
            out_name = '{0}_{1}.parquet'.format(
                pdbfile.parent.name, mol_name.replace(':', '-'))
            if Path(output_path, out_name).exists():
                continue
            chunks = mol_name.split(':')
            het = ':'.join(chunks[:-1])  # Het id : chain id
            if het not in hets:
                # Some other ligand or metal ion we aren't interested in
                continue
            if remove_suspected_duplicates:
                # We don't want duplicates caused by n-mers
                if het in already_processed:
                    continue
                already_processed.add(het)
            print(pdbfile, mol_name)

            # Process interactions
            hbonds_rec_acceptors = pl_interaction.hbonds_ldon
            hbonds_rec_donators = pl_interaction.hbonds_pdon
            interaction_info[mol_name]['rec_acceptors'] = np.array([
                h.a_orig_idx for h in hbonds_rec_acceptors], dtype=np.int32)
            interaction_info[mol_name]['rec_donors'] = np.array([
                h.d_orig_idx for h in hbonds_rec_donators], dtype=np.int32)

            pi_lists = [interaction.proteinring.atoms_orig_idx for interaction
                        in pl_interaction.pistacking]
            interaction_info[mol_name]['pi_stacking'] = np.array([
                idx for idx_list in pi_lists for idx in idx_list],
                dtype=np.int32)

            hydrophobic_indices = np.array([
                interaction.bsatom_orig_idx for interaction
                in pl_interaction.hydrophobic_contacts], dtype=np.int32)
            interaction_info[mol_name]['hydrophobic'] = hydrophobic_indices
            interaction_info[mol_name]['ligand_indices'] = None

            for ligand in mol.ligands:
                lig_name = ligand.mol.title
                if lig_name == mol_name:
                    interaction_info[mol_name]['ligand_indices'] = np.array(
                        list(ligand.can_to_pdb.values()), dtype=np.int32)
                    interaction_info[mol_name][
                        'mean_ligand_coords'] = [
                        float('{:.3f}'.format(i)) for i in np.mean(
                            np.array([np.array(atom.coords) for
                                      atom in ligand.mol.atoms]), axis=0)]
                    break
            if interaction_info[mol_name]['ligand_indices'] is None:
                raise RuntimeError(
                    'No indexing information found for {}'.format(mol_name))

        all_ligand_indices = [
            list(ligand.can_to_pdb.values()) for ligand in mol.ligands]
        all_ligand_indices = [idx for idx_list in all_ligand_indices
                              for idx in idx_list]

        results = []

        for mol_name, info in interaction_info.items():
            df = self.featurise_interaction(mol, info, all_ligand_indices)
            results.append(data(
                pdbid=Path(pdbfile.parent.name).stem, molname=mol_name,
                df=df, ligand_centre=info['mean_ligand_coords']))

        if len(results):
            ligand_centres = {}
            for result in results:
                mol_name = '{0}_{1}'.format(
                    result.pdbid, result.molname.replace(':', '-'))
                out_name = mol_name + '.parquet'
                result.df.to_parquet(output_path / out_name)
                ligand_centres[mol_name] = result.ligand_centre

            with open(output_path / 'ligand_centres.yaml', 'w') as f:
                yaml.dump(ligand_centres, f)

    def parallel_process_directory(self, base_path, output_path, het_map):
        """Use multiprocessing to process all receptors in base_path."""
        base_path = Path(base_path).expanduser()
        all_pdbs = list(base_path.glob('**/receptor.pdb'))
        cpus = mp.cpu_count()
        output_paths = [Path(output_path, pdb.parent.name) for pdb in all_pdbs]
        hets = [het_map[pdb.parent.name] for pdb in all_pdbs]
        inputs = [(pdb, het, out) for pdb, het, out in zip(
            all_pdbs, hets, output_paths)]
        with mp.Pool(processes=cpus) as pool:
            pool.starmap(self.calculate_interactions, inputs)

    @staticmethod
    def get_het_map(pdb_list):
        het_map = defaultdict(set)
        with open(pdb_list, 'r') as f:
            for line in f.readlines():
                pdbid, het, chain = line.strip().split(',')
                if len(het) == 3:
                    het_map[pdbid.lower()].add(
                        ':'.join([het.upper(), chain.upper()]))
        return het_map

    def download_and_process(self, pdb_list, output_path):
        output_path = Path(output_path).expanduser()
        pdb_output = output_path / 'pdb'
        parquet_output = output_path / 'parquets'
        self.download_pdbs_from_csv(pdb_list, pdb_output)
        het_map = self.get_het_map(pdb_list)
        self.parallel_process_directory(pdb_output, parquet_output, het_map)

    def _test_single_file(self, pdbfile):
        pdbfile = Path(pdbfile).expanduser()
        hetmap = self.get_het_map('data/scpdb/binding_sites_fixed.csv')
        hetset = hetmap[pdbfile.parent.name]
        self.calculate_interactions(pdbfile, hetset, 'test')


if __name__ == '__main__':
    # dt = DistanceCalculator()
    # dt._test_single_file('data/scpdb/pdb/11bg/receptor.pdb')  # 11bg
    # exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('pdb_list', type=str,
                        help='CSV file containing PDB IDs of structures to be '
                             'processed')
    parser.add_argument('output_path', type=str,
                        help='Directory in which to save output')
    args = parser.parse_args()

    dt = DistanceCalculator()
    dt.download_and_process(args.pdb_list, args.output_path)
