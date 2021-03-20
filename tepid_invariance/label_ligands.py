import argparse
import os
import warnings
from collections import defaultdict
from pathlib import Path

from Bio import PDB as PDB
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit import Chem, RDLogger, RDConfig
from rdkit.Chem import ChemicalFeatures

from tepid_invariance.preprocessing.pdb_to_parquet import get_positions, \
    get_aromatic_indices, get_hbd_indices, get_hba_indices

try:
    from openbabel import pybel
except ImportError:
    import pybel


class PDBStringParser(PDB.PDBParser):
    """Parse from text block instead of file."""

    def get_structure(self, structure_id, pdb_block):
        """Return the structure.

        Arguments:
            structure_id: string, the id that will be used for the structure
            pdb_block: text block with PDB file structure
        """
        with warnings.catch_warnings():
            if self.QUIET:
                warnings.filterwarnings("ignore",
                                        category=PDBConstructionWarning)

            self.header = None
            self.trailer = None
            # Make a StructureBuilder instance (pass id of structure as
            # parameter)
            self.structure_builder.init_structure(structure_id)

            lines = pdb_block.split('\n')
            self._parse(lines)

            self.structure_builder.set_header(self.header)
            # Return the Structure instance
            structure = self.structure_builder.get_structure()

        return structure


def label_ligand(
        rdkit_mol, output_fname, atom_condition, factory,
        pdb_parser=PDBStringParser()):
    """Save rdkit molecule as PDB file, with aromatics labelled.

    Aromatic atoms are given a b-factor of 1, with the rest 0. This can be
    vislialised in pymol, using the following command:

        color pink, b > 0.5

    Arguments:
        rdkit_mol: rdkit molecule object (rdkit.Chem.Mol())
        output_fname: where to save output pdb file
        atom_condition: 'acceptor', 'donor' or 'aromatic', condition for atoms
            to be labelled as 1 rather than 0
        factory: rdkit
        pdb_parser: PDBStringParser object
    """
    filter_fn = {
        'aromatic': get_aromatic_indices,
        'hba': get_hba_indices,
        'hbd': get_hbd_indices
    }[atom_condition]
    positive_indices = filter_fn(rdkit_mol, factory)
    coords = get_positions(rdkit_mol)

    # Indexing in biopython is different to rdkit and this is the only way to
    # do the book-keeping
    pos_to_aromaticity = defaultdict(lambda: defaultdict(dict))
    for idx in range(len(coords)):
        x, y, z = [str(i) for i in coords[idx, :]]
        pos_to_aromaticity[x][y][z] = idx in positive_indices

    # biopython can only read pdb format, so create pdb block then parse that
    pdb_str = Chem.rdmolfiles.MolToPDBBlock(rdkit_mol)
    bp_structure = pdb_parser.get_structure('mol', pdb_str)
    for atom in bp_structure.get_atoms():
        x, y, z = [str(i) for i in atom.get_coord()]
        try:
            aromatic = pos_to_aromaticity[x][y][z]
        except KeyError:
            print('Position', x, y, z, 'not found. Output {} failed'.format(
                output_fname))
            return
        atom.set_bfactor(int(aromatic))

    io = PDB.PDBIO()
    io.set_structure(bp_structure)
    io.save(str(Path(output_fname).expanduser()))


def sdf_to_labelled_pdb(sdf_filename, output_path, atom_condition):
    """Label aromaticity and convert to pdb all molecules in an sdf file.

    Aromatic atoms are given a b-factor of 1, with the rest 0. This can be
    vislialised in pymol, using the following command:

        color pink, b > 0.5

    Arguments:
        sdf_filename: sdf file comtaining one or more molecules
        output_path: directory in which pdb files are dumped (one per molecule
            in sdf_filename)
        atom_condition: 'hba', 'hbd' or 'aromatic'
    """
    output_path = Path(output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    sdf_filename = Path(sdf_filename).expanduser()
    ligands = Chem.SDMolSupplier(str(Path(sdf_filename).expanduser()))
    pdb_parser = PDBStringParser()
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    for ligand in ligands:
        if ligand is None:
            continue
        output_fname = Path(output_path, ligand.GetProp('_Name') + '.pdb')
        label_ligand(
            ligand, output_fname, atom_condition, factory=factory,
            pdb_parser=pdb_parser)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ligands', type=str,
                        help='SDF file containing ligand coordinates (possibly '
                             'multiple molecules)')
    parser.add_argument('filter', type=str,
                        help='Atoms will be labelled 1 if they are classed as '
                             'belonging to the filter, and zero otherwise. Can '
                             'be any of "hba", "hbd", or "aromatic"')
    parser.add_argument('output_path', type=str,
                        help='Directory in which to save output')
    args = parser.parse_args()

    RDLogger.DisableLog('*')
    pybel.ob.obErrorLog.SetOutputLevel(0)

    sdf_to_labelled_pdb(args.ligands, args.output_path, args.filter)
