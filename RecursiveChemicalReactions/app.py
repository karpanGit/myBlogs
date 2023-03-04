import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdRGroupDecomposition
from rdkit.Chem import rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor

import pandas as pd
import numpy as np
import math
from pathlib import Path
from typing import List, Tuple, Optional, Union
import itertools

# read the amino acids
f = r'input/amino_acids.smiles'
amino_acids = pd.read_csv(f, sep='|', header=0)
amino_acids['mol'] = amino_acids['smiles'].apply(Chem.MolFromSmiles)
amino_acids = amino_acids.apply(lambda row: (row['mol'].SetProp('_Name', row['name']), row)[1], axis='columns')
print(Chem.MolToMolBlock(amino_acids['mol'].iloc[0]))

# create grid with molecular structures
def create_mol_grid(mols: List[Chem.Mol],
                    legends: Optional[List[str]],
                    impath: Path,
                    legend_offset: float = 0.1,
                    n_cols: int = 4,
                    figsize: Tuple[int] = (4, 4),
                    dpi: int = 300) -> None:
    '''
    It creates a gid of molecular structure images and saves the resulting
    figure.

    :param mols: an array of RDKit molecules
    :param legends: the legends to display under the molecules
    :param impath: the path to save the generated figure
    :param legend_offset: vertical offset of the legend
    :param n_cols: number of columns in the molecular image grid
    :param figsize: figure size
    :param dpi: figure resolution
    :return:
    '''
    # set interactivity to off
    plt.ioff()
    n_rows = math.ceil(len(mols) / n_cols)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.subplots_adjust()
    axs = fig.subplots(nrows=n_rows, ncols=n_cols,
                       gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
    for i_mol, ax in enumerate(np.atleast_1d(axs).flatten()):
        if i_mol < len(mols):
            plt.setp(ax, frame_on=False)
            plt.setp(plt.getp(ax, 'xaxis'), visible=False)
            plt.setp(plt.getp(ax, 'yaxis'), visible=False)
            # set the title
            if legends:
                mol_title = legends[i_mol]
                h = ax.text(0.5, legend_offset, mol_title, ha="center", va="center",
                            rotation=0, size=2.0, bbox=None,
                            transform=ax.transAxes)
            im = Chem.Draw.MolToImage(mols[i_mol])
            ax.imshow(im)
        else:
            plt.setp(ax, visible=False)
    fig.savefig(impath, bbox_inches='tight')
    plt.close(fig)
    # set interactivity to on
    plt.ion()


# visualise the amino acids
mols = amino_acids['mol'].to_list()
legends = (amino_acids['name']+'\ncommon: '+amino_acids['common']).to_list()
legend_offset = -0.1
impath = Path('images') / 'amino_acids.png'
n_cols = 5
figsize = (4, 5)
dpi = 600
create_mol_grid(mols, legends, impath, legend_offset, n_cols, figsize, dpi)


# R-group decomposition settings
ps = rdRGroupDecomposition.RGroupDecompositionParameters()
ps.onlyMatchAtRGroups = True

# set the core with explicit R group lables
amino_acid_backbone = '[*:1][C@H](N[*:2])C(O)=O'
amino_acid_backbone = Chem.MolFromSmiles(amino_acid_backbone)

# carry out the R-group decomposition
rgd, fails = rdRGroupDecomposition.RGroupDecompose([amino_acid_backbone], amino_acids['mol'].to_list(), options=ps)
successes = [i_mol for i_mol in range(len(amino_acids)) if i_mol not in fails]

# post-process the results for visualisation
RDecomposition_results = pd.concat([amino_acids.iloc[successes]['mol'].reset_index(drop=True), pd.DataFrame(rgd).iloc[:, 1:]], axis='columns', sort=False)
mols = RDecomposition_results.stack().to_list()
legends = amino_acids.iloc[successes]['name'].to_frame().assign(R1='R1', R2='R2').stack().to_list()
legend_offset = 0.
impath = Path('images') / 'amino_acids_R_decomposed.png'
n_cols = 6
figsize = (4, 8)
dpi = 600
create_mol_grid(mols, legends, impath, legend_offset, n_cols, figsize, dpi)


# substructure matching
# peptide
threonine_arginine_methionine = '[H]N[C@@H]([C@@H](C)O)C(=O)N[C@@H](CCCNC(N)=N)C(=O)N[C@@H](CCSC)C(O)=O'
threonine_arginine_methionine = Chem.MolFromSmiles(threonine_arginine_methionine)
# generate coordinates and orient canonically
rdDepictor.SetPreferCoordGen(True)
rdDepictor.Compute2DCoords(threonine_arginine_methionine)
# define peptide bond
peptide_bond_moiety = Chem.MolFromSmarts('NCC(=O)NCC(=O)')
d = rdMolDraw2D.MolDraw2DCairo(width=900, height=300) # or MolDraw2DSVG to get SVGs
d.drawOptions().setHighlightColour((0.8,0.8,0.8))
# matching atoms
hit_ats = threonine_arginine_methionine.GetSubstructMatches(peptide_bond_moiety)
hit_ats = list(itertools.chain(*hit_ats))
# matching bonds
hit_bonds = []
for bond in threonine_arginine_methionine.GetBonds():
    aid1 = bond.GetBeginAtomIdx()
    aid2 = bond.GetEndAtomIdx()
    if aid1 in hit_ats and aid2 in hit_ats:
        hit_bonds.append(bond.GetIdx())
# prepare molecule, draw molecule and create png
rdMolDraw2D.PrepareAndDrawMolecule(d, threonine_arginine_methionine, highlightAtoms=hit_ats,
                                   highlightBonds=hit_bonds)
with open('images/threonine_arginine_methionine.png', mode='wb') as f:
    f.write(d.GetDrawingText())


# define the reaction
peptide_hydrolysis_reaction = rdChemReactions.ReactionFromSmarts('[N:1][C:2][C:3](=[O:4])[N:5][C:6][C:7](=[O:8])>>[N:1][C:2][C:3](=[O:4])O.[N:5][C:6][C:7](=[O:8])')
# define the reactants
reacts = (threonine_arginine_methionine,)
# apply the reaction
products = peptide_hydrolysis_reaction.RunReactants(reacts, maxProducts=1000)
products = list(itertools.chain(*products))
rdDepictor.SetPreferCoordGen(True)
for product in products:
    Chem.SanitizeMol(product)
    rdDepictor.Compute2DCoords(product)
# draw the reactants in a grid
legends = None
legend_offset = 0.1
impath = Path('images') / 'reactants.png'
n_cols = 2
figsize = (1, 1)
dpi = 600
create_mol_grid(products, legends, impath, legend_offset, n_cols, figsize, dpi)


# apply the hydrolysis reaction recursively
def check_identity(mol1: Chem.rdchem.Mol, mol2: Chem.rdchem.Mol) -> bool:
    '''
    Checks if two rdkit molecules are identical
    :param mol1: first molecule
    :param mol2: second molecule
    :return: True if identical and False otherwise
    '''
    return all([mol1.HasSubstructMatch(mol2, useChirality=True),
                mol2.HasSubstructMatch(mol1, useChirality=True)])


def check_if_in_list(mol: Chem.rdchem.Mol, mols: List[Chem.rdchem.Mol]) -> bool:
    '''
    Checks if a molecule is identical to molecule in a list
    :param mol: molecule to be checked if present in a list
    :param mols: list of molecules
    :return: True if molecule is identical to a molecule in a list and False otherwise
    '''
    for mol_ in mols:
        if check_identity(mol, mol_):
            return True
    else:
        return False

def is_peptide(mol: Chem.rdchem.Mol,
               known_amino_acids: List[Chem.rdchem.Mol],
               rxn: Chem.rdChemReactions.ChemicalReaction) -> Union[Chem.rdchem.Mol, bool]:
    '''
    Takes a rdkit molecule and the definition of a peptide bond hydrolysis reaction and examines if the molecule is
    a linear peptide by applying the reaction recursively
    :param mol: molecule that will be checked if it is a peptide or not
    :param known_amino_acids: list of known amino acid molecules
    :param rxn: peptide hydrolysis reaction
    :return: True if molecule is a peptide is and False otherwise
    '''
    print(f'processing structure {Chem.MolToSmiles(mol)}')

    # check if peptide is already a single amino acid
    if check_if_in_list(mol, known_amino_acids):
        print(f'structure {Chem.MolToSmiles(mol)} is an amino acid')
        return True

    # apply the hydrolysis reaction, we only follow the first hydrolysis reaction as we
    # will apply the hydrolysis recursively anyway
    reacts = (mol,)
    products = rxn.RunReactants(reacts, maxProducts=1)
    # peptide hydrolysis could not be applied
    if not products:
        print('peptide bond hydrolysis could not be applied')
        return False
    # peptide hydrolysis was applied successfully
    else:
        # reactants of the hydrolysis
        mol1 = products[0][0]
        Chem.SanitizeMol(mol1)
        mol2 = products[0][1]
        Chem.SanitizeMol(mol2)
        print(f'applied hydrolysis reaction: {Chem.MolToSmiles(mol)} -> {Chem.MolToSmiles(mol1)} + {Chem.MolToSmiles(mol2)}')

        return all([is_peptide(mol1, known_amino_acids, rxn), is_peptide(mol2, known_amino_acids, rxn)])


# 5 amino acids
structures = amino_acids[['name', 'mol']].assign(type='amino acid').iloc[:5]
# di-peptide: arginine, alanine
smi = '[H]N[C@@H](CCCNC(N)=N)C(=O)N[C@@H](C)C(O)=O'
mol = Chem.MolFromSmiles(smi)
structures = pd.concat([structures, pd.Series({'name': 'arginine, alanine', 'mol': mol, 'type': 'di-peptide'}).to_frame().T], axis='index', ignore_index=True, sort=False)
# oligo-peptide: arginine, alanine, threonine, methionine
smi = '[H]N[C@@H](CCCNC(N)=N)C(=O)N[C@@H](C)C(=O)N[C@@H]([C@@H](C)O)C(=O)N[C@@H](CCSC)C(O)=O'
mol = Chem.MolFromSmiles(smi)
structures = pd.concat([structures, pd.Series({'name': 'arginine, alanine, threonine, methionine', 'mol': mol, 'type': 'oligo-peptide'}).to_frame().T], axis='index', ignore_index=True, sort=False)
# longer-peptide: ATTAMSSTA
smi = 'CSCC[C@H](NC(=O)[C@H](C)NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@H](C)N)[C@@H](C)O)' \
      '[C@@H](C)O)C(=O)N[C@@H](CO)C(=O)N[C@@H](CO)C(=O)N[C@@H]([C@@H](C)O)C(=O)N[C@@H](C)C(O)=O'
mol = Chem.MolFromSmiles(smi)
structures = pd.concat([structures, pd.Series({'name': 'ATTAMSSTA', 'mol': mol, 'type': 'longer peptide'}).to_frame().T], axis='index', ignore_index=True, sort=False)
# non-peptide 1
smi = '[H]N[C@@H](CCCNC(N)=N)C(=O)N[C@@H](C)C(O)'
mol = Chem.MolFromSmiles(smi)
structures = pd.concat([structures, pd.Series({'name': 'non-peptide 1', 'mol': mol, 'type': 'non-peptide'}).to_frame().T], axis='index', ignore_index=True, sort=False)
# non-peptide 2
smi = '[H]N[C@@H](CCCNC(N)=N)CC(=O)N[C@@H](C)C(O)=O'
mol = Chem.MolFromSmiles(smi)
structures = pd.concat([structures, pd.Series({'name': 'non-peptide 2', 'mol': mol, 'type': 'non-peptide'}).to_frame().T], axis='index', ignore_index=True, sort=False)

# examine if structures are peptides
structures['result'] = structures['mol'].apply(lambda mol: is_peptide(mol, known_amino_acids=amino_acids['mol'].to_list(), rxn=peptide_hydrolysis_reaction))

# visualise the results
mols = structures['mol'].to_list()
legends = (structures['type']+'\nis peptide: '+structures['result'].astype(str)).to_list()
legend_offset = -0.1
impath = Path('images') / 'is_peptide_application.png'
n_cols = 4
figsize = (4, 4)
dpi = 600
create_mol_grid(mols, legends, impath, legend_offset, n_cols, figsize, dpi)


