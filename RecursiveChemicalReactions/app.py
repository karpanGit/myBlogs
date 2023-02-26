import matplotlib
matplotlib.use('Tkagg')

import matplotlib.pyplot as plt


from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdChemReactions

import pandas as pd
import math
from pathlib import Path
from typing import List, Tuple

# read the amino acids
f = r'input/amino_acids.smiles'
amino_acids = pd.read_csv(f, sep='|', header=0)
amino_acids['mol'] = amino_acids['smiles'].apply(Chem.MolFromSmiles)

# visualise the amino acids
def create_mol_grid(mols: List[Chem.Mol],
                    legends: List[str],
                    impath: Path,
                    n_cols: int = 4,
                    figsize: Tuple[int] = (4, 4),
                    dpi: int = 300) -> None:
    # set interactivity to off
    plt.ioff()
    n_rows = math.ceil(len(mols) / n_cols)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.subplots_adjust()
    axs = fig.subplots(nrows=n_rows, ncols=n_cols,
                       gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
    for i_mol, ax in enumerate(axs.flatten()):
        if i_mol < len(mols):
            plt.setp(ax, frame_on=False)
            plt.setp(plt.getp(ax, 'xaxis'), visible=False)
            plt.setp(plt.getp(ax, 'yaxis'), visible=False)

            mol_title = str(i_mol)
            h = ax.text(0.5, -.1, mol_title, ha="center", va="center",
                        rotation=0, size=2.0, bbox=None,
                        transform=ax.transAxes)
            im = Chem.Draw.MolToImage(mols[i_mol])
            ax.imshow(im)
        else:
            plt.setp(ax, visible=False)
    fig.savefig(impath)
    # set interactivity to on
    plt.ion()


# visualise the amino acids
mols = amino_acids['mol'].to_list()
legends = amino_acids['name'].to_list()
impath = Path('images') / 'amino_acids.png'
n_cols = 5
figsize = (4, 4)
dpi = 300
create_mol_grid(mols, impath, n_cols, figsize, dpi)

