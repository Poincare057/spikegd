"""Shared style so every figure reads at a glance at its printed size (text width 5.5 in, fonts 8-9 pt)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'font.family': 'serif', 'mathtext.fontset': 'cm',
    'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.2,
    'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
WIDTH = 5.5                               # inches, NeurIPS text width
# colour-blind safe (Okabe-Ito)
BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERM, PURPLE = '#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7'
GREY = '0.6'
def save(fig, name):
    fig.savefig(name + '.pdf'); fig.savefig(name + '.png', dpi=200)
    fig.savefig(name + '.svg')
