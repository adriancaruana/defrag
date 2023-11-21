from pathlib import Path

from matplotlib import pyplot as plt


def thesis_figure_formatting():
    plt.style.use('ggplot')

def thesis_figure_save(directory: Path, name: str):
    plt.tight_layout()
    if not isinstance(directory, Path):
        directory = Path(directory)
    if "." in name: 
        raise ValueError("Name should not contain a file extension")
    plt.savefig(str(directory / name) + ".pdf")
    plt.savefig(str(directory / name) + ".png", dpi=300)
    plt.rcParams['svg.fonttype'] = 'path'
    plt.savefig(str(directory / name) + "_fontpath.svg")
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(str(directory / name) + "_fontnone.svg")


# 12 colours
cmap12 = plt.colormaps["Paired"].colors
cmap12 = cmap12[1::2] + cmap12[::2]
cmap12 *= 20
# 40 colours
cmap40 = plt.colormaps["tab20b"].colors + plt.colormaps["tab20c"].colors
cmap40 = cmap40[1::4] + cmap40[3::4] + cmap40[2::4] + cmap40[::4]
cmap40 *= 20
