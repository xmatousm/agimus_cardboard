from typing import Optional
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

def segments(seg: list, ax: Optional[Axes] = None,
             linewidth=5, rot=None, t=None, color=None):

    if ax is None:
        ax = plt.gca()

    for s in seg:
        u1 = s.u1
        u2 = s.u2

        if rot is not None:
            u1 = rot @ u1 + t
            u2 = rot @ u2 + t

        ax.plot((u1[0], u2[0]), (u1[1], u2[1]), linewidth=linewidth,
                color=color)
