from typing import Optional

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from . import crbtools as crb


def segments(seg: list, ax_: Optional[Axes] = None,
             linewidth=5, rot=None, t=None, color=None):
    ax = ax_ or plt.gca()
    x_lim = [100000.0, -100000.0]
    y_lim = [100000.0, -100000.0]

    for s in seg:
        u1 = s.u1
        u2 = s.u2

        if rot is not None:
            u1 = rot@u1+t
            u2 = rot@u2+t

        ax.plot((u1[0], u2[0]), (u1[1], u2[1]), linewidth=linewidth,
                color=color)
        x_lim[0] = min(x_lim[0], min(u1[0], u2[0]))
        x_lim[1] = max(x_lim[1], max(u1[0], u2[0]))
        y_lim[0] = min(y_lim[0], min(u1[1], u2[1]))
        y_lim[1] = max(y_lim[1], max(u1[1], u2[1]))

    return x_lim, y_lim

def pairs(pairs: list, ax_: Optional[Axes] = None, seg_linewidth=5,
          pair_linewidth: int = 1, color=None, rot=None, t=None,):

    ax = ax_ or plt.gca()

    for pr in pairs:
        s1_u1, s1_u2 = pr[3].u1, pr[3].u2
        s2_u1, s2_u2 = pr[4].u1, pr[4].u2

        if rot is not None:
            s1_u1 = rot@s1_u1+t
            s1_u2 = rot@s1_u2+t
            s2_u1 = rot@s2_u1+t
            s2_u2 = rot@s2_u2+t

        plt.plot((s1_u1[0], s1_u2[0]), (s1_u1[1], s1_u2[1]), color=color, linewidth=seg_linewidth)
        plt.plot((s2_u1[0], s2_u2[0]), (s2_u1[1], s2_u2[1]), color=color, linewidth=seg_linewidth)
        mid1 = (s1_u1+s1_u2)/2
        mid2 = (s2_u1+s2_u2)/2
        plt.plot((mid1[0], mid2[0]), (mid1[1], mid2[1]), color=color, linewidth=pair_linewidth)

    ax.set_ylim(*sorted(ax.get_ylim(), reverse=True))
    ax.set_aspect('equal')


def template(tmpl: 'crb.Template', ax_: Optional[Axes] = None,
             rot=None, t=None, points: bool=True):
    """Visualize a template."""

    ax = ax_ or plt.gca()

    if points:
        for h in tmpl.hole_pt:
            ax.plot(h[0], h[1], '.')

        for h in tmpl.nb_hole_pt:
            ax.plot(h[0], h[1], 'o')

    for i in range(len(tmpl.hole_line)):
        h = tmpl.hole_line[i]
        if rot is not None:
            h = rot @ h + t

        ax.plot(h[0], h[1], 'r--', linewidth=2)
        p, p_ngh = tmpl.hole_polygon[i]

        if rot is not None:
            p = rot @ p + t
            p_ngh = rot @ p_ngh + t

        ax.plot(p[0], p[1], 'r', linewidth=1)
        ax.plot(p_ngh[0], p_ngh[1], 'k', linewidth=1)

    segments(tmpl.seg, ax, rot=rot, t=t, linewidth=3, color='g')
    x_lim, y_lim = segments(tmpl.seg, ax, rot=rot, t=t, linewidth=1, color='w')

    ax.set_ylim(*sorted(ax.get_ylim(), reverse=True))
    ax.set_aspect('equal')

    return x_lim, y_lim

def template_metric(tmpl: 'crb.TemplateMetric',
                    ax_: Optional[Axes] = None) -> None:
    """Visualize a metric template."""

    ax = ax_ or plt.gca()

    for s in tmpl.seg:
        u1 = s[:, 0]
        u2 = s[:, 1]

        ax.plot((u1[0], u2[0]), (u1[1], u2[1]), 'b', linewidth=2)

    for i in range(len(tmpl.hole_line)):
        h = tmpl.hole_line[i]
        ax.plot(h[0], h[1], 'r--', linewidth=2)
        p, p_ngh = tmpl.hole_polygon(i)
        ax.plot(p[0], p[1], 'r', linewidth=1)
        ax.plot(p_ngh[0], p_ngh[1], 'k', linewidth=1)

    ax.set_ylim(*sorted(ax.get_ylim(), reverse=True))
    ax.set_aspect('equal')
