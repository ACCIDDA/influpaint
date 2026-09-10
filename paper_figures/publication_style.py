"""Shared final-size typography and export settings for the four main figures."""

import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib import ticker
from matplotlib.text import Text
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.axes_grid1.inset_locator import AnchoredSizeLocator
import seaborn as sns


PAPER_WIDTH_IN = 6.1
PAPER_HEIGHT_IN = 4.8
FIGURE1_LATEX_WIDTH_FRACTION = 0.75
WIDE_FIGURE_WIDTH_IN = 6.8
WIDE_FIGURE_HEIGHT_IN = 3.8
WIDE_FONT_SCALE = 6.8 / PAPER_WIDTH_IN
PAPER_FONT_PT = 7
FORECAST_DATE_FONT_PT = 5
PAPER_DPI = 600


def compact_count(value, position=None):
    """Format hospitalization counts with compact thousands."""
    if abs(value) >= 1000:
        return f'{value / 1000:.1f}'.rstrip('0').rstrip('.') + 'k'
    return f'{value:g}'


def save_paper_figure(fig, save_path, main_axes, *, calendar_dates=False,
                      mask_panels=False):
    """Export fixed canvases with typography sized for the manuscript text width."""
    main_axes = list(main_axes)
    ncols = len(main_axes) // 2
    width = WIDE_FIGURE_WIDTH_IN if ncols == 4 else PAPER_WIDTH_IN
    height = WIDE_FIGURE_HEIGHT_IN if ncols == 4 else PAPER_HEIGHT_IN
    font_scale = WIDE_FONT_SCALE if ncols == 4 else 1 / FIGURE1_LATEX_WIDTH_FRACTION
    font_size = PAPER_FONT_PT * font_scale
    fig.set_size_inches(width, height)
    fig.set_layout_engine(None)
    margins = dict(left=0.09, right=0.975, top=0.92,
                   bottom=0.17 if fig.legends else 0.12,
                   wspace=0.24 if ncols == 4 else 0.48,
                   hspace=0.38 if ncols == 4 else 0.40)
    fig.subplots_adjust(**margins)
    for gs in {ax.get_subplotspec().get_gridspec() for ax in main_axes}:
        gs.update(**margins)

    for text in fig.findobj(Text):
        text.set_fontsize(font_size)

    for ax in fig.axes:
        ax.tick_params(axis='both', labelsize=font_size, length=2.5 * font_scale,
                       width=0.6 * font_scale, pad=2 * font_scale)
        ax.tick_params(axis='y', labelsize=(PAPER_FONT_PT - 0.5) * font_scale)
        ax.tick_params(axis='x', labelsize=(PAPER_FONT_PT - 0.5) * font_scale)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6 * font_scale)
        for line in ax.lines:
            line.set_linewidth(max(0.4, line.get_linewidth() * 0.5) * font_scale)
        for patch in ax.patches:
            patch.set_linewidth(0.6 * font_scale)
        for gridline in ax.get_xgridlines() + ax.get_ygridlines():
            gridline.set_linewidth(0.35 * font_scale)

    for index, ax in enumerate(main_axes):
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, min_n_ticks=3))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(compact_count))
        ylabel = ax.get_ylabel()
        if ncols == 4:
            ax.set_ylabel('Incident flu hospitalizations' if index % ncols == 0 else '',
                          fontsize=(PAPER_FONT_PT - 0.5) * font_scale)
        elif ylabel == 'Incident flu hospitalizations':
            ax.set_ylabel(ylabel, fontsize=(PAPER_FONT_PT - 0.5) * font_scale)
        elif ylabel == 'Correlation across US states':
            ax.set_ylabel(ylabel, fontsize=(PAPER_FONT_PT - 0.5) * font_scale)
            ax.tick_params(axis='x', labelrotation=45)
            for label in ax.get_xticklabels():
                label.set_ha('right')
                label.set_rotation_mode('anchor')

        if calendar_dates:
            if ax.get_xlabel():
                dates = ax.lines[0].get_xdata(orig=False)
                valid_dates = dates[np.isfinite(dates)]
                ax.set_xlim(valid_dates[[0, -1]])
                ax.set_xlabel('')
            left, right = ax.get_xlim()
            ax.set_xticks(np.linspace(left, right, 4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%y'))
            ax.tick_params(axis='x', labelbottom=True)
            labels = ax.get_xticklabels()
            labels[0].set_ha('left')
            labels[-1].set_ha('right')
        elif ax.get_xlabel() or ylabel != 'Correlation across US states':
            ax.set_xticks([1, 17, 31, 44], ['Aug', 'Dec', 'Mar', 'Jun'])

        if mask_panels:
            panel_label = next(text for text in ax.texts if text.get_gid() == 'panel-label')
            state_label = next(text for text in ax.texts
                               if text.get_fontweight() == 'bold'
                               and text.get_gid() != 'panel-label')
            state_label.set_text(f'{panel_label.get_text()}. {state_label.get_text()}')
            panel_label.set_visible(False)

        for text in ax.texts:
            if text.get_gid() == 'panel-label':
                text.set_fontsize((PAPER_FONT_PT + 2) * font_scale)
                text.set_position((-0.15, 1.18 if ncols == 4 and not mask_panels else 1.12))
            elif text.get_fontweight() == 'bold':
                name = text.get_text()
                text.set_text(name)
                text.set_position((0, 1.035))
                text.set_va('bottom')
                text.set_bbox(None)

        reference_labels = [text for text in ax.texts if text.get_rotation() == 90]
        for text in reference_labels:
            text.set_visible(True)
            text.set_fontsize(FORECAST_DATE_FONT_PT * font_scale)
            text.set_text(datetime.strptime(text.get_text(), '%Y-%m-%d').strftime('%b %d'))
            text.set_ha('left')
        sns.despine(ax=ax, trim=False)

    inset_axes = [ax for ax in fig.axes if ax not in main_axes]
    if mask_panels:
        for parent, inset in zip(main_axes, inset_axes, strict=True):
            inset.set_axes_locator(AnchoredSizeLocator(
                (-0.01, 0.02, 1, 1), '41.2965%', '41.2965%',
                loc='upper right', bbox_transform=parent.transAxes))

    for index, ax in enumerate(inset_axes):
        if ax.lines:
            inset_transform = main_axes[index].transAxes + ScaledTranslation(
                5 / 25.4, 0, fig.dpi_scale_trans)
            ax.set_axes_locator(AnchoredSizeLocator(
                (0, 0, 0.98, 0.98), '43%', '43%', loc='upper right',
                bbox_transform=inset_transform))
            xticks = ax.get_xticks()[[0, -1]]
            yticks = ax.get_yticks()[[0, -1]]
            ax.set_xticks(xticks, [f'{int(t)}' for t in xticks])
            ax.set_yticks(yticks, [compact_count(int(t)) for t in yticks])
            sns.despine(ax=ax, trim=False)

    for ax in fig.axes:
        ax.spines['bottom'].set_bounds(*ax.get_xlim())
        ax.spines['left'].set_bounds(*ax.get_ylim())

    for legend in list(fig.legends):
        handles = legend.legend_handles
        for handle in handles:
            handle.set_linewidth(font_scale)
        labels = [text.get_text() for text in legend.get_texts()]
        title = legend.get_title().get_text()
        legend.remove()
        fig.legend(handles, labels, title=title, loc='upper center',
                   bbox_to_anchor=(0.40, 0.105), ncol=len(labels),
                   fontsize=font_size, title_fontsize=font_size,
                   handlelength=2.2, columnspacing=1.2, borderaxespad=0,
                   frameon=False)

    fig.align_ylabels(main_axes[::ncols])
    fig.canvas.draw()
    fig.savefig(save_path, dpi=PAPER_DPI, bbox_inches='tight', pad_inches=0.02)
