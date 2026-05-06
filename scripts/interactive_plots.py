#!/usr/bin/env python3
"""Interactive plotting tool for ProXtal-LM training metrics.

Provides a fully interactive Jupyter widget interface for exploring
training metrics CSVs. Controls include data visibility toggles,
title/axis editing, colour/style pickers, legend/grid toggles,
axis-limit sliders, and a save-to-file button.

Environment requirements::

    pip install matplotlib ipywidgets pandas numpy
    # or with uv:
    uv pip install matplotlib ipywidgets pandas numpy

    # For JupyterLab ≥ 3 the widgets extension is bundled.
    # For classic Notebook you may need:
    jupyter nbextension enable --py widgetsnbextension

Usage (notebook)::

    %matplotlib widget          # recommended backend for interactivity
    from scripts.interactive_plots import launch
    launch('../checkpoints/v7_esmc_0/training_metrics.csv')
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
except ImportError as exc:
    raise ImportError(
        "ipywidgets and IPython are required. Install with:\n"
        "  uv pip install ipywidgets IPython"
    ) from exc


# ---------------------------------------------------------------------------
# Colour-blind-friendly palette (same as publication_graphs.py)
# ---------------------------------------------------------------------------

CB_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # cyan
    "#F0E442",  # yellow
    "#E69F00",  # amber
    "#000000",  # black
]

LINE_STYLES = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}
PLOT_MODES = ["line", "scatter", "both"]

# ---------------------------------------------------------------------------
# Metric groups — logical panels the user can toggle
# ---------------------------------------------------------------------------

METRIC_GROUPS: Dict[str, Dict[str, List[str]]] = {
    "Loss (train + val)": {
        "Train Loss": ["train_loss"],
        "Val Loss": ["val_loss"],
    },
    "Learning Rate": {
        "LR": ["lr"],
    },
    "Precision @ L / L2 / L5": {
        "Train P@L": ["train_precision_L"],
        "Train P@L/2": ["train_precision_L2"],
        "Train P@L/5": ["train_precision_L5"],
        "Val P@L": ["val_precision_L"],
        "Val P@L/2": ["val_precision_L2"],
        "Val P@L/5": ["val_precision_L5"],
    },
    "Recall @ L": {
        "Train Recall": ["train_recall_L"],
        "Val Recall": ["val_recall_L"],
    },
    "F1 @ L": {
        "Train F1": ["train_f1_L"],
        "Val F1": ["val_f1_L"],
    },
    "AUPRC": {
        "Train AUPRC": ["train_auprc"],
        "Val AUPRC": ["val_auprc"],
    },
    "Crystal-Only Precision": {
        "Train Cryst P@L": ["train_crystal_only_precision_L"],
        "Train Cryst P@L/2": ["train_crystal_only_precision_L2"],
        "Train Cryst P@L/5": ["train_crystal_only_precision_L5"],
        "Val Cryst P@L": ["val_crystal_only_precision_L"],
        "Val Cryst P@L/2": ["val_crystal_only_precision_L2"],
        "Val Cryst P@L/5": ["val_crystal_only_precision_L5"],
    },
    "Crystal-Only Accuracy": {
        "Train Cryst Acc": ["train_crystal_only_accuracy"],
        "Val Cryst Acc": ["val_crystal_only_accuracy"],
    },
    "Changed Positions (%)": {
        "Train % Changed": ["train_pct_changed_positions"],
        "Val % Changed": ["val_pct_changed_positions"],
    },
    "Hypothesis Losses (best)": {
        "Hyp 0 Best": ["val_hyp0_best_loss"],
        "Hyp 1 Best": ["val_hyp1_best_loss"],
        "Hyp 2 Best": ["val_hyp2_best_loss"],
    },
}


def _available_series(df: pd.DataFrame) -> Dict[str, str]:
    """Return {label: column} for every plottable column present in *df*.

    Automatically discovers hypothesis columns and prefixed multi-run
    columns (e.g. ``v7_esmc_0/val_loss``).
    """
    out: Dict[str, str] = {}
    # Standard (non-prefixed) columns from METRIC_GROUPS
    for _group_name, series_map in METRIC_GROUPS.items():
        for label, cols in series_map.items():
            for col in cols:
                if col in df.columns:
                    out[label] = col
    # Dynamically pick up any remaining numeric column (covers
    # prefixed multi-run columns and extra hypothesis columns)
    skip = {"epoch", "best_val_loss"}
    for col in sorted(df.columns):
        if col in skip or col in out.values():
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        # Build a readable label from the column name
        label = col.replace("/", " · ").replace("_", " ").strip().title()
        out[label] = col
    return out


# ---------------------------------------------------------------------------
# Widget builder
# ---------------------------------------------------------------------------


class InteractivePlotter:
    """Interactive matplotlib figure driven by ipywidgets."""

    def __init__(self, csv_path: str, default_show_series: bool = True) -> None:
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.series = _available_series(self.df)
        self.labels = list(self.series.keys())
        self._default_show_series = default_show_series

        # Assign default colours cycling through palette
        self._default_colours = {
            lbl: CB_PALETTE[i % len(CB_PALETTE)]
            for i, lbl in enumerate(self.labels)
        }
        self._default_styles = {lbl: "-" for lbl in self.labels}

        # ---- Build widgets ----

        # 1. Data visibility — multi-select with sensible defaults
        if default_show_series:
            default_visible = [
                l for l in self.labels
                if any(k in l.lower() for k in ("loss", "val loss", "train loss"))
            ]
            if not default_visible:
                default_visible = self.labels[:2]
        else:
            # Start with nothing visible (for multi-run to avoid slow load)
            default_visible = []

        # Scale height with the number of series (min 180px, max 400px)
        sel_height = f"{min(400, max(180, 20 * len(self.labels)))}px"

        self.w_series = widgets.SelectMultiple(
            options=self.labels,
            value=default_visible,
            description="Series:",
            layout=widgets.Layout(width="100%", height=sel_height),
            style={"description_width": "60px"},
        )

        # 2. Title controls
        self.w_title = widgets.Text(
            value="ProXtal-LM Training Metrics",
            description="Title:",
            layout=widgets.Layout(width="70%"),
        )
        self.w_show_title = widgets.Checkbox(value=True, description="Show title")

        # 3. Axis labels
        self.w_xlabel = widgets.Text(value="Epoch", description="X label:")
        self.w_ylabel = widgets.Text(value="Value", description="Y label:")

        # 4. Per-series colour, style, linewidth, plot mode, and legend-label widgets
        self.w_colours: Dict[str, widgets.ColorPicker] = {}
        self.w_styles: Dict[str, widgets.Dropdown] = {}
        self.w_linewidths: Dict[str, widgets.FloatSlider] = {}
        self.w_plot_modes: Dict[str, widgets.Dropdown] = {}
        self.w_legend_labels: Dict[str, widgets.Text] = {}
        for lbl in self.labels:
            self.w_colours[lbl] = widgets.ColorPicker(
                value=self._default_colours[lbl],
                description="Colour",
                style={"description_width": "55px"},
                layout=widgets.Layout(width="180px"),
            )
            self.w_styles[lbl] = widgets.Dropdown(
                options=list(LINE_STYLES.keys()),
                value="solid",
                description="Style",
                style={"description_width": "50px"},
                layout=widgets.Layout(width="140px"),
            )
            self.w_linewidths[lbl] = widgets.FloatSlider(
                value=1.5, min=0.3, max=6.0, step=0.1,
                description="Width",
                style={"description_width": "45px"},
                layout=widgets.Layout(width="170px"),
                readout_format=".1f",
            )
            self.w_plot_modes[lbl] = widgets.Dropdown(
                options=PLOT_MODES,
                value="line",
                description="Mode",
                style={"description_width": "42px"},
                layout=widgets.Layout(width="120px"),
            )
            self.w_legend_labels[lbl] = widgets.Text(
                value=lbl,
                description="Label",
                style={"description_width": "45px"},
                layout=widgets.Layout(width="240px"),
            )

        # Container for the dynamic style rows (only visible series)
        self.style_container = widgets.VBox([])

        # 5. Legend toggle
        self.w_legend = widgets.Checkbox(value=True, description="Show legend")

        # 6. Grid toggle
        self.w_grid = widgets.Checkbox(value=True, description="Show grid")

        # 6b. Typography controls (Universally safe generic and Matplotlib Native fonts)
        _font_options = [
            "sans-serif", "serif", "monospace", 
            "DejaVu Sans", "DejaVu Serif", "DejaVu Sans Mono"
        ]
        self.w_font_family = widgets.Dropdown(
            options=_font_options,
            value="sans-serif",
            description="Font:",
            style={"description_width": "50px"},
            layout=widgets.Layout(width="220px"),
        )
        self.w_title_size = widgets.IntSlider(
            value=13, min=2, max=30, description="Title size:",
            style={"description_width": "80px"},
            layout=widgets.Layout(width="280px"),
        )
        self.w_axis_label_size = widgets.IntSlider(
            value=11, min=2, max=26, description="Axis label:",
            style={"description_width": "80px"},
            layout=widgets.Layout(width="280px"),
        )
        self.w_tick_size = widgets.IntSlider(
            value=9, min=2, max=20, description="Tick size:",
            style={"description_width": "80px"},
            layout=widgets.Layout(width="280px"),
        )
        self.w_legend_size = widgets.IntSlider(
            value=9, min=2, max=20, description="Legend size:",
            style={"description_width": "80px"},
            layout=widgets.Layout(width="280px"),
        )

        # 7. Axis limit sliders
        epoch_min, epoch_max = int(self.df["epoch"].min()), int(self.df["epoch"].max())
        self.w_xlim = widgets.IntRangeSlider(
            value=[epoch_min, epoch_max],
            min=epoch_min,
            max=epoch_max,
            step=1,
            description="X range:",
            layout=widgets.Layout(width="90%"),
        )
        # Y limits — computed dynamically, start with auto
        self.w_ylim_auto = widgets.Checkbox(value=True, description="Auto Y-range")
        self.w_ylim = widgets.FloatRangeSlider(
            value=[0.0, 1.0],
            min=-0.5,
            max=10.0,
            step=0.01,
            description="Y range:",
            layout=widgets.Layout(width="90%"),
            readout_format=".3f",
        )

        # 8. Figure size controls (BoundedIntText for +/- style input)
        self.w_fig_width = widgets.BoundedFloatText(
            value=12, min=1, max=32, step=1,
            description="Width:",
            style={"description_width": "60px"},
            layout=widgets.Layout(width="130px"),
        )
        self.w_fig_height = widgets.BoundedFloatText(
            value=8, min=1, max=24, step=1,
            description="Height:",
            style={"description_width": "60px"},
            layout=widgets.Layout(width="130px"),
        )

        # 9. Grid layout controls (BoundedIntText for +/- style input)
        self.w_grid_rows = widgets.BoundedIntText(
            value=1, min=1, max=6, step=1,
            description="Rows:",
            style={"description_width": "50px"},
            layout=widgets.Layout(width="110px"),
        )
        self.w_grid_cols = widgets.BoundedIntText(
            value=1, min=1, max=6, step=1,
            description="Cols:",
            style={"description_width": "50px"},
            layout=widgets.Layout(width="110px"),
        )

        # Per-column X-axis sharing (share X within each column)
        self.w_share_x_label = widgets.HTML("<b>Share X-axis per Column:</b>")
        self.w_share_x_cols: List[widgets.Checkbox] = []
        for i in range(6):
            w = widgets.Checkbox(
                value=True,
                description=f"Col {i+1}",
                layout=widgets.Layout(width="80px"),
            )
            self.w_share_x_cols.append(w)

        # Per-row Y-axis sharing (share Y within each row)
        self.w_share_y_label = widgets.HTML("<b>Share Y-axis per Row:</b>")
        self.w_share_y_rows: List[widgets.Checkbox] = []
        for i in range(6):
            w = widgets.Checkbox(
                value=False,
                description=f"Row {i+1}",
                layout=widgets.Layout(width="80px"),
            )
            self.w_share_y_rows.append(w)

        # 9b. Column width ratios (for different-sized columns)
        self.w_col_ratios_label = widgets.HTML("<b>Column Width Ratios:</b>")
        self.w_col_ratios: List[widgets.BoundedIntText] = []
        for i in range(6):  # Max 6 columns
            w = widgets.BoundedIntText(
                value=1, min=1, max=10, step=1,
                description=f"Col {i+1}:",
                style={"description_width": "45px"},
                layout=widgets.Layout(width="100px"),
            )
            self.w_col_ratios.append(w)

        # 9c. Row height ratios (for different-sized rows)
        self.w_row_ratios_label = widgets.HTML("<b>Row Height Ratios:</b>")
        self.w_row_ratios: List[widgets.BoundedIntText] = []
        for i in range(6):  # Max 6 rows
            w = widgets.BoundedIntText(
                value=1, min=1, max=10, step=1,
                description=f"Row {i+1}:",
                style={"description_width": "45px"},
                layout=widgets.Layout(width="100px"),
            )
            self.w_row_ratios.append(w)

        # 10. Per-series subplot assignment (row/column selectors + span)
        self.w_subplot_row: Dict[str, widgets.BoundedIntText] = {}
        self.w_subplot_col: Dict[str, widgets.BoundedIntText] = {}
        self.w_subplot_rowspan: Dict[str, widgets.BoundedIntText] = {}
        self.w_subplot_colspan: Dict[str, widgets.BoundedIntText] = {}
        for lbl in self.labels:
            self.w_subplot_row[lbl] = widgets.BoundedIntText(
                value=1, min=1, max=1, step=1,
                description="",
                layout=widgets.Layout(width="45px"),
            )
            self.w_subplot_col[lbl] = widgets.BoundedIntText(
                value=1, min=1, max=1, step=1,
                description="",
                layout=widgets.Layout(width="45px"),
            )
            self.w_subplot_rowspan[lbl] = widgets.BoundedIntText(
                value=1, min=1, max=1, step=1,
                description="",
                layout=widgets.Layout(width="45px"),
            )
            self.w_subplot_colspan[lbl] = widgets.BoundedIntText(
                value=1, min=1, max=1, step=1,
                description="",
                layout=widgets.Layout(width="45px"),
            )

        # 10b. Per-subplot cell settings: X column, X label, Y label
        self._all_numeric_cols = sorted(
            [c for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c])],
            key=lambda c: (c != "epoch", c),  # epoch first
        )
        self.w_subplot_xcol: Dict[tuple, widgets.Dropdown] = {}
        self.w_subplot_xlabel: Dict[tuple, widgets.Text] = {}
        self.w_subplot_ylabel: Dict[tuple, widgets.Text] = {}
        for r in range(6):
            for c in range(6):
                key = (r, c)
                self.w_subplot_xcol[key] = widgets.Dropdown(
                    options=self._all_numeric_cols,
                    value="epoch",
                    description="X col:",
                    style={"description_width": "50px"},
                    layout=widgets.Layout(width="240px"),
                )
                self.w_subplot_xlabel[key] = widgets.Text(
                    value="",
                    placeholder="Epoch",
                    description="X label:",
                    style={"description_width": "55px"},
                    layout=widgets.Layout(width="200px"),
                )
                self.w_subplot_ylabel[key] = widgets.Text(
                    value="",
                    placeholder="Value",
                    description="Y label:",
                    style={"description_width": "55px"},
                    layout=widgets.Layout(width="200px"),
                )
        self.subplot_settings_container = widgets.VBox([])

        # 10c. Per-subplot panel labels (A, B, C, …) — editable
        _default_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [
            f"A{i}" for i in range(1, 11)
        ]
        self.w_subplot_panel_label: Dict[tuple, widgets.Text] = {}
        for r in range(6):
            for c in range(6):
                idx = r * 6 + c
                default_lbl = _default_labels[idx] if idx < len(_default_labels) else ""
                self.w_subplot_panel_label[(r, c)] = widgets.Text(
                    value=default_lbl,
                    description="Panel:",
                    style={"description_width": "45px"},
                    layout=widgets.Layout(width="100px"),
                )

        # Global panel-label controls
        self.w_show_panel_labels = widgets.Checkbox(
            value=False, description="Show panel labels",
        )
        self.w_panel_label_size = widgets.IntSlider(
            value=14, min=8, max=30, description="Panel size:",
            style={"description_width": "80px"},
            layout=widgets.Layout(width="250px"),
        )

        # 11. Bulk move controls
        self.w_bulk_series = widgets.SelectMultiple(
            options=self.labels,
            value=[],
            description="Series:",
            layout=widgets.Layout(width="250px", height="120px"),
            style={"description_width": "55px"},
        )
        self.w_bulk_row = widgets.BoundedIntText(
            value=1, min=1, max=1, step=1,
            description="To Row:",
            style={"description_width": "55px"},
            layout=widgets.Layout(width="90px"),
        )
        self.w_bulk_col = widgets.BoundedIntText(
            value=1, min=1, max=1, step=1,
            description="To Col:",
            style={"description_width": "55px"},
            layout=widgets.Layout(width="90px"),
        )
        self.w_bulk_move_btn = widgets.Button(
            description="Move Selected",
            button_style="info",
            icon="arrows-alt",
            layout=widgets.Layout(width="130px"),
        )
        self.w_bulk_move_btn.on_click(self._on_bulk_move)

        # 12. Quick assign buttons for common layouts
        self.w_assign_all_1x1 = widgets.Button(
            description="All to Grid 1",
            layout=widgets.Layout(width="110px"),
        )
        self.w_assign_all_1x1.on_click(lambda _: self._bulk_assign_all(1, 1))
        self.w_assign_split_trainval = widgets.Button(
            description="Train/Val Split",
            layout=widgets.Layout(width="110px"),
        )
        self.w_assign_split_trainval.on_click(self._assign_train_val_split)

        # 11. Save button with format selector
        self.w_save_format = widgets.Dropdown(
            options=["png", "pdf", "svg", "eps"],
            value="png",
            description="Format:",
            style={"description_width": "60px"},
            layout=widgets.Layout(width="120px"),
        )
        self.w_save_path = widgets.Text(
            value="interactive_plot",
            description="Filename:",
            layout=widgets.Layout(width="50%"),
        )
        self.w_save_dpi = widgets.IntSlider(
            value=300, min=72, max=600, step=1, description="DPI:"
        )
        self.w_save_btn = widgets.Button(
            description="Save Figure",
            button_style="success",
            icon="download",
        )
        self.w_save_btn.on_click(self._on_save)

        # 12. Template save/load controls
        self.w_template_name = widgets.Text(
            value="my_template",
            description="Name:",
            style={"description_width": "50px"},
            layout=widgets.Layout(width="150px"),
        )
        self.w_save_template_btn = widgets.Button(
            description="Save Template",
            button_style="warning",
            icon="save",
            layout=widgets.Layout(width="130px"),
        )
        self.w_save_template_btn.on_click(self._on_save_template)
        self.w_load_template = widgets.Dropdown(
            options=["-- select --"],
            value="-- select --",
            description="Load:",
            style={"description_width": "50px"},
            layout=widgets.Layout(width="200px"),
        )
        self.w_load_template.observe(self._on_load_template, names="value")
        self.w_delete_template_btn = widgets.Button(
            description="Delete",
            button_style="danger",
            icon="trash",
            layout=widgets.Layout(width="70px"),
        )
        self.w_delete_template_btn.on_click(self._on_delete_template)

        # Store templates
        self._templates: Dict[str, dict] = {}
        self.template_status = widgets.Output()

        # Load available templates from disk
        self._load_available_templates()

        # Output area for the plot
        self.out = widgets.Output()
        self.save_status = widgets.Output()

        # Build the figure once (will be rebuilt when grid changes)
        self.fig = None
        self.axes = None
        self._rebuild_figure()

    # ------------------------------------------------------------------ #
    # Figure Management
    # ------------------------------------------------------------------ #

    def _rebuild_figure(self, _change=None) -> None:
        """Rebuild the figure with current grid layout."""
        import matplotlib.gridspec as gridspec

        rows = self.w_grid_rows.value
        cols = self.w_grid_cols.value
        width = self.w_fig_width.value
        height = self.w_fig_height.value

        # Get column and row ratios
        col_ratios = [self.w_col_ratios[i].value for i in range(cols)]
        row_ratios = [self.w_row_ratios[i].value for i in range(rows)]

        # Close old figure if exists
        if self.fig is not None:
            plt.close(self.fig)

        # Create figure with GridSpec for custom sizes
        self.fig = plt.figure(figsize=(width, height), layout="constrained")
        self.gs = gridspec.GridSpec(
            rows, cols,
            width_ratios=col_ratios if cols > 1 else None,
            height_ratios=row_ratios if rows > 1 else None,
            figure=self.fig,
        )

        # Store grid dimensions
        self._grid_rows = rows
        self._grid_cols = cols

        plt.close(self.fig)

        # Update subplot row/col widget bounds
        for lbl in self.labels:
            self.w_subplot_row[lbl].max = rows
            self.w_subplot_col[lbl].max = cols
            self.w_subplot_rowspan[lbl].max = rows
            self.w_subplot_colspan[lbl].max = cols
            # Clamp values to valid range
            if self.w_subplot_row[lbl].value > rows:
                self.w_subplot_row[lbl].value = rows
            if self.w_subplot_col[lbl].value > cols:
                self.w_subplot_col[lbl].value = cols
            # Clamp spans too
            if self.w_subplot_rowspan[lbl].value > rows:
                self.w_subplot_rowspan[lbl].value = rows
            if self.w_subplot_colspan[lbl].value > cols:
                self.w_subplot_colspan[lbl].value = cols

        # Update bulk move controls
        self.w_bulk_row.max = rows
        self.w_bulk_col.max = cols

        # Trigger a re-render
        self._render()

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, _change=None) -> None:
        """Redraw the plot from widget state."""
        if self.fig is None:
            return

        # Clear figure completely (handles dynamic axes)
        self.fig.clf()

        # Typography
        font_family = self.w_font_family.value
        title_size = self.w_title_size.value
        axis_size = self.w_axis_label_size.value
        tick_size = self.w_tick_size.value
        legend_size = self.w_legend_size.value

        visible = list(self.w_series.value)
        x_lo, x_hi = self.w_xlim.value
        mask = (self.df["epoch"] >= x_lo) & (self.df["epoch"] <= x_hi)
        epochs = self.df.loc[mask, "epoch"]

        rows = self._grid_rows
        cols = self._grid_cols

        # Group series by subplot position (including span)
        # Key: (row, col, rowspan, colspan) -> list of series labels
        subplot_groups: Dict[tuple, List[str]] = {}
        for lbl in visible:
            row = self.w_subplot_row[lbl].value - 1  # 0-indexed
            col = self.w_subplot_col[lbl].value - 1
            rowspan = self.w_subplot_rowspan[lbl].value
            colspan = self.w_subplot_colspan[lbl].value
            key = (row, col, rowspan, colspan)
            if key not in subplot_groups:
                subplot_groups[key] = []
            subplot_groups[key].append(lbl)

        # Create axes for each unique subplot position with proper sharing
        # First pass: identify which columns/rows should share axes
        share_x_enabled = [self.w_share_x_cols[i].value if i < len(self.w_share_x_cols) else False for i in range(cols)]
        share_y_enabled = [self.w_share_y_rows[i].value if i < len(self.w_share_y_rows) else False for i in range(rows)]

        # Track first axis in each column for X-sharing
        col_first_ax: Dict[int, plt.Axes] = {}
        # Track first axis in each row for Y-sharing
        row_first_ax: Dict[int, plt.Axes] = {}

        axes_map: Dict[tuple, plt.Axes] = {}

        # Sort keys to ensure we create axes in order (important for sharing)
        sorted_keys = sorted(subplot_groups.keys())

        for key in sorted_keys:
            row, col, rowspan, colspan = key

            # Determine sharex/sharey parameters
            sharex_ax = None
            sharey_ax = None

            # If X-sharing is enabled for this column and we have a first axis
            if share_x_enabled[col] and col in col_first_ax:
                sharex_ax = col_first_ax[col]

            # If Y-sharing is enabled for this row and we have a first axis
            if share_y_enabled[row] and row in row_first_ax:
                sharey_ax = row_first_ax[row]

            # Create subplot with optional sharing
            ax = self.fig.add_subplot(
                self.gs[row:row+rowspan, col:col+colspan],
                sharex=sharex_ax,
                sharey=sharey_ax
            )
            axes_map[key] = ax

            # Record as first axis for this column/row if not already set
            if col not in col_first_ax:
                col_first_ax[col] = ax
            if row not in row_first_ax:
                row_first_ax[row] = ax

        # If nothing visible, create a single empty axes
        if not axes_map:
            ax = self.fig.add_subplot(self.gs[:, :])
            axes_map[(0, 0, rows, cols)] = ax

        # Track y ranges per subplot
        subplot_y_ranges: Dict[tuple, tuple] = {key: (float("inf"), float("-inf")) for key in axes_map}

        # Plot each series on its assigned subplot
        for key, series_list in subplot_groups.items():
            ax = axes_map[key]
            cell_key = (key[0], key[1])  # (row, col) for per-subplot settings

            # Per-subplot X column
            x_col = self.w_subplot_xcol[cell_key].value
            x_data = self.df.loc[mask, x_col] if x_col in self.df.columns else epochs

            for lbl in series_list:
                col_name = self.series[lbl]
                if col_name not in self.df.columns:
                    continue
                y = self.df.loc[mask, col_name]
                colour = self.w_colours[lbl].value
                ls = LINE_STYLES[self.w_styles[lbl].value]
                lw = self.w_linewidths[lbl].value
                plot_mode = self.w_plot_modes[lbl].value
                legend_label = self.w_legend_labels[lbl].value or lbl

                if plot_mode == "line":
                    ax.plot(x_data, y, label=legend_label, color=colour,
                            linestyle=ls, linewidth=lw)
                elif plot_mode == "scatter":
                    ax.scatter(x_data, y, label=legend_label, color=colour,
                               s=max(8, lw * 8), alpha=0.8, edgecolors="none")
                else:  # both
                    ax.plot(x_data, y, color=colour, linestyle=ls, linewidth=lw)
                    ax.scatter(x_data, y, label=legend_label, color=colour,
                               s=max(8, lw * 8), alpha=0.8, edgecolors="none")

                finite = y.replace([np.inf, -np.inf], np.nan).dropna()
                if not finite.empty:
                    y_min, y_max = subplot_y_ranges[key]
                    subplot_y_ranges[key] = (min(y_min, finite.min()), max(y_max, finite.max()))

        # Style each subplot
        for key, ax in axes_map.items():
            row, col, rowspan, colspan = key

            # Title (only on single grid, otherwise use suptitle)
            if len(axes_map) == 1 and self.w_show_title.value and self.w_title.value.strip():
                ax.set_title(self.w_title.value.strip(),
                             fontsize=title_size, fontweight="bold",
                             fontfamily=font_family)

            # Determine if X/Y should be shown based on per-column/row sharing
            is_bottom_row = row + rowspan >= rows
            is_left_col = col == 0

            # Check if this column has X sharing enabled
            x_shared = col < len(self.w_share_x_cols) and self.w_share_x_cols[col].value
            # Check if this row has Y sharing enabled
            y_shared = row < len(self.w_share_y_rows) and self.w_share_y_rows[row].value

            show_xlabel = is_bottom_row or not x_shared
            show_ylabel = is_left_col or not y_shared

            # Per-subplot labels (fall back to global default if empty)
            cell_key = (row, col)
            xlabel = (self.w_subplot_xlabel[cell_key].value.strip()
                      or self.w_xlabel.value)
            ylabel = (self.w_subplot_ylabel[cell_key].value.strip()
                      or self.w_ylabel.value)

            if show_xlabel:
                ax.set_xlabel(xlabel,
                              fontsize=axis_size, fontfamily=font_family)
            if show_ylabel:
                ax.set_ylabel(ylabel,
                              fontsize=axis_size, fontfamily=font_family)

            ax.tick_params(axis="both", labelsize=tick_size)
            for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
                tick_label.set_fontfamily(font_family)

            # Grid
            ax.grid(self.w_grid.value, alpha=0.3, linewidth=0.5)

            # Legend (only if this subplot has series)
            if self.w_legend.value and key in subplot_groups and subplot_groups[key]:
                ax.legend(loc="best", framealpha=0.8,
                          prop={"family": font_family, "size": legend_size})

            # Y limits
            if not self.w_ylim_auto.value:
                ax.set_ylim(self.w_ylim.value)
            elif key in subplot_y_ranges:
                y_min, y_max = subplot_y_ranges[key]
                if y_min < y_max:
                    margin = (y_max - y_min) * 0.05
                    ax.set_ylim(y_min - margin, y_max + margin)

            # Spine styling
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Add suptitle for multi-grid layouts
        if len(axes_map) > 1 and self.w_show_title.value and self.w_title.value.strip():
            self.fig.suptitle(self.w_title.value.strip(),
                              fontsize=title_size, fontweight="bold",
                              fontfamily=font_family)

        # Panel labels — added after layout so positions are correct.
        # Use ax.set_title with loc='left' which matplotlib places
        # reliably at the top-left of each axes.
        if self.w_show_panel_labels.value:
            panel_size = self.w_panel_label_size.value
            for key, ax in axes_map.items():
                cell_key = (key[0], key[1])
                panel_text = self.w_subplot_panel_label[cell_key].value.strip()
                if panel_text:
                    # Place using axes-transData offset in points so the
                    # horizontal position is identical regardless of
                    # subplot width (spanning columns).
                    from matplotlib.transforms import ScaledTranslation
                    offset = ScaledTranslation(
                        -15 / 72, 8 / 72,  # -15pt left, +8pt up
                        self.fig.dpi_scale_trans,
                    )
                    trans = ax.transAxes + offset
                    ax.text(
                        0, 1, panel_text,
                        transform=trans,
                        fontsize=panel_size,
                        fontweight="bold",
                        fontfamily=font_family,
                        va="bottom", ha="left",
                    )

        with self.out:
            clear_output(wait=True)
            display(self.fig)

    def _on_save(self, _btn) -> None:
        """Save the current figure to disk."""
        base_path = self.w_save_path.value.strip()
        if not base_path:
            base_path = "interactive_plot"
        fmt = self.w_save_format.value
        path = f"{base_path}.{fmt}"
        dpi = self.w_save_dpi.value
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        with self.save_status:
            clear_output(wait=True)
            print(f"✓ Saved to {os.path.abspath(path)}  ({dpi} DPI, {fmt.upper()})")

    # ------------------------------------------------------------------ #
    # Template Operations
    # ------------------------------------------------------------------ #

    def _get_template_state(self) -> dict:
        """Capture the complete graph state as a template."""
        # Per-series state
        series_state = {}
        for lbl in self.labels:
            series_state[lbl] = {
                "colour": self.w_colours[lbl].value,
                "style": self.w_styles[lbl].value,
                "linewidth": self.w_linewidths[lbl].value,
                "plot_mode": self.w_plot_modes[lbl].value,
                "legend_label": self.w_legend_labels[lbl].value,
                "subplot_row": self.w_subplot_row[lbl].value,
                "subplot_col": self.w_subplot_col[lbl].value,
                "subplot_rowspan": self.w_subplot_rowspan[lbl].value,
                "subplot_colspan": self.w_subplot_colspan[lbl].value,
            }

        # Per-subplot cell state
        subplot_cells = {}
        for r in range(6):
            for c in range(6):
                key = (r, c)
                xcol = self.w_subplot_xcol[key].value
                xlabel = self.w_subplot_xlabel[key].value
                ylabel = self.w_subplot_ylabel[key].value
                # Only store non-default values to keep JSON compact
                if xcol != "epoch" or xlabel or ylabel:
                    subplot_cells[f"{r},{c}"] = {
                        "xcol": xcol,
                        "xlabel": xlabel,
                        "ylabel": ylabel,
                    }
                # Panel label
                panel_lbl = self.w_subplot_panel_label[(r, c)].value
                cell_key_str = f"{r},{c}"
                if cell_key_str not in subplot_cells:
                    subplot_cells[cell_key_str] = {}
                subplot_cells[cell_key_str]["panel_label"] = panel_lbl

        # Panel label global settings
        panel_settings = {
            "show": self.w_show_panel_labels.value,
            "size": self.w_panel_label_size.value,
        }

        return {
            # Visible series
            "visible_series": list(self.w_series.value),
            # Layout
            "grid_rows": self.w_grid_rows.value,
            "grid_cols": self.w_grid_cols.value,
            "fig_width": self.w_fig_width.value,
            "fig_height": self.w_fig_height.value,
            "col_ratios": [w.value for w in self.w_col_ratios],
            "row_ratios": [w.value for w in self.w_row_ratios],
            "share_x_cols": [w.value for w in self.w_share_x_cols],
            "share_y_rows": [w.value for w in self.w_share_y_rows],
            # Title & global axis labels
            "title": self.w_title.value,
            "show_title": self.w_show_title.value,
            "xlabel": self.w_xlabel.value,
            "ylabel": self.w_ylabel.value,
            # Toggles
            "show_legend": self.w_legend.value,
            "show_grid": self.w_grid.value,
            # Typography
            "font_family": self.w_font_family.value,
            "title_size": self.w_title_size.value,
            "axis_label_size": self.w_axis_label_size.value,
            "tick_size": self.w_tick_size.value,
            "legend_size": self.w_legend_size.value,
            # Axis limits
            "xlim": list(self.w_xlim.value),
            "ylim_auto": self.w_ylim_auto.value,
            "ylim": list(self.w_ylim.value),
            # Per-series
            "series": series_state,
            # Per-subplot cell
            "subplot_cells": subplot_cells,
            # Panel labels
            "panel_settings": panel_settings,
        }

    def _apply_template(self, template: dict) -> None:
        """Apply a saved template, restoring the full graph state."""
        rows = template.get("grid_rows", 1)
        cols = template.get("grid_cols", 1)

        # --- Grid settings ---
        self.w_grid_rows.value = rows
        self.w_grid_cols.value = cols
        self.w_fig_width.value = template.get("fig_width", 12)
        self.w_fig_height.value = template.get("fig_height", 8)

        # Immediately update subplot widget bounds so that subsequent
        # value assignments are not clamped by stale max=1 constraints.
        for lbl in self.labels:
            self.w_subplot_row[lbl].max = rows
            self.w_subplot_col[lbl].max = cols
            self.w_subplot_rowspan[lbl].max = rows
            self.w_subplot_colspan[lbl].max = cols
        self.w_bulk_row.max = rows
        self.w_bulk_col.max = cols

        for i, v in enumerate(template.get("col_ratios", [1] * 6)):
            if i < len(self.w_col_ratios):
                self.w_col_ratios[i].value = v
        for i, v in enumerate(template.get("row_ratios", [1] * 6)):
            if i < len(self.w_row_ratios):
                self.w_row_ratios[i].value = v
        for i, v in enumerate(template.get("share_x_cols", [True] * 6)):
            if i < len(self.w_share_x_cols):
                self.w_share_x_cols[i].value = v
        for i, v in enumerate(template.get("share_y_rows", [False] * 6)):
            if i < len(self.w_share_y_rows):
                self.w_share_y_rows[i].value = v

        # --- Title & global labels ---
        self.w_title.value = template.get("title", "")
        self.w_show_title.value = template.get("show_title", True)
        self.w_xlabel.value = template.get("xlabel", "")
        self.w_ylabel.value = template.get("ylabel", "")
        self.w_legend.value = template.get("show_legend", True)
        self.w_grid.value = template.get("show_grid", True)

        # --- Typography ---
        self.w_font_family.value = template.get("font_family", "sans-serif")
        self.w_title_size.value = template.get("title_size", 13)
        self.w_axis_label_size.value = template.get("axis_label_size", 11)
        self.w_tick_size.value = template.get("tick_size", 9)
        self.w_legend_size.value = template.get("legend_size", 9)

        # --- Axis limits ---
        if "xlim" in template:
            lo, hi = template["xlim"]
            # Update slider bounds before setting the value
            self.w_xlim.min = min(self.w_xlim.min, lo)
            self.w_xlim.max = max(self.w_xlim.max, hi)
            self.w_xlim.value = [lo, hi]
        self.w_ylim_auto.value = template.get("ylim_auto", True)
        if "ylim" in template:
            self.w_ylim.value = template["ylim"]

        # --- Per-series state ---
        saved_series = template.get("series", {})
        for lbl in self.labels:
            if lbl not in saved_series:
                continue
            s = saved_series[lbl]
            self.w_colours[lbl].value = s.get("colour", self._default_colours[lbl])
            style_val = s.get("style", "solid")
            if style_val in LINE_STYLES:
                self.w_styles[lbl].value = style_val
            self.w_linewidths[lbl].value = s.get("linewidth", 1.5)
            mode_val = s.get("plot_mode", "line")
            if mode_val in PLOT_MODES:
                self.w_plot_modes[lbl].value = mode_val
            self.w_legend_labels[lbl].value = s.get("legend_label", lbl)
            self.w_subplot_row[lbl].value = s.get("subplot_row", 1)
            self.w_subplot_col[lbl].value = s.get("subplot_col", 1)
            self.w_subplot_rowspan[lbl].value = s.get("subplot_rowspan", 1)
            self.w_subplot_colspan[lbl].value = s.get("subplot_colspan", 1)

        # --- Per-subplot cell state ---
        saved_cells = template.get("subplot_cells", {})
        for rc_str, cell in saved_cells.items():
            r, c = (int(x) for x in rc_str.split(","))
            key = (r, c)
            if key in self.w_subplot_xcol:
                xcol = cell.get("xcol", "epoch")
                if xcol in self._all_numeric_cols:
                    self.w_subplot_xcol[key].value = xcol
                self.w_subplot_xlabel[key].value = cell.get("xlabel", "")
                self.w_subplot_ylabel[key].value = cell.get("ylabel", "")
                if "panel_label" in cell:
                    self.w_subplot_panel_label[key].value = cell["panel_label"]

        # --- Panel label global settings ---
        panel_s = template.get("panel_settings", {})
        self.w_show_panel_labels.value = panel_s.get("show", False)
        self.w_panel_label_size.value = panel_s.get("size", 14)

        # --- Visible series (set last so all assignments are ready) ---
        saved_visible = template.get("visible_series", [])
        valid_visible = tuple(lbl for lbl in saved_visible if lbl in self.labels)
        if valid_visible:
            self.w_series.value = valid_visible

    def _on_save_template(self, _btn) -> None:
        """Save current layout as a named template to file."""
        import json

        name = self.w_template_name.value.strip()
        if not name:
            with self.template_status:
                clear_output(wait=True)
                print("⚠ Please enter a template name")
            return

        # Create templates directory if it doesn't exist
        templates_dir = Path("templates")
        templates_dir.mkdir(exist_ok=True)

        # Save to JSON file
        template_path = templates_dir / f"{name}.json"
        template_data = self._get_template_state()

        with open(template_path, "w") as f:
            json.dump(template_data, f, indent=2)

        # Update in-memory cache and dropdown
        self._templates[name] = template_data
        self._refresh_template_dropdown()

        with self.template_status:
            clear_output(wait=True)
            print(f"✓ Template '{name}' saved to {template_path}")

    def _on_load_template(self, change) -> None:
        """Load a saved template from file."""
        import json

        name = change["new"]
        if name == "-- select --":
            return

        # Try to load from memory first, then from file
        if name in self._templates:
            template = self._templates[name]
        else:
            template_path = Path("templates") / f"{name}.json"
            if not template_path.exists():
                with self.template_status:
                    clear_output(wait=True)
                    print(f"⚠ Template '{name}' not found")
                return
            with open(template_path, "r") as f:
                template = json.load(f)
            self._templates[name] = template

        self._apply_template(template)
        with self.template_status:
            clear_output(wait=True)
            print(f"✓ Template '{name}' loaded")

    def _refresh_template_dropdown(self) -> None:
        """Refresh the template dropdown with available templates."""
        options = ["-- select --"] + list(self._templates.keys())
        self.w_load_template.options = options

    def _load_available_templates(self) -> None:
        """Load all available templates from the templates directory."""
        import json

        templates_dir = Path("templates")
        if not templates_dir.exists():
            return

        for template_file in templates_dir.glob("*.json"):
            name = template_file.stem
            try:
                with open(template_file, "r") as f:
                    self._templates[name] = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

        self._refresh_template_dropdown()

    # ------------------------------------------------------------------ #
    # Bulk Operations
    # ------------------------------------------------------------------ #

    def _on_bulk_move(self, _btn) -> None:
        """Move selected series to specified row/column."""
        selected = list(self.w_bulk_series.value)
        target_row = self.w_bulk_row.value
        target_col = self.w_bulk_col.value

        for lbl in selected:
            self.w_subplot_row[lbl].value = target_row
            self.w_subplot_col[lbl].value = target_col

        self._render()

    def _bulk_assign_all(self, row: int, col: int) -> None:
        """Assign all series to a specific grid cell."""
        for lbl in self.labels:
            self.w_subplot_row[lbl].value = row
            self.w_subplot_col[lbl].value = col
        self._render()

    def _assign_train_val_split(self, _=None) -> None:
        """Assign train series to grid 1, val series to grid 2."""
        for lbl in self.labels:
            lbl_lower = lbl.lower()
            if "train" in lbl_lower:
                self.w_subplot_row[lbl].value = 1
                self.w_subplot_col[lbl].value = 1
            elif "val" in lbl_lower or "valid" in lbl_lower:
                self.w_subplot_row[lbl].value = 1
                self.w_subplot_col[lbl].value = 2
            else:
                # Default non-train/val to grid 1
                self.w_subplot_row[lbl].value = 1
                self.w_subplot_col[lbl].value = 1
        self._render()

    def _on_delete_template(self, _btn) -> None:
        """Delete selected template from memory and disk."""
        name = self.w_load_template.value
        if name == "-- select --" or name not in self._templates:
            return

        # Remove from memory
        del self._templates[name]

        # Remove from disk
        template_path = Path("templates") / f"{name}.json"
        if template_path.exists():
            template_path.unlink()

        self._refresh_template_dropdown()
        self.w_load_template.value = "-- select --"

        with self.template_status:
            clear_output(wait=True)
            print(f"✓ Template '{name}' deleted")

    # ------------------------------------------------------------------ #
    # Layout & display
    # ------------------------------------------------------------------ #

    def _refresh_subplot_settings(self, _change=None) -> None:
        """Rebuild the per-subplot settings panel for active cells."""
        visible = list(self.w_series.value)
        # Find which cells are in use
        active_cells: set = set()
        for lbl in visible:
            r = self.w_subplot_row[lbl].value - 1
            c = self.w_subplot_col[lbl].value - 1
            active_cells.add((r, c))

        rows_ui = []
        for (r, c) in sorted(active_cells):
            key = (r, c)
            header = widgets.HTML(
                f"<b style='min-width:80px'>Subplot [{r+1}, {c+1}]</b>",
                layout=widgets.Layout(width="110px"),
            )
            rows_ui.append(
                widgets.HBox(
                    [header,
                     self.w_subplot_panel_label[key],
                     self.w_subplot_xcol[key],
                     self.w_subplot_xlabel[key],
                     self.w_subplot_ylabel[key]],
                    layout=widgets.Layout(margin="3px 0"),
                )
            )
        if not rows_ui:
            rows_ui = [widgets.HTML("<i>No active subplots — select series above.</i>")]
        self.subplot_settings_container.children = rows_ui

    def _refresh_style_panel(self, _change=None) -> None:
        """Rebuild the style accordion to show only selected series."""
        visible = list(self.w_series.value)
        rows_ui = []
        for lbl in visible:
            row_label = widgets.HTML(
                f"<b style='min-width:100px;display:inline-block'>{lbl}</b>",
                layout=widgets.Layout(width="110px"),
            )
            # Position labels
            r_label = widgets.HTML("R:", layout=widgets.Layout(width="12px"))
            c_label = widgets.HTML("C:", layout=widgets.Layout(width="12px"))
            rs_label = widgets.HTML("RS:", layout=widgets.Layout(width="18px"))
            cs_label = widgets.HTML("CS:", layout=widgets.Layout(width="18px"))
            rows_ui.append(
                widgets.HBox(
                    [row_label,
                     r_label, self.w_subplot_row[lbl],
                     c_label, self.w_subplot_col[lbl],
                     rs_label, self.w_subplot_rowspan[lbl],
                     cs_label, self.w_subplot_colspan[lbl],
                     self.w_legend_labels[lbl],
                     self.w_plot_modes[lbl],
                     self.w_colours[lbl], self.w_styles[lbl],
                     self.w_linewidths[lbl]],
                    layout=widgets.Layout(margin="2px 0"),
                )
            )
        if not rows_ui:
            rows_ui = [widgets.HTML("<i>Select series above to configure.</i>")]
        self.style_container.children = rows_ui

    def _build_ui(self) -> widgets.Widget:
        """Assemble the full widget layout."""
        # Populate the style panel with the initial selection
        self._refresh_style_panel()

        style_accordion = widgets.Accordion(
            children=[self.style_container],
            layout=widgets.Layout(width="100%"),
        )
        style_accordion.set_title(0, "Per-Series: Grid Assignment, Colours & Styles (selected only)")
        style_accordion.selected_index = None  # collapsed by default

        # Typography accordion
        typo_box = widgets.VBox([
            self.w_font_family,
            widgets.HBox([self.w_title_size, self.w_axis_label_size]),
            widgets.HBox([self.w_tick_size, self.w_legend_size]),
        ])
        typo_accordion = widgets.Accordion(
            children=[typo_box],
            layout=widgets.Layout(width="100%"),
        )
        typo_accordion.set_title(0, "Typography (font family & sizes)")
        typo_accordion.selected_index = None

        # Figure & Grid Layout accordion
        layout_box = widgets.VBox([
            widgets.HTML("<b>Figure Size</b>"),
            widgets.HBox([self.w_fig_width, self.w_fig_height]),
            widgets.HTML("<b>Grid Layout</b>"),
            widgets.HBox([self.w_grid_rows, self.w_grid_cols]),
            self.w_share_x_label,
            widgets.HBox(self.w_share_x_cols[:4]),
            widgets.HBox(self.w_share_x_cols[4:]),
            self.w_share_y_label,
            widgets.HBox(self.w_share_y_rows[:4]),
            widgets.HBox(self.w_share_y_rows[4:]),
            self.w_col_ratios_label,
            widgets.HBox(self.w_col_ratios[:4]),
            widgets.HBox(self.w_col_ratios[4:]),
            self.w_row_ratios_label,
            widgets.HBox(self.w_row_ratios[:4]),
            widgets.HBox(self.w_row_ratios[4:]),
        ])
        layout_accordion = widgets.Accordion(
            children=[layout_box],
            layout=widgets.Layout(width="100%"),
        )
        layout_accordion.set_title(0, "Figure Size & Grid Layout")
        layout_accordion.selected_index = None  # collapsed by default

        # Bulk move accordion
        bulk_box = widgets.VBox([
            widgets.HTML("<i>Select series below and choose target row/col</i>"),
            self.w_bulk_series,
            widgets.HBox([self.w_bulk_row, self.w_bulk_col, self.w_bulk_move_btn]),
            widgets.HTML("<b>Quick Assign:</b>"),
            widgets.HBox([self.w_assign_all_1x1, self.w_assign_split_trainval]),
        ])
        bulk_accordion = widgets.Accordion(
            children=[bulk_box],
            layout=widgets.Layout(width="100%"),
        )
        bulk_accordion.set_title(0, "Bulk Move Series Between Grids")
        bulk_accordion.selected_index = None  # collapsed by default

        # Per-subplot settings accordion
        self._refresh_subplot_settings()
        subplot_settings_accordion = widgets.Accordion(
            children=[self.subplot_settings_container],
            layout=widgets.Layout(width="100%"),
        )
        subplot_settings_accordion.set_title(
            0, "Per-Subplot Settings: Panel Label, X Column & Axis Labels"
        )
        subplot_settings_accordion.selected_index = None

        # Panel labels global controls
        panel_box = widgets.HBox([
            self.w_show_panel_labels,
            self.w_panel_label_size,
        ])

        # Controls panel
        controls = widgets.VBox([
            widgets.HTML("<h3>📊 ProXtal-LM Interactive Plotter</h3>"),
            widgets.HBox([
                widgets.VBox([
                    widgets.HTML("<b>Visible Series</b>"),
                    self.w_series,
                ], layout=widgets.Layout(width="40%")),
                widgets.VBox([
                    widgets.HTML("<b>Title & Default Axes Labels</b>"),
                    widgets.HBox([self.w_title, self.w_show_title]),
                    self.w_xlabel,
                    self.w_ylabel,
                    widgets.HTML("<i style='font-size:10px'>"
                                 "(Per-subplot labels override these — see accordion below)</i>"),
                    widgets.HBox([self.w_legend, self.w_grid]),
                ], layout=widgets.Layout(width="60%")),
            ]),
            widgets.HTML("<b>Axis Limits</b>"),
            self.w_xlim,
            widgets.HBox([self.w_ylim_auto, self.w_ylim]),
            layout_accordion,
            bulk_accordion,
            typo_accordion,
            style_accordion,
            panel_box,
            subplot_settings_accordion,
            widgets.HTML("<b>Templates (Save/Load Layout)</b>"),
            widgets.HBox([
                self.w_template_name, self.w_save_template_btn,
                self.w_load_template, self.w_delete_template_btn
            ]),
            self.template_status,
            widgets.HTML("<b>Export</b>"),
            widgets.HBox([self.w_save_format, self.w_save_path, self.w_save_dpi, self.w_save_btn]),
            self.save_status,
        ])

        return widgets.VBox([controls, self.out])

    def _observe_all(self) -> None:
        """Wire up all widgets to trigger a re-render."""
        simple = [
            self.w_series, self.w_title, self.w_show_title,
            self.w_xlabel, self.w_ylabel,
            self.w_legend, self.w_grid,
            self.w_xlim, self.w_ylim, self.w_ylim_auto,
            # Typography
            self.w_font_family, self.w_title_size,
            self.w_axis_label_size, self.w_tick_size, self.w_legend_size,
        ]
        for w in simple:
            w.observe(self._render, names="value")

        # Grid/layout controls that require figure rebuild
        rebuild_widgets = [
            self.w_fig_width, self.w_fig_height,
            self.w_grid_rows, self.w_grid_cols,
        ] + self.w_col_ratios + self.w_row_ratios + self.w_share_x_cols + self.w_share_y_rows
        for w in rebuild_widgets:
            w.observe(self._rebuild_figure, names="value")

        # Also refresh panels when the series selection changes
        self.w_series.observe(self._refresh_style_panel, names="value")
        self.w_series.observe(self._refresh_subplot_settings, names="value")
        for lbl in self.labels:
            self.w_colours[lbl].observe(self._render, names="value")
            self.w_styles[lbl].observe(self._render, names="value")
            self.w_linewidths[lbl].observe(self._render, names="value")
            self.w_plot_modes[lbl].observe(self._render, names="value")
            self.w_legend_labels[lbl].observe(self._render, names="value")
            self.w_subplot_row[lbl].observe(self._render, names="value")
            self.w_subplot_col[lbl].observe(self._render, names="value")
            self.w_subplot_rowspan[lbl].observe(self._render, names="value")
            self.w_subplot_colspan[lbl].observe(self._render, names="value")
            # Refresh per-subplot panel when series assignment changes
            self.w_subplot_row[lbl].observe(self._refresh_subplot_settings, names="value")
            self.w_subplot_col[lbl].observe(self._refresh_subplot_settings, names="value")

        # Per-subplot cell widgets trigger re-render
        for r in range(6):
            for c in range(6):
                key = (r, c)
                self.w_subplot_xcol[key].observe(self._render, names="value")
                self.w_subplot_xlabel[key].observe(self._render, names="value")
                self.w_subplot_ylabel[key].observe(self._render, names="value")
                self.w_subplot_panel_label[key].observe(self._render, names="value")

        # Panel label global controls
        self.w_show_panel_labels.observe(self._render, names="value")
        self.w_panel_label_size.observe(self._render, names="value")

    def show(self) -> None:
        """Display the interactive plotter in a Jupyter notebook."""
        self._observe_all()
        ui = self._build_ui()
        display(ui)
        # Initial render (figure already built in __init__)
        self._render()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def launch(csv_path: str = "checkpoints/v7_esmc_0/training_metrics.csv",
           default_show_series: bool = True) -> InteractivePlotter:
    """Create and display the interactive plotter.

    Args:
        csv_path: Path to a ProXtal-LM training_metrics.csv file.
        default_show_series: If True, show loss series by default.
                             If False, start with no series visible (faster for multi-run).

    Returns:
        The InteractivePlotter instance (for programmatic access).
    """
    plotter = InteractivePlotter(csv_path, default_show_series=default_show_series)
    plotter.show()
    return plotter


def launch_multi(csv_paths: Dict[str, str]) -> InteractivePlotter:
    """Launch with multiple runs merged into a single DataFrame.

    Starts with no series visible to avoid slow initial render.

    Args:
        csv_paths: Mapping of ``{run_name: csv_path}``.

    Returns:
        The InteractivePlotter instance.
    """
    frames = []
    for name, path in csv_paths.items():
        df = pd.read_csv(path)
        df = df.rename(columns={
            c: f"{name}/{c}" for c in df.columns if c != "epoch"
        })
        frames.append(df)
    if not frames:
        raise ValueError("No CSV files provided.")
    merged = frames[0]
    for other in frames[1:]:
        merged = merged.merge(other, on="epoch", how="outer")
    tmp_path = "/tmp/_proxtal_lm_merged_metrics.csv"
    merged.to_csv(tmp_path, index=False)
    # Start with no series visible for multi-run (faster load)
    return launch(tmp_path, default_show_series=False)


def render_from_template(
    csv_path: str,
    template_path: str,
    output_path: str = "figure.png",
    dpi: int = 300,
    fmt: Optional[str] = None,
) -> None:
    """Render a figure from a saved template without any GUI.

    Loads the CSV data and a JSON template, applies all saved state
    (series visibility, colours, layout, labels, etc.) and saves
    the resulting figure directly to disk.

    Args:
        csv_path: Path to a training_metrics.csv file.
        template_path: Path to a template JSON file (saved via the GUI).
        output_path: Where to save the figure. Extension determines format
                     unless *fmt* is given.
        dpi: Output resolution.
        fmt: Explicit format (``'png'``, ``'pdf'``, ``'svg'``, ``'eps'``).
             Defaults to the extension of *output_path*.

    Example::

        from scripts.interactive_plots import render_from_template
        render_from_template(
            'checkpoints/v7_esmc_0/training_metrics.csv',
            'templates/esmcv7.json',
            'figures/esmcv7_overview.pdf',
        )
    """
    import json
    import matplotlib
    matplotlib.use("Agg")  # headless backend

    with open(template_path, "r") as f:
        template = json.load(f)

    # Build plotter without showing it.
    # __init__ calls _rebuild_figure → _render with default state,
    # but that's fine — we'll rebuild after applying the template.
    plotter = InteractivePlotter(csv_path, default_show_series=False)

    # Apply template (updates all widget values + bounds)
    plotter._apply_template(template)

    # Rebuild figure with the template's grid/size settings,
    # which also triggers _render with the correct widget state.
    plotter._rebuild_figure()

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_kwargs: dict = {"dpi": dpi, "bbox_inches": "tight", "pad_inches": 0.1}
    if fmt:
        save_kwargs["format"] = fmt
    plotter.fig.savefig(output_path, **save_kwargs)
    plt.close(plotter.fig)
    print(f"Saved: {os.path.abspath(output_path)}  ({dpi} DPI)")