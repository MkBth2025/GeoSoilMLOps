import os
import re
import argparse
import sys
import shutil
import subprocess
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import (
    ttk,
    filedialog,
    messagebox,
    simpledialog,
)

import pandas as pd
import yaml

from project_config import (
    DEFAULT_PATHS,
    load_params as load_project_params,
    resolve_project_paths,
    update_paths_in_params,
    write_install_txt,
)


# ============================================================
# PROJECT PATHS
# ============================================================

# The project root defaults to the directory from which the GUI was
# launched.  The user can change it at runtime with the Current MLOps
# folder browser in the main window.
BASE_DIR = Path.cwd().resolve()

# Keep the original --params argument so a relative params filename can
# be re-resolved whenever the user changes the project root.
PARAMS_ARGUMENT = "params.yaml"

DEFAULT_DATASET = None
ALL_EVALUATIONS = None
CLASS_ALL_EVALUATIONS = None
REG_REPORT_DIR = None
CLASS_REPORT_DIR = None
DQ_REPORT_DIR = None
MULTICOLLINEARITY_REPORT_DIR = None
PROCESSED_DATA_DIR = None
REG_MODELS_DIR = None
CLASS_MODELS_DIR = None
INSTALL_FILE = None
DEFAULT_PARAMS_FILE = None
PARAMS_FILE = None
README_FILE = None
PARAMS_BACKUP_DIR = None
RUN_PIPELINE = None
PREDICT_REG = None
PREDICT_CLASS = None


def configure_project_directory(base_directory=None):
    """
    Configure the active SOIL MLOPS project root.

    Every project-relative path used by the GUI is recalculated from this
    directory.  If *base_directory* is omitted, the current working
    directory is used.
    """

    global BASE_DIR
    global DEFAULT_DATASET
    global ALL_EVALUATIONS
    global CLASS_ALL_EVALUATIONS
    global REG_REPORT_DIR
    global CLASS_REPORT_DIR
    global DQ_REPORT_DIR
    global MULTICOLLINEARITY_REPORT_DIR
    global SUMMARY_REPORT_DIR
    global PROCESSED_DATA_DIR
    global REG_MODELS_DIR
    global CLASS_MODELS_DIR
    global INSTALL_FILE
    global DEFAULT_PARAMS_FILE
    global PARAMS_FILE
    global README_FILE
    global PARAMS_BACKUP_DIR
    global RUN_PIPELINE
    global PREDICT_REG
    global PREDICT_CLASS
    global SUMMARY_REPORT_SCRIPT

    if base_directory is None:
        selected = Path.cwd()
    else:
        selected = Path(base_directory).expanduser()

    BASE_DIR = selected.resolve()

    DEFAULT_DATASET = (
        BASE_DIR / "data" / "raw" / "samples.csv"
    )
    PROCESSED_DATA_DIR = BASE_DIR / DEFAULT_PATHS["processed_data"]
    REG_REPORT_DIR = BASE_DIR / DEFAULT_PATHS["reports_regression"]
    CLASS_REPORT_DIR = BASE_DIR / DEFAULT_PATHS["reports_classification"]
    REG_MODELS_DIR = BASE_DIR / DEFAULT_PATHS["models_regression"]
    CLASS_MODELS_DIR = BASE_DIR / DEFAULT_PATHS["models_classification"]
    DQ_REPORT_DIR = BASE_DIR / DEFAULT_PATHS["dq_report"]
    MULTICOLLINEARITY_REPORT_DIR = BASE_DIR / DEFAULT_PATHS["multicollinearity_report"]
    SUMMARY_REPORT_DIR = BASE_DIR / DEFAULT_PATHS["summary_report"]
    ALL_EVALUATIONS = REG_REPORT_DIR / "all_evaluations.csv"
    CLASS_ALL_EVALUATIONS = CLASS_REPORT_DIR / "all_evaluations_class.csv"
    INSTALL_FILE = BASE_DIR / "install.txt"
    DEFAULT_PARAMS_FILE = BASE_DIR / "params.yaml"
    README_FILE = BASE_DIR / "README.md"
    RUN_PIPELINE = BASE_DIR / "run_pipeline.py"
    PREDICT_REG = BASE_DIR / "src" / "predict_Reg.py"
    PREDICT_CLASS = BASE_DIR / "src" / "predict_class.py"
    SUMMARY_REPORT_SCRIPT = BASE_DIR / "src" / "generate_summary_report.py"

    # Re-resolve the configured params argument against the new project
    # directory. Absolute --params paths remain absolute.
    configure_params_file(PARAMS_ARGUMENT)

    return BASE_DIR


def configure_params_file(params_path=None):
    """
    Configure the YAML parameter file used by the GUI.

    Relative paths are resolved from the active MLOps project directory,
    so changing Current MLOps also changes a relative --params file.
    """

    global PARAMS_ARGUMENT
    global PARAMS_FILE
    global PARAMS_BACKUP_DIR

    if params_path is None:
        params_path = "params.yaml"

    PARAMS_ARGUMENT = str(params_path)
    selected = Path(params_path).expanduser()

    if not selected.is_absolute():
        selected = BASE_DIR / selected

    PARAMS_FILE = selected.resolve()
    PARAMS_BACKUP_DIR = PARAMS_FILE.parent / "params_backups"

    # Optional generic path configuration. Missing keys retain safe defaults.
    try:
        if PARAMS_FILE.exists():
            with PARAMS_FILE.open("r", encoding="utf-8") as handle:
                _cfg = yaml.safe_load(handle) or {}
            _data_cfg = _cfg.get("data", {}) or {}
            _paths_cfg = _cfg.get("paths", {}) or {}

            def _configured_path(value, fallback):
                candidate = Path(str(value or fallback)).expanduser()
                return candidate if candidate.is_absolute() else (BASE_DIR / candidate).resolve()

            global DEFAULT_DATASET, ALL_EVALUATIONS, CLASS_ALL_EVALUATIONS, REG_REPORT_DIR, CLASS_REPORT_DIR
            global DQ_REPORT_DIR, MULTICOLLINEARITY_REPORT_DIR, SUMMARY_REPORT_DIR, PROCESSED_DATA_DIR
            global REG_MODELS_DIR, CLASS_MODELS_DIR
            DEFAULT_DATASET = _configured_path(_data_cfg.get("input_csv"), "data/raw/samples.csv")
            PROCESSED_DATA_DIR = _configured_path(_paths_cfg.get("processed_data"), DEFAULT_PATHS["processed_data"])
            REG_REPORT_DIR = _configured_path(_paths_cfg.get("reports_regression"), DEFAULT_PATHS["reports_regression"])
            CLASS_REPORT_DIR = _configured_path(_paths_cfg.get("reports_classification"), DEFAULT_PATHS["reports_classification"])
            REG_MODELS_DIR = _configured_path(_paths_cfg.get("models_regression"), DEFAULT_PATHS["models_regression"])
            CLASS_MODELS_DIR = _configured_path(_paths_cfg.get("models_classification"), DEFAULT_PATHS["models_classification"])
            DQ_REPORT_DIR = _configured_path(_paths_cfg.get("dq_report"), DEFAULT_PATHS["dq_report"])
            MULTICOLLINEARITY_REPORT_DIR = _configured_path(_paths_cfg.get("multicollinearity_report"), DEFAULT_PATHS["multicollinearity_report"])
            SUMMARY_REPORT_DIR = _configured_path(_paths_cfg.get("summary_report"), DEFAULT_PATHS["summary_report"])
            ALL_EVALUATIONS = REG_REPORT_DIR / "all_evaluations.csv"
            CLASS_ALL_EVALUATIONS = CLASS_REPORT_DIR / "all_evaluations_class.csv"
    except Exception as error:
        print(f"[GUI][WARN] Could not apply YAML path configuration: {error}")

    return PARAMS_FILE


def parse_command_line(argv=None):
    """Parse command-line options for the GUI."""

    parser = argparse.ArgumentParser(
        description="Configurable GeoSoilMLOps MLOps graphical interface."
    )

    parser.add_argument(
        "--params",
        default="params.yaml",
        help=(
            "YAML parameter filename/path used by the GUI. Relative paths "
            "are resolved from the active Current MLOps project folder. "
            "Example: python soil_mlops_gui.py --params params.yaml"
        )
    )

    return parser.parse_args(argv)


# Initialize all paths from the directory used to launch the program.
configure_project_directory(BASE_DIR)

# ============================================================
# GENERAL HELPERS
# ============================================================

def natural_sort_key(value):
    """
    Natural sorting.

    Example:
        fs1, fs2, fs3, fs10

    rather than:
        fs1, fs10, fs2, fs3
    """

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(
            r"(\d+)",
            str(value)
        )
    ]


def load_yaml_file(path):
    """
    Load YAML and always return a dictionary.
    """

    path = Path(path)

    if not path.exists():
        return {}

    with path.open(
        "r",
        encoding="utf-8"
    ) as stream:

        loaded = yaml.safe_load(stream)

    if isinstance(loaded, dict):
        return loaded

    return {}


def make_params_backup(
    source_file,
    backup_directory
):
    """
    Make a timestamped backup of params.yaml before modification.

    Example output:
        params_backups/
            params_backup_20260807_125500.yaml
    """

    source_file = Path(source_file)
    backup_directory = Path(
        backup_directory
    )

    if not source_file.exists():
        return None

    backup_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    candidate = (
        backup_directory
        / f"params_backup_{timestamp}.yaml"
    )

    counter = 1

    while candidate.exists():

        candidate = (
            backup_directory
            / (
                f"params_backup_"
                f"{timestamp}_"
                f"{counter:02d}.yaml"
            )
        )

        counter += 1

    shutil.copy2(
        source_file,
        candidate
    )

    return candidate


def yaml_text(data):
    """
    Return YAML text suitable for preview.
    """

    return yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

class SoilMLOpsApp(tk.Tk):

    def __init__(self):

        super().__init__()

        self.title("Geo / Engineering MLOps")

        self._configure_main_window()

        self.current_mlops_var = tk.StringVar(
            value=str(BASE_DIR)
        )

        self.selected_csv = tk.StringVar(
            value=str(DEFAULT_DATASET)
        )

        self.params_profile_var = tk.StringVar(value=PARAMS_FILE.name if PARAMS_FILE else "params.yaml")
        self.processed_dir_var = tk.StringVar(value=str(PROCESSED_DATA_DIR))
        self.reg_reports_dir_var = tk.StringVar(value=str(REG_REPORT_DIR))
        self.class_reports_dir_var = tk.StringVar(value=str(CLASS_REPORT_DIR))
        self.reg_models_dir_var = tk.StringVar(value=str(REG_MODELS_DIR))
        self.class_models_dir_var = tk.StringVar(value=str(CLASS_MODELS_DIR))
        self.dq_report_dir_var = tk.StringVar(value=str(DQ_REPORT_DIR))
        self.multicollinearity_dir_var = tk.StringVar(value=str(MULTICOLLINEARITY_REPORT_DIR))
        self.summary_report_dir_var = tk.StringVar(value=str(SUMMARY_REPORT_DIR))

        # Per-run model-group selection. Defaults are loaded from the active
        # params.yaml, but the checkboxes can override them for one run.
        self.run_regression_var = tk.BooleanVar(value=True)
        self.run_classification_var = tk.BooleanVar(value=True)
        self._load_training_task_defaults()

        # Console/process state. Only one external program is allowed
        # to stream output at a time so messages never interleave.
        self._process_running = False
        self._run_counter = 0

        self._configure_styles()

        self._build_interface()
        self.refresh_params_profiles()

        # Previous-output cleanup is intentionally NOT shown at startup.
        # It is offered only after the user clicks Run Training Pipeline.

    def _load_training_task_defaults(self):
        """Load Regression/Classification checkbox defaults from active params.yaml."""
        try:
            cfg = load_project_params(PARAMS_FILE) if PARAMS_FILE and Path(PARAMS_FILE).exists() else {}
            pipeline_cfg = cfg.get("pipeline", {}) or {}
            class_cfg = cfg.get("classification", {}) or {}
            self.run_regression_var.set(bool(pipeline_cfg.get("regression", True)))
            self.run_classification_var.set(
                bool(pipeline_cfg.get("classification", True))
                and bool(class_cfg.get("enabled", False))
            )
        except Exception as exc:
            self.write_log(f"[GUI][WARN] Could not load training-task defaults: {exc}") if hasattr(self, "console") else None

    # ========================================================
    # SAFE STARTUP CLEANUP
    # ========================================================

    def _cleanup_protected_directory(self, folder: Path) -> bool:
        """Return True when *folder* must never be offered for deletion."""
        try:
            folder = Path(folder).resolve()
        except Exception:
            return True

        # Never allow deleting the project root itself or anything outside it.
        try:
            relative = folder.relative_to(BASE_DIR.resolve())
        except Exception:
            return True

        if str(relative) in {"", "."}:
            return True

        parts_lower = [part.lower() for part in relative.parts]

        # Protect real input/working data folders, but DO NOT protect
        # data_processing_report: it is generated output and should be
        # available in the pre-training cleanup list.
        protected_data_roots = {
            "data", "data_ac", "data_class", "datasets", "dataset"
        }
        if parts_lower and parts_lower[0] in protected_data_roots:
            return True

        # Source/configuration/environment/project-management folders are
        # never cleanup targets.
        protected_names = {
            "src", "configs", "config", ".git", ".github", ".venv", "venv",
            "env", "environment", "params_backups", "docs", "tests", "test",
            ".vscode", ".idea",
        }
        if any(part in protected_names for part in parts_lower):
            return True

        return False

    def _startup_cleanup_candidates(self):
        """
        Return every existing safe top-level project folder that the user may
        choose to remove before training.

        Critical input/source/config/environment folders are filtered by
        _cleanup_protected_directory(). Generated folders such as
        data_processing_report, regression_report, classification_report,
        model folders, summary_report, mlruns, and legacy output folders are
        therefore all visible.
        """
        candidates = []
        seen = set()

        try:
            children = sorted(
                (p for p in BASE_DIR.iterdir() if p.is_dir()),
                key=lambda p: p.name.lower(),
            )
        except Exception:
            children = []

        # Include all safe top-level folders.
        for folder in children:
            try:
                path = folder.resolve()
            except Exception:
                continue
            key = str(path).lower()
            if key in seen:
                continue
            if self._cleanup_protected_directory(path):
                continue
            seen.add(key)
            candidates.append(path)

        # Also make sure configured generated-output folders are represented
        # even if custom path names are used.
        configured = [
            REG_REPORT_DIR,
            CLASS_REPORT_DIR,
            REG_MODELS_DIR,
            CLASS_MODELS_DIR,
            DQ_REPORT_DIR.parent if DQ_REPORT_DIR is not None else None,
            SUMMARY_REPORT_DIR,
        ]
        for folder in configured:
            if folder is None:
                continue
            try:
                path = Path(folder).resolve()
            except Exception:
                continue
            key = str(path).lower()
            if (
                key not in seen
                and path.exists()
                and path.is_dir()
                and not self._cleanup_protected_directory(path)
            ):
                seen.add(key)
                candidates.append(path)

        return sorted(candidates, key=lambda value: str(value).lower())

    @staticmethod
    def _folder_item_count(folder: Path) -> int:
        """Count descendants for a useful but inexpensive cleanup preview."""
        try:
            return sum(1 for _ in folder.rglob("*"))
        except Exception:
            return -1

    def offer_startup_cleanup(self, continue_callback=None):
        """
        Offer cleanup immediately before training.

        This dialog is called only after Run Training Pipeline is clicked.
        All choices default to OFF. Critical data/source/config/environment
        directories are excluded and protected again at deletion time.
        """
        candidates = self._startup_cleanup_candidates()
        if not candidates:
            if continue_callback is not None:
                continue_callback()
            return

        win = tk.Toplevel(self)
        win.title("Clean Previous MLOps Outputs")
        win.transient(self)
        win.grab_set()
        win.geometry("760x520")
        win.minsize(620, 420)

        outer = ttk.Frame(win, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)

        ttk.Label(
            outer,
            text="Previous MLOps outputs were found",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(
            outer,
            text=(
                "Select any generated/project-output folders you want to delete before "
                "this training run. All safe top-level folders are shown and nothing is "
                "selected by default. Input data, source code, configs, parameter backups, "
                "and virtual environments are protected."
            ),
            wraplength=700,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(6, 10))

        toolbar = ttk.Frame(outer)
        toolbar.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        canvas = tk.Canvas(outer, highlightthickness=1, highlightbackground="#c8c8c8")
        scroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.grid(row=3, column=0, sticky="nsew")
        scroll.grid(row=3, column=1, sticky="ns")

        list_frame = ttk.Frame(canvas, padding=8)
        canvas_window = canvas.create_window((0, 0), window=list_frame, anchor="nw")

        def update_scroll(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def fit_width(event):
            canvas.itemconfigure(canvas_window, width=event.width)

        list_frame.bind("<Configure>", update_scroll)
        canvas.bind("<Configure>", fit_width)

        selections = []
        for row_index, folder in enumerate(candidates):
            variable = tk.BooleanVar(value=False)
            selections.append((folder, variable))
            try:
                rel = folder.relative_to(BASE_DIR)
                display = str(rel)
            except Exception:
                display = str(folder)
            count = self._folder_item_count(folder)
            suffix = f"  ({count} items)" if count >= 0 else ""
            ttk.Checkbutton(
                list_frame,
                text=display + suffix,
                variable=variable,
            ).grid(row=row_index, column=0, sticky="w", pady=3)

        def set_all(value):
            for _, variable in selections:
                variable.set(value)

        ttk.Button(toolbar, text="Select All Outputs", command=lambda: set_all(True)).pack(side="left")
        ttk.Button(toolbar, text="Clear Selection", command=lambda: set_all(False)).pack(side="left", padx=(6, 0))

        buttons = ttk.Frame(outer)
        buttons.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        buttons.columnconfigure(0, weight=1)

        def close_and_continue():
            try:
                win.grab_release()
            except Exception:
                pass
            win.destroy()
            if continue_callback is not None:
                self.after(50, continue_callback)

        def delete_selected():
            selected = [folder for folder, variable in selections if variable.get()]
            if not selected:
                messagebox.showinfo(
                    "Nothing Selected",
                    "No folders are selected. The project will be left unchanged.",
                    parent=win,
                )
                return

            # Re-validate safety at the destructive step itself.
            safe = [folder for folder in selected if not self._cleanup_protected_directory(folder)]
            if len(safe) != len(selected):
                messagebox.showerror(
                    "Protected Folder",
                    "One or more selected folders are protected and will not be deleted.",
                    parent=win,
                )
                return

            preview = "\n".join(f"• {folder}" for folder in safe)
            if not messagebox.askyesno(
                "Confirm Cleanup",
                "Permanently delete the selected previous-output folders?\n\n"
                + preview
                + "\n\nThis cannot be undone.",
                parent=win,
            ):
                return

            deleted = []
            errors = []
            for folder in safe:
                try:
                    shutil.rmtree(folder)
                    deleted.append(folder)
                except Exception as exc:
                    errors.append(f"{folder}: {exc}")

            for folder in deleted:
                self.write_log(f"[CLEANUP] Deleted previous output folder: {folder}")

            if errors:
                messagebox.showerror(
                    "Cleanup Partially Completed",
                    "Some folders could not be removed:\n\n" + "\n".join(errors),
                    parent=win,
                )
            else:
                messagebox.showinfo(
                    "Cleanup Completed",
                    f"Removed {len(deleted)} selected output folder(s).",
                    parent=win,
                )
                close_and_continue()

        ttk.Button(
            buttons,
            text="Keep Everything / Start Training",
            command=close_and_continue,
        ).grid(row=0, column=0, sticky="w")

        ttk.Button(
            buttons,
            text="Delete Selected / Start Training",
            command=delete_selected,
        ).grid(row=0, column=1, sticky="e", padx=(8, 0))

        win.protocol("WM_DELETE_WINDOW", win.destroy)

    # ========================================================
    # WINDOW SIZE
    # ========================================================

    def _configure_main_window(self):
        """
        Fit the main program window safely inside the screen.

        This intentionally avoids a large fixed minimum size.
        """

        self.update_idletasks()

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()

        # Use most of the available desktop while leaving a safe margin
        # for the taskbar and window borders.  This is intentionally based
        # on Tk logical pixels so it also behaves well with Windows scaling.
        usable_w = max(
            780,
            screen_w - 40
        )

        usable_h = max(
            560,
            screen_h - 70
        )

        width = min(
            1320,
            usable_w
        )

        height = min(
            900,
            usable_h
        )

        x = max(
            5,
            (screen_w - width) // 2
        )

        y = max(
            5,
            (screen_h - height) // 2
        )

        self.geometry(
            f"{width}x{height}+{x}+{y}"
        )

        # Deliberately modest minimum size so 1366x768 laptops
        # and scaled Windows desktops are still usable.
        self.minsize(
            760,
            520
        )

    # ========================================================
    # STYLE
    # ========================================================

    def _configure_styles(self):

        style = ttk.Style(self)

        # Compact vertical sizing on laptops / scaled Windows desktops.
        # winfo_screenheight() is reported in Tk logical pixels, which is
        # exactly what matters for whether widgets fit inside the window.
        compact_ui = self.winfo_screenheight() < 850
        title_size = 22 if compact_ui else 27
        section_size = 11 if compact_ui else 12
        large_button_padding = 4 if compact_ui else 7

        try:
            style.theme_use("vista")

        except tk.TclError:
            pass

        style.configure(
            "Title.TLabel",
            font=(
                "Segoe UI",
                title_size,
                "bold"
            ),
            anchor="center"
        )

        style.configure(
            "Section.TLabelframe.Label",
            font=(
                "Segoe UI",
                section_size,
                "bold"
            )
        )

        style.configure(
            "Large.TButton",
            font=(
                "Segoe UI",
                10
            ),
            padding=large_button_padding
        )

        style.configure(
            "Normal.TLabel",
            font=(
                "Segoe UI",
                10
            )
        )

        style.configure(
            "Small.TLabel",
            font=(
                "Segoe UI",
                9
            )
        )

        style.configure(
            "Toolbar.TButton",
            font=(
                "Segoe UI",
                9
            ),
            padding=5
        )

        style.configure(
            "Help.TButton",
            font=(
                "Segoe UI",
                9,
                "bold"
            ),
            padding=(
                4,
                2
            )
        )


    # ========================================================
    # README-BASED HELP SYSTEM
    # ========================================================

    def _read_help_topic(
        self,
        topic_key
    ):
        """
        Read a help block from README.md.

        Expected format:

            <!-- HELP:topic_key -->
            Help text...
            <!-- ENDHELP -->
        """

        if not README_FILE.exists():

            return (
                "README.md was not found.\n\n"
                "Expected location:\n"
                f"{README_FILE}"
            )

        content = README_FILE.read_text(
            encoding="utf-8",
            errors="replace"
        )

        # More robust pattern that handles variations in whitespace
        pattern = re.compile(
            rf"<!--\s*HELP\s*:\s*{re.escape(topic_key)}\s*-->(.*?)<!--\s*ENDHELP\s*-->",
            re.IGNORECASE | re.DOTALL
        )

        match = pattern.search(
            content
        )

        if not match:

            # Try to find any HELP section with this topic in a more lenient way
            # This handles cases where the format might have extra spaces or line breaks
            pattern_lenient = re.compile(
                rf"<!--\s*HELP\s*:\s*{re.escape(topic_key)}.*?-->(.*?)<!--\s*ENDHELP",
                re.IGNORECASE | re.DOTALL
            )
            
            match = pattern_lenient.search(content)

        if not match:

            return (
                "No help section was found "
                f"for '{topic_key}'.\n\n"
                "Add this block to README.md:\n\n"
                f"<!-- HELP:{topic_key} -->\n"
                "Help text here.\n"
                "<!-- ENDHELP -->"
            )

        return match.group(
            1
        ).strip()

    def show_help(
        self,
        topic_key,
        title="Help"
    ):
        """
        Display the requested README help topic.
        """

        help_text = self._read_help_topic(
            topic_key
        )

        win = tk.Toplevel(
            self
        )

        win.title(
            title
        )

        self._fit_child_to_screen(
            win,
            width_ratio=0.68,
            height_ratio=0.68
        )

        win.transient(
            self
        )

        frame = ttk.Frame(
            win,
            padding=10
        )

        frame.pack(
            fill="both",
            expand=True
        )

        frame.columnconfigure(
            0,
            weight=1
        )

        frame.rowconfigure(
            1,
            weight=1
        )

        ttk.Label(
            frame,
            text=title,
            font=(
                "Segoe UI",
                15,
                "bold"
            )
        ).grid(
            row=0,
            column=0,
            sticky="w",
            pady=(
                0,
                8
            )
        )

        text_box = tk.Text(
            frame,
            wrap="word",
            font=(
                "Segoe UI",
                10
            ),
            padx=8,
            pady=8
        )

        text_box.grid(
            row=1,
            column=0,
            sticky="nsew"
        )

        scrollbar = ttk.Scrollbar(
            frame,
            orient="vertical",
            command=(
                text_box.yview
            )
        )

        scrollbar.grid(
            row=1,
            column=1,
            sticky="ns"
        )

        text_box.configure(
            yscrollcommand=(
                scrollbar.set
            )
        )

        text_box.insert(
            "1.0",
            help_text
        )

        text_box.configure(
            state="disabled"
        )

        ttk.Button(
            frame,
            text="Close",
            command=(
                win.destroy
            )
        ).grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="e",
            pady=(
                8,
                0
            )
        )

    def _button_with_help(
        self,
        parent,
        *,
        text,
        command,
        help_key=None,
        help_title=None,
        row=None,
        column=None,
        columnspan=1,
        sticky="ew",
        padx=0,
        pady=0,
        width=None,
        **kwargs
    ):
        """Create a normal GUI button without a separate help button."""
        button_kwargs = dict(kwargs)
        if width is not None:
            button_kwargs["width"] = width

        button = ttk.Button(
            parent,
            text=text,
            command=command,
            **button_kwargs
        )

        if row is not None and column is not None:
            button.grid(
                row=row,
                column=column,
                columnspan=columnspan,
                sticky=sticky,
                padx=padx,
                pady=pady,
            )
        else:
            button.pack(
                fill="x",
                expand=True,
                padx=padx,
                pady=pady,
            )

        return button


    def _build_interface(self):

        main = ttk.Frame(
            self,
            padding=(
                10,
                6,
                10,
                6
            )
        )

        main.pack(
            fill="both",
            expand=True
        )

        main.columnconfigure(
            0,
            weight=1
        )

        # The output pane owns all spare vertical space and is never allowed
        # to collapse to an unusably small strip.
        main.rowconfigure(
            5,
            weight=1,
            minsize=105
        )

        header = ttk.Frame(main)
        header.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 5)
        )
        header.columnconfigure(1, weight=1)

        ttk.Label(
            header,
            text="GEO / ENGINEERING MLOPS",
            style="Title.TLabel"
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 18)
        )

        project_frame = ttk.Frame(header)
        project_frame.grid(
            row=0,
            column=1,
            sticky="ew"
        )
        project_frame.columnconfigure(1, weight=1)

        ttk.Label(
            project_frame,
            text="Current MLOps:",
            font=("Segoe UI", 11, "bold")
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 7)
        )

        ttk.Entry(
            project_frame,
            textvariable=self.current_mlops_var,
            state="readonly"
        ).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(0, 7)
        )

        ttk.Button(
            project_frame,
            text="Change...",
            command=self.select_mlops_directory
        ).grid(
            row=0,
            column=2,
            sticky="e"
        )

        ttk.Label(
            project_frame,
            text="Params profile:",
            font=("Segoe UI", 10, "bold")
        ).grid(row=1, column=0, sticky="w", padx=(0, 7), pady=(5, 0))

        self.params_profile_combo = ttk.Combobox(
            project_frame,
            textvariable=self.params_profile_var,
            state="readonly"
        )
        self.params_profile_combo.grid(row=1, column=1, sticky="ew", padx=(0, 7), pady=(5, 0))
        self.params_profile_combo.bind("<Button-1>", lambda _e: self.refresh_params_profiles())

        config_buttons = ttk.Frame(project_frame)
        config_buttons.grid(row=1, column=2, sticky="e", pady=(5, 0))
        ttk.Button(config_buttons, text="Select...", command=self.select_params_profile_file).pack(side="left")
        ttk.Button(config_buttons, text="Activate", command=self.activate_selected_params).pack(side="left", padx=(4, 0))
        ttk.Button(config_buttons, text="Paths...", command=self.configure_output_directories).pack(side="left", padx=(4, 0))

        ttk.Separator(
            main,
            orient="horizontal"
        ).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(
                0,
                8
            )
        )

        self._build_analysis_section(
            main
        )

        self._build_report_section(
            main
        )

        self._build_training_section(
            main
        )

        self._build_log_section(
            main
        )

    # ========================================================
    # FINAL ANALYSIS
    # ========================================================

    def _build_analysis_section(
        self,
        parent
    ):

        frame = ttk.LabelFrame(
            parent,
            text=(
                "Final Analysis "
                "and Prediction"
            ),
            style=(
                "Section."
                "TLabelframe"
            ),
            padding=6
        )

        frame.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(0, 7)
        )

        for column in range(3):
            frame.columnconfigure(column, weight=1)

        self._button_with_help(
            frame,
            text="Regression Report",
            command=self.run_regression_report,
            help_key="regression_report",
            help_title="Regression Report",
            row=0,
            column=0,
            padx=(0, 4)
        )

        self._button_with_help(
            frame,
            text="Classification Report",
            command=self.run_classification_report,
            help_key="classification_report",
            help_title="Classification Report",
            row=0,
            column=1,
            padx=(4, 4)
        )

        self._button_with_help(
            frame,
            text="Summary Report",
            command=self.show_summary_report_menu,
            help_key="summary_report",
            help_title="Summary Report",
            row=0,
            column=2,
            padx=(4, 0)
        )

    # ========================================================
    # REPORT EXECUTION
    # ========================================================

    def run_regression_report(self):

        if not self.check_required_file(
            ALL_EVALUATIONS
        ):
            return

        if not self.check_required_file(
            PREDICT_REG
        ):
            return

        command = [
            sys.executable,
            str(PREDICT_REG),
            "--gui",
            "--data",
            str(PROCESSED_DATA_DIR),
            "--models_dir",
            str(REG_MODELS_DIR),
            "--reports_dir",
            str(
                REG_REPORT_DIR
            ),
        ]

        self.run_command_async(
            command,
            (
                "Running regression "
                "prediction/report..."
            )
        )

    def run_classification_report(self):

        if not self.check_required_file(
            CLASS_ALL_EVALUATIONS
        ):
            return

        if not self.check_required_file(
            PREDICT_CLASS
        ):
            return

        command = [
            sys.executable,
            str(PREDICT_CLASS),
            "--gui",
            "--data",
            str(PROCESSED_DATA_DIR),
            "--models_dir",
            str(CLASS_MODELS_DIR),
            "--reports_dir",
            str(
                CLASS_REPORT_DIR
            ),
            "--params",
            str(PARAMS_FILE),
        ]

        self.run_command_async(
            command,
            (
                "Running classification "
                "prediction/report..."
            )
        )

    def show_summary_report_menu(self):
        """
        Show a small Summary Report action chooser.

        Processing:
            Regenerate the summary report, then open the output folder
            only after successful completion.

        Browsing Folder:
            Open the existing summary_report folder immediately.
        """

        win = tk.Toplevel(self)
        win.title("Summary Report")
        win.transient(self)
        win.grab_set()
        win.resizable(False, False)

        body = ttk.Frame(win, padding=14)
        body.pack(fill="both", expand=True)

        ttk.Label(
            body,
            text="Summary Report",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        ttk.Label(
            body,
            text="Choose an action:",
        ).pack(anchor="w", pady=(0, 10))

        def process_summary():
            try:
                win.grab_release()
            except Exception:
                pass
            win.destroy()
            self.run_summary_report()

        def browse_summary():
            try:
                win.grab_release()
            except Exception:
                pass
            win.destroy()
            self.open_folder(BASE_DIR / "summary_report")

        buttons = ttk.Frame(body)
        buttons.pack(fill="x")

        ttk.Button(
            buttons,
            text="Processing",
            command=process_summary,
            width=20,
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            buttons,
            text="Browsing Folder",
            command=browse_summary,
            width=20,
        ).pack(side="left")

        ttk.Button(
            body,
            text="Cancel",
            command=win.destroy,
        ).pack(anchor="e", pady=(12, 0))

        win.update_idletasks()

        # Center the compact popup over the main window.
        width = max(360, win.winfo_reqwidth())
        height = max(160, win.winfo_reqheight())

        x = self.winfo_rootx() + max(
            0,
            (self.winfo_width() - width) // 2
        )
        y = self.winfo_rooty() + max(
            0,
            (self.winfo_height() - height) // 2
        )

        win.geometry(
            f"{width}x{height}+{x}+{y}"
        )

        win.protocol(
            "WM_DELETE_WINDOW",
            win.destroy,
        )

    def run_summary_report(self):
        """
        Generate the consolidated summary report and, if generation succeeds,
        open the summary_report output folder automatically.
        """

        if not self.check_required_file(SUMMARY_REPORT_SCRIPT):
            return

        output_dir = Path(SUMMARY_REPORT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            str(SUMMARY_REPORT_SCRIPT),
            "--params",
            str(PARAMS_FILE),
            "--output_dir",
            str(output_dir),
        ]

        def browse_generated_summary():
            self.write_log(
                f"[SUMMARY] Opening generated summary folder: {output_dir}"
            )
            self.open_folder(output_dir)

        self.run_command_async(
            command,
            "Generating consolidated MLOps summary report...",
            on_success=browse_generated_summary,
        )

    # ========================================================
    # REPORT FOLDERS
    # ========================================================

    def _build_report_section(
        self,
        parent
    ):

        frame = ttk.LabelFrame(
            parent,
            text="Reports",
            style=(
                "Section."
                "TLabelframe"
            ),
            padding=6
        )

        frame.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(
                0,
                7
            )
        )

        frame.columnconfigure(
            0,
            weight=1
        )

        frame.columnconfigure(
            1,
            weight=1
        )

        self._button_with_help(
            frame,
            text=(
                "Browse Regression "
                "Reports"
            ),
            command=lambda: self.open_folder(
                REG_REPORT_DIR
            ),
            help_key="browse_regression_reports",
            help_title="Browse Regression Reports",
            row=0,
            column=0,
            padx=(
                0,
                4
            ),
            pady=(
                0,
                4
            )
        )

        self._button_with_help(
            frame,
            text=(
                "Browse Classification "
                "Reports"
            ),
            command=lambda: self.open_folder(
                CLASS_REPORT_DIR
            ),
            help_key="browse_classification_reports",
            help_title="Browse Classification Reports",
            row=0,
            column=1,
            padx=(
                4,
                0
            ),
            pady=(
                0,
                4
            )
        )

        self._button_with_help(
            frame,
            text=(
                "Data Qualification "
                "Results"
            ),
            command=lambda: self.open_folder(
                BASE_DIR / "data_processing_report" / "dq_report"
            ),
            help_key="data_qualification_results",
            help_title="Data Qualification Results",
            row=1,
            column=0,
            padx=(
                0,
                4
            ),
            pady=(
                4,
                0
            )
        )

        self._button_with_help(
            frame,
            text=(
                "Data Multicollinearity "
                "Results"
            ),
            command=lambda: self.open_folder(
                BASE_DIR / "data_processing_report" / "multicollinearity"
            ),
            help_key="data_multicollinearity_results",
            help_title="Data Multicollinearity Results",
            row=1,
            column=1,
            padx=(
                4,
                0
            ),
            pady=(
                4,
                0
            )
        )

    # ========================================================
    # NEW TRAINING
    # ========================================================

    def _build_training_section(
        self,
        parent
    ):

        frame = ttk.LabelFrame(
            parent,
            text="New Training",
            style=(
                "Section."
                "TLabelframe"
            ),
            padding=6
        )

        self.training_frame = frame

        frame.grid(
            row=4,
            column=0,
            sticky="ew",
            pady=(
                0,
                7
            )
        )

        frame.columnconfigure(
            1,
            weight=1
        )

        ttk.Label(
            frame,
            text="Dataset:",
            style="Normal.TLabel"
        ).grid(
            row=0,
            column=0,
            sticky="w"
        )

        self.dataset_entry = ttk.Entry(
            frame,
            textvariable=(
                self.selected_csv
            )
        )
        self.dataset_entry.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=6
        )

        default_holder = ttk.Frame(
            frame
        )

        default_holder.grid(
            row=0,
            column=2,
            padx=3
        )

        self.default_data_btn = ttk.Button(
            default_holder,
            text="Default Data",
            command=(
                self
                .use_default_dataset
            )
        )
        self.default_data_btn.pack(
            side="left"
        )

        ttk.Button(
            default_holder,
            text="?",
            width=3,
            style="Help.TButton",
            command=lambda: self.show_help(
                "default_data",
                "Default Data"
            )
        ).pack(
            side="left",
            padx=(
                4,
                0
            )
        )

        new_data_holder = ttk.Frame(
            frame
        )

        new_data_holder.grid(
            row=0,
            column=3,
            padx=3
        )

        self.new_dataset_btn = ttk.Button(
            new_data_holder,
            text="New Dataset...",
            command=(
                self
                .select_new_dataset
            )
        )
        self.new_dataset_btn.pack(
            side="left"
        )

        ttk.Button(
            new_data_holder,
            text="?",
            width=3,
            style="Help.TButton",
            command=lambda: self.show_help(
                "new_dataset",
                "New Dataset"
            )
        ).pack(
            side="left",
            padx=(
                4,
                0
            )
        )

        ttk.Label(
            frame,
            text="Feature Sets:",
            style="Normal.TLabel"
        ).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(
                7,
                0
            )
        )

        self.manage_feature_sets_holder = self._button_with_help(
            frame,
            text=(
                "Manage Target "
                "and Feature Sets"
            ),
            command=(
                self
                .open_feature_set_manager
            ),
            help_key="manage_feature_sets",
            help_title="Manage Target and Feature Sets",
            row=1,
            column=1,
            columnspan=3,
            padx=(
                6,
                0
            ),
            pady=(
                7,
                0
            )
        )

        ttk.Label(
            frame,
            text="Training:",
            style="Normal.TLabel"
        ).grid(
            row=2,
            column=0,
            sticky="w",
            pady=(
                7,
                0
            )
        )

        training_holder = ttk.Frame(
            frame
        )

        training_holder.grid(
            row=2,
            column=1,
            columnspan=3,
            sticky="ew",
            padx=(
                6,
                0
            ),
            pady=(
                7,
                0
            )
        )

        training_holder.columnconfigure(0, weight=1)

        task_holder = ttk.Frame(training_holder)
        task_holder.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Label(task_holder, text="Model groups:").pack(side="left", padx=(0, 8))
        ttk.Checkbutton(
            task_holder, text="Regression", variable=self.run_regression_var
        ).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(
            task_holder, text="Classification", variable=self.run_classification_var
        ).pack(side="left")

        self.run_training_btn = ttk.Button(
            training_holder,
            text="Run Training Pipeline",
            style="Large.TButton",
            command=self.run_training
        )
        self.run_training_btn.grid(row=1, column=0, sticky="ew")

        ttk.Button(
            training_holder,
            text="?",
            width=3,
            style="Help.TButton",
            command=lambda: self.show_help(
                "run_training_pipeline",
                "Run Training Pipeline"
            )
        ).grid(row=1, column=1, padx=(4, 0))

        self.progress = ttk.Progressbar(
            frame,
            mode="indeterminate"
        )

        self.progress.grid(
            row=3,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(
                7,
                0
            )
        )

        # Training controls are available only when the canonical
        # project dataset exists.  New Dataset remains available so
        # the user can populate data/raw/samples.csv.
        self._refresh_training_availability()

    def _set_widget_tree_state(self, widget, state):
        """Apply a ttk state recursively to buttons/entries below widget."""
        try:
            if isinstance(widget, (ttk.Button, ttk.Entry, ttk.Combobox)):
                widget.configure(state=state)
        except tk.TclError:
            pass

        for child in widget.winfo_children():
            self._set_widget_tree_state(child, state)

    def _refresh_training_availability(self):
        """Enable training only when data/raw/samples.csv exists."""
        dataset_exists = bool(
            DEFAULT_DATASET is not None
            and Path(DEFAULT_DATASET).is_file()
        )
        state = "normal" if dataset_exists else "disabled"

        # Keep the canonical path visible at all times.
        self.selected_csv.set(str(DEFAULT_DATASET))

        for widget_name in (
            "dataset_entry",
            "default_data_btn",
            "run_training_btn",
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                try:
                    widget.configure(state=state)
                except tk.TclError:
                    pass

        holder = getattr(self, "manage_feature_sets_holder", None)
        if holder is not None:
            self._set_widget_tree_state(holder, state)

        # This button must always stay usable; it is how a missing
        # canonical dataset is created/replaced.
        new_button = getattr(self, "new_dataset_btn", None)
        if new_button is not None:
            new_button.configure(state="normal")

        return dataset_exists

    # ========================================================
    # PROJECT DIRECTORY
    # ========================================================

    def select_mlops_directory(self):
        """Select the active MLOps project root folder."""

        selected = filedialog.askdirectory(
            title="Select MLOps Project Folder",
            initialdir=str(BASE_DIR),
            mustexist=True,
            parent=self
        )

        if not selected:
            return

        configure_project_directory(selected)
        self.current_mlops_var.set(str(BASE_DIR))
        self.selected_csv.set(str(DEFAULT_DATASET))
        self._sync_config_vars_from_active_params()
        self.refresh_params_profiles()
        self._refresh_training_availability()

        self.write_log(
            "\nCurrent MLOps project changed to:\n"
            f"{BASE_DIR}"
        )
        self.write_log(f"Parameters: {PARAMS_FILE}")
        self.write_log(f"Default dataset: {DEFAULT_DATASET}")

    # ========================================================
    # PARAMETER PROFILES AND OUTPUT PATHS
    # ========================================================

    def _sync_config_vars_from_active_params(self):
        """Refresh GUI directory variables after params/project changes."""
        self.params_profile_var.set("params.yaml")
        self.processed_dir_var.set(str(PROCESSED_DATA_DIR))
        self.reg_reports_dir_var.set(str(REG_REPORT_DIR))
        self.class_reports_dir_var.set(str(CLASS_REPORT_DIR))
        self.reg_models_dir_var.set(str(REG_MODELS_DIR))
        self.class_models_dir_var.set(str(CLASS_MODELS_DIR))
        self.dq_report_dir_var.set(str(DQ_REPORT_DIR))
        self.multicollinearity_dir_var.set(str(MULTICOLLINEARITY_REPORT_DIR))
        self.summary_report_dir_var.set(str(SUMMARY_REPORT_DIR))

    def refresh_params_profiles(self):
        """List YAML configuration profiles in the project root."""
        candidates = [p for p in BASE_DIR.glob("*.yaml") if p.is_file()]
        config_dir = BASE_DIR / "configs"
        if config_dir.is_dir():
            candidates.extend(p for p in config_dir.glob("*.yaml") if p.is_file())
        profiles = sorted(
            [p.relative_to(BASE_DIR).as_posix() for p in candidates],
            key=natural_sort_key,
        )
        if "params.yaml" not in profiles and (BASE_DIR / "params.yaml").exists():
            profiles.insert(0, "params.yaml")
        if hasattr(self, "params_profile_combo"):
            self.params_profile_combo["values"] = profiles
        if self.params_profile_var.get() not in profiles and profiles:
            self.params_profile_var.set("params.yaml" if "params.yaml" in profiles else profiles[0])
        return profiles

    def select_params_profile_file(self):
        """Select/import a YAML profile and make it available for activation."""
        selected = filedialog.askopenfilename(
            title="Select Parameter Profile",
            initialdir=str(BASE_DIR),
            filetypes=[("YAML files", "*.yaml"), ("YML files", "*.yml"), ("All files", "*.*")],
            parent=self,
        )
        if not selected:
            return
        source = Path(selected).expanduser().resolve()
        try:
            load_project_params(source)  # validation
            try:
                rel = source.relative_to(BASE_DIR).as_posix()
            except ValueError:
                config_dir = BASE_DIR / "configs"
                config_dir.mkdir(parents=True, exist_ok=True)
                destination = config_dir / source.name
                if destination.exists() and destination.resolve() != source:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    destination = config_dir / f"{source.stem}_{stamp}{source.suffix}"
                shutil.copy2(source, destination)
                rel = destination.relative_to(BASE_DIR).as_posix()
            self.refresh_params_profiles()
            self.params_profile_var.set(rel)
            self.write_log(f"Selected parameter profile: {rel}")
        except Exception as exc:
            messagebox.showerror("Parameter Error", str(exc), parent=self)

    def activate_selected_params(self):
        """Activate a selected YAML profile by copying it to params.yaml."""
        name = self.params_profile_var.get().strip()
        if not name:
            return
        source = (BASE_DIR / name).resolve()
        target = (BASE_DIR / "params.yaml").resolve()
        if not source.is_file():
            messagebox.showerror("Parameters Missing", f"Could not find:\n\n{source}", parent=self)
            return
        try:
            # Validate before replacing the active configuration.
            loaded = load_project_params(source)
            if not isinstance(loaded.get("TARGETS", {}), dict):
                raise ValueError("TARGETS must be a YAML mapping.")
            if target.exists() and source != target:
                make_params_backup(target, BASE_DIR / "params_backups")
                shutil.copy2(source, target)
            configure_params_file("params.yaml")
            self._sync_config_vars_from_active_params()
            self._load_training_task_defaults()
            write_install_txt(BASE_DIR, PARAMS_FILE)
            self.selected_csv.set(str(DEFAULT_DATASET))
            self._refresh_training_availability()
            self.params_profile_var.set("params.yaml")
            self.write_log(f"\nActivated parameter profile: {name}\nActive params: {PARAMS_FILE}")
            messagebox.showinfo(
                "Parameters Activated",
                f"{name} is now active as params.yaml.\n\nExisting programs can continue using --params params.yaml.",
                parent=self,
            )
        except Exception as exc:
            messagebox.showerror("Parameter Error", str(exc), parent=self)

    def configure_output_directories(self):
        """Select project output folders and persist them to params.yaml/install.txt."""
        win = tk.Toplevel(self)
        win.title("Configure Output Directories")
        win.transient(self)
        win.grab_set()
        win.columnconfigure(1, weight=1)

        fields = [
            ("Processed data", "processed_data", self.processed_dir_var),
            ("Regression reports", "reports_regression", self.reg_reports_dir_var),
            ("Classification reports", "reports_classification", self.class_reports_dir_var),
            ("Regression models", "models_regression", self.reg_models_dir_var),
            ("Classification models", "models_classification", self.class_models_dir_var),
            ("Data-quality reports", "dq_report", self.dq_report_dir_var),
            ("Multicollinearity reports", "multicollinearity_report", self.multicollinearity_dir_var),
            ("Research summary report", "summary_report", self.summary_report_dir_var),
        ]

        ttk.Label(
            win,
            text="Choose folders for the active params.yaml. Project-local folders are saved as relative paths.",
            wraplength=680,
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(10, 8))

        def browse(var):
            initial = var.get().strip() or str(BASE_DIR)
            chosen = filedialog.askdirectory(
                title="Select Output Folder",
                initialdir=initial if Path(initial).exists() else str(BASE_DIR),
                parent=win,
            )
            if chosen:
                var.set(chosen)

        for row, (label, _key, var) in enumerate(fields, start=1):
            ttk.Label(win, text=f"{label}:").grid(row=row, column=0, sticky="w", padx=(10, 6), pady=4)
            ttk.Entry(win, textvariable=var).grid(row=row, column=1, sticky="ew", pady=4)
            ttk.Button(win, text="Browse...", command=lambda v=var: browse(v)).grid(row=row, column=2, padx=8, pady=4)

        def reset_defaults():
            mapping = {
                "processed_data": self.processed_dir_var,
                "reports_regression": self.reg_reports_dir_var,
                "reports_classification": self.class_reports_dir_var,
                "models_regression": self.reg_models_dir_var,
                "models_classification": self.class_models_dir_var,
                "dq_report": self.dq_report_dir_var,
                "multicollinearity_report": self.multicollinearity_dir_var,
                "summary_report": self.summary_report_dir_var,
            }
            for key, var in mapping.items():
                var.set(str((BASE_DIR / DEFAULT_PATHS[key]).resolve()))

        def confirm():
            selected = {key: var.get().strip() for _label, key, var in fields}
            missing = [label for label, key, var in fields if not selected[key]]
            if missing:
                messagebox.showerror("Missing Directory", "Please specify every output directory.", parent=win)
                return
            try:
                make_params_backup(PARAMS_FILE, BASE_DIR / "params_backups")
                update_paths_in_params(BASE_DIR, PARAMS_FILE, selected)
                configure_params_file("params.yaml")
                for path in (PROCESSED_DATA_DIR, REG_REPORT_DIR, CLASS_REPORT_DIR, REG_MODELS_DIR, CLASS_MODELS_DIR, DQ_REPORT_DIR, MULTICOLLINEARITY_REPORT_DIR, SUMMARY_REPORT_DIR):
                    Path(path).mkdir(parents=True, exist_ok=True)
                install_path = write_install_txt(BASE_DIR, PARAMS_FILE)
                self._sync_config_vars_from_active_params()
                self.write_log("\nOutput directories updated in params.yaml")
                self.write_log(f"install.txt regenerated: {install_path}")
                win.destroy()
                messagebox.showinfo(
                    "Configuration Saved",
                    "Output paths were saved to params.yaml and install.txt was regenerated with the matching program arguments.",
                    parent=self,
                )
            except Exception as exc:
                messagebox.showerror("Configuration Error", str(exc), parent=win)

        buttons = ttk.Frame(win)
        buttons.grid(row=len(fields) + 1, column=0, columnspan=3, sticky="e", padx=10, pady=10)
        ttk.Button(buttons, text="Defaults", command=reset_defaults).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Cancel", command=win.destroy).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Confirm", command=confirm).pack(side="left")

    # ========================================================
    # DATASET HANDLING
    # ========================================================

    def use_default_dataset(self):

        if not DEFAULT_DATASET.exists():

            messagebox.showerror(
                "Default Data Not Found",
                (
                    "Could not find:\n\n"
                    f"{DEFAULT_DATASET}"
                )
            )

            return

        self.selected_csv.set(
            str(DEFAULT_DATASET)
        )

        self.write_log(
            (
                "Using default "
                "dataset:\n"
                f"{DEFAULT_DATASET}"
            )
        )

    def select_new_dataset(self):

        selected = filedialog.askopenfilename(
            title="Select Soil CSV Dataset",
            initialdir=str(BASE_DIR),
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
            parent=self,
        )

        if not selected:
            # Cancel changes nothing; availability still reflects the
            # canonical samples.csv file in the active project.
            self._refresh_training_availability()
            return

        source = Path(selected)

        if source.suffix.lower() != ".csv":
            messagebox.showwarning(
                "CSV Required",
                "Please select a .csv dataset file.",
                parent=self,
            )
            self._refresh_training_availability()
            return

        try:
            DEFAULT_DATASET.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            # The selected file becomes the canonical training dataset.
            # Replace an existing samples.csv without keeping alternate
            # dataset names inside the project workflow.
            if source.resolve() != DEFAULT_DATASET.resolve():
                shutil.copy2(
                    source,
                    DEFAULT_DATASET
                )

            self.selected_csv.set(
                str(DEFAULT_DATASET)
            )
            self._refresh_training_availability()

            self.write_log(
                "New dataset installed as canonical training data:\n"
                f"{DEFAULT_DATASET}"
            )

            messagebox.showinfo(
                "Dataset Loaded",
                (
                    "Dataset successfully installed as:\n\n"
                    f"{DEFAULT_DATASET}\n\n"
                    "New Training is now enabled."
                ),
                parent=self,
            )

        except Exception as exc:
            self._refresh_training_availability()
            messagebox.showerror(
                "Dataset Error",
                str(exc),
                parent=self,
            )

    # ========================================================
    # FEATURE SET MANAGER
    # ========================================================

    def open_feature_set_manager(self):

        dataset_file = Path(
            self.selected_csv.get()
        )

        if not dataset_file.exists():

            messagebox.showerror(
                "Dataset Missing",
                (
                    "Please select a valid "
                    "CSV dataset first."
                )
            )

            return

        try:
            dataframe = pd.read_csv(
                dataset_file,
                nrows=5
            )

            dataset_columns = (
                dataframe
                .columns
                .tolist()
            )

        except Exception as exc:

            messagebox.showerror(
                "CSV Error",
                (
                    "Could not read "
                    "CSV file:\n\n"
                    f"{exc}"
                )
            )

            return

        if not dataset_columns:

            messagebox.showerror(
                "CSV Error",
                (
                    "No columns were found "
                    "in the selected dataset."
                )
            )

            return

        full_params = load_yaml_file(
            PARAMS_FILE
        )

        yaml_targets = full_params.get(
            "TARGETS",
            {}
        )

        if not isinstance(
            yaml_targets,
            dict
        ):
            yaml_targets = {}

        # Work on a temporary in-memory copy.
        # params.yaml is not changed until the
        # final Save button is pressed.
        working_targets = deepcopy(
            yaml_targets
        )

        classification_cfg = full_params.get(
            "classification",
            {}
        )
        if not isinstance(classification_cfg, dict):
            classification_cfg = {}

        yaml_class_boundaries = classification_cfg.get(
            "class_boundaries",
            {}
        )
        if not isinstance(yaml_class_boundaries, dict):
            yaml_class_boundaries = {}

        working_class_boundaries = deepcopy(
            yaml_class_boundaries
        )

        yaml_target_modes = classification_cfg.get("target_modes", {}) or {}
        if not isinstance(yaml_target_modes, dict):
            yaml_target_modes = {}
        working_target_modes = deepcopy(yaml_target_modes)

        # Train/test split is stored in params.yaml as ratios
        # (for example 0.80 / 0.20) but displayed in this GUI as
        # percentages (80 / 20).
        yaml_split = full_params.get(
            "split",
            {}
        )
        if not isinstance(yaml_split, dict):
            yaml_split = {}

        working_split = deepcopy(
            yaml_split
        )

        # ----------------------------------------------------
        # MODEL / HYPERPARAMETER SETTINGS
        # ----------------------------------------------------
        # Regression models and grids are top-level params.yaml sections:
        #   models:
        #   param_grids:
        #
        # Classification equivalents are:
        #   classification:
        #     models:
        #     param_grids:
        yaml_regression_models = full_params.get("models", [])
        if not isinstance(yaml_regression_models, list):
            yaml_regression_models = []

        yaml_regression_grids = full_params.get("param_grids", {})
        if not isinstance(yaml_regression_grids, dict):
            yaml_regression_grids = {}

        yaml_classification_models = classification_cfg.get("models", [])
        if not isinstance(yaml_classification_models, list):
            yaml_classification_models = []

        yaml_classification_grids = classification_cfg.get("param_grids", {})
        if not isinstance(yaml_classification_grids, dict):
            yaml_classification_grids = {}

        # Built-in model-search presets are deliberately stored separately from
        # the active params.yaml.  This keeps new experiment profiles compact
        # while still giving users a scientifically reasonable starting search
        # space that can be edited freely in this manager.  Values explicitly
        # present in params.yaml ALWAYS override these defaults.
        preset_file = BASE_DIR / "configs" / "hyperparameter_presets.yaml"
        preset_cfg = {}
        if preset_file.exists():
            try:
                with preset_file.open("r", encoding="utf-8") as handle:
                    preset_cfg = yaml.safe_load(handle) or {}
                if not isinstance(preset_cfg, dict):
                    preset_cfg = {}
            except Exception:
                preset_cfg = {}

        preset_regression = preset_cfg.get("regression", {}) or {}
        if not isinstance(preset_regression, dict):
            preset_regression = {}
        preset_classification = preset_cfg.get("classification", {}) or {}
        if not isinstance(preset_classification, dict):
            preset_classification = {}

        preset_regression_models = preset_regression.get("models", []) or []
        if not isinstance(preset_regression_models, list):
            preset_regression_models = []
        preset_regression_grids = preset_regression.get("param_grids", {}) or {}
        if not isinstance(preset_regression_grids, dict):
            preset_regression_grids = {}

        preset_classification_models = preset_classification.get("models", []) or []
        if not isinstance(preset_classification_models, list):
            preset_classification_models = []
        preset_classification_grids = preset_classification.get("param_grids", {}) or {}
        if not isinstance(preset_classification_grids, dict):
            preset_classification_grids = {}

        def _unique_model_names(*groups):
            names = []
            for group in groups:
                for name in group:
                    name = str(name).strip()
                    if name and name not in names:
                        names.append(name)
            return names

        # Start from presets, then overlay the active profile.  This means a
        # fresh/minimal params.yaml automatically exposes predefined search
        # spaces, while project-specific edits are never overwritten.
        working_regression_models = _unique_model_names(
            yaml_regression_models,
            preset_regression_models
        )
        working_regression_grids = deepcopy(preset_regression_grids)
        working_regression_grids.update(deepcopy(yaml_regression_grids))

        working_classification_models = _unique_model_names(
            yaml_classification_models,
            preset_classification_models
        )
        working_classification_grids = deepcopy(preset_classification_grids)
        working_classification_grids.update(deepcopy(yaml_classification_grids))

        # ----------------------------------------------------
        # DIALOG
        # ----------------------------------------------------

        win = tk.Toplevel(self)

        win.title(
            "Target and Feature Set Manager"
        )

        self._fit_child_to_screen(
            win,
            width_ratio=0.97,
            height_ratio=0.88
        )

        win.transient(self)
        win.grab_set()

        container = ttk.Frame(
            win,
            padding=(
                8,
                7,
                8,
                7
            )
        )

        container.pack(
            fill="both",
            expand=True
        )

        container.columnconfigure(
            0,
            weight=1
        )

        # The PanedWindow gets all extra vertical space.
        container.rowconfigure(
            2,
            weight=1
        )

        # ====================================================
        # TARGET BAR
        # ====================================================

        topbar = ttk.Frame(
            container
        )

        topbar.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(
                0,
                6
            )
        )

        # Layout inspired by the requested manager design:
        # Target | Split % (Train/Test) | Class boundaries.
        topbar.columnconfigure(1, weight=2)
        topbar.columnconfigure(2, weight=1)
        topbar.columnconfigure(3, weight=1)
        topbar.columnconfigure(4, weight=2)
        topbar.columnconfigure(5, weight=1)

        ttk.Label(
            topbar,
            text="Target:",
            font=(
                "Segoe UI",
                10,
                "bold"
            )
        ).grid(
            row=0,
            column=0,
            rowspan=2,
            sticky="w",
            padx=(
                0,
                6
            )
        )

        target_var = tk.StringVar()

        target_combo = ttk.Combobox(
            topbar,
            textvariable=target_var,
            state="readonly",
            width=18
        )

        target_combo.grid(
            row=0,
            column=1,
            rowspan=2,
            sticky="ew",
            padx=(0, 18)
        )

        # Split values are displayed in the right configuration pane.
        train_split_var = tk.StringVar()
        test_split_var = tk.StringVar()
        grouping_enabled_var = tk.BooleanVar()
        group_column_var = tk.StringVar()

        # Classification-boundary controls are displayed in the
        # right configuration pane.
        class_mode_var = tk.StringVar(value="auto")
        lower_boundary_var = tk.StringVar()
        upper_boundary_var = tk.StringVar()


        # ====================================================
        # TOOLBAR
        # ====================================================

        toolbar = ttk.Frame(
            container
        )

        toolbar.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(
                0,
                6
            )
        )

        # ====================================================
        # MAIN HORIZONTAL PANED WINDOW
        # ====================================================

        panes = ttk.Panedwindow(
            container,
            orient="horizontal"
        )

        panes.grid(
            row=2,
            column=0,
            sticky="nsew"
        )

        # ----------------------------------------------------
        # LEFT PANE: AVAILABLE COLUMNS
        # ----------------------------------------------------

        left_panel = ttk.Frame(
            panes,
            padding=(
                5,
                5,
                5,
                5
            )
        )

        left_panel.columnconfigure(
            0,
            weight=1
        )

        left_panel.rowconfigure(
            1,
            weight=1
        )

        ttk.Label(
            left_panel,
            text="Available Columns",
            font=(
                "Segoe UI",
                10,
                "bold"
            )
        ).grid(
            row=0,
            column=0,
            sticky="w",
            pady=(
                0,
                4
            )
        )

        available_holder = ttk.Frame(
            left_panel
        )

        available_holder.grid(
            row=1,
            column=0,
            sticky="nsew"
        )

        available_holder.columnconfigure(
            0,
            weight=1
        )

        available_holder.rowconfigure(
            0,
            weight=1
        )

        available_list = tk.Listbox(
            available_holder,
            font=(
                "Segoe UI",
                10
            ),
            selectmode=tk.EXTENDED,
            exportselection=False,
            borderwidth=1,
            relief="solid"
        )

        available_list.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        available_scroll = ttk.Scrollbar(
            available_holder,
            orient="vertical",
            command=(
                available_list.yview
            )
        )

        available_scroll.grid(
            row=0,
            column=1,
            sticky="ns"
        )

        available_list.configure(
            yscrollcommand=(
                available_scroll.set
            )
        )

        # ----------------------------------------------------
        # CENTER PANE: FEATURE SET EDITOR
        # ----------------------------------------------------

        center_panel = ttk.Frame(
            panes,
            padding=(
                5,
                5,
                5,
                5
            )
        )

        center_panel.columnconfigure(
            0,
            weight=1
        )

        center_panel.rowconfigure(
            2,
            weight=1
        )

        set_title_var = tk.StringVar(
            value="Feature Set Inputs"
        )

        ttk.Label(
            center_panel,
            textvariable=set_title_var,
            font=(
                "Segoe UI",
                10,
                "bold"
            )
        ).grid(
            row=0,
            column=0,
            sticky="w",
            pady=(
                0,
                4
            )
        )

        arrow_bar = ttk.Frame(
            center_panel
        )

        arrow_bar.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(
                0,
                4
            )
        )

        arrow_bar.columnconfigure(
            0,
            weight=1
        )

        arrow_bar.columnconfigure(
            3,
            weight=1
        )

        feature_holder = ttk.Frame(
            center_panel
        )

        feature_holder.grid(
            row=2,
            column=0,
            sticky="nsew"
        )

        feature_holder.columnconfigure(
            0,
            weight=1
        )

        feature_holder.rowconfigure(
            0,
            weight=1
        )

        feature_list = tk.Listbox(
            feature_holder,
            font=(
                "Segoe UI",
                10
            ),
            selectmode=tk.EXTENDED,
            exportselection=False,
            borderwidth=1,
            relief="solid"
        )

        feature_list.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        feature_scroll = ttk.Scrollbar(
            feature_holder,
            orient="vertical",
            command=(
                feature_list.yview
            )
        )

        feature_scroll.grid(
            row=0,
            column=1,
            sticky="ns"
        )

        feature_list.configure(
            yscrollcommand=(
                feature_scroll.set
            )
        )

        # ----------------------------------------------------
        # RIGHT PANE: EXISTING FEATURE SETS
        # ----------------------------------------------------

        right_panel = ttk.Frame(
            panes,
            padding=(
                5,
                5,
                5,
                5
            )
        )

        right_panel.columnconfigure(
            0,
            weight=1
        )

        right_panel.rowconfigure(
            1,
            weight=1
        )

        ttk.Label(
            right_panel,
            text="Current Feature Sets",
            font=(
                "Segoe UI",
                10,
                "bold"
            )
        ).grid(
            row=0,
            column=0,
            sticky="w",
            pady=(
                0,
                4
            )
        )

        sets_holder = ttk.Frame(
            right_panel
        )

        sets_holder.grid(
            row=1,
            column=0,
            sticky="nsew"
        )

        sets_holder.columnconfigure(
            0,
            weight=1
        )

        sets_holder.rowconfigure(
            0,
            weight=1
        )

        feature_set_list = tk.Listbox(
            sets_holder,
            font=(
                "Segoe UI",
                10
            ),
            selectmode=tk.SINGLE,
            exportselection=False,
            borderwidth=1,
            relief="solid"
        )

        feature_set_list.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        sets_scroll = ttk.Scrollbar(
            sets_holder,
            orient="vertical",
            command=(
                feature_set_list.yview
            )
        )

        sets_scroll.grid(
            row=0,
            column=1,
            sticky="ns"
        )

        feature_set_list.configure(
            yscrollcommand=(
                sets_scroll.set
            )
        )

        # ----------------------------------------------------
        # FAR-RIGHT PANE: TRAINING / SEARCH-SPACE CONFIGURATION
        # ----------------------------------------------------

        model_panel = ttk.Frame(
            panes,
            padding=(10, 5, 6, 5)
        )
        model_panel.columnconfigure(0, weight=1)
        model_panel.rowconfigure(9, weight=1)

        # -----------------------------
        # Train/test split
        # -----------------------------
        split_config = ttk.Frame(model_panel)
        split_config.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 8)
        )
        split_config.columnconfigure(1, weight=1)
        split_config.columnconfigure(3, weight=1)

        ttk.Label(
            split_config,
            text="Splits %",
            font=("Segoe UI", 10, "bold")
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 12)
        )

        ttk.Label(
            split_config,
            text="Train:",
            font=("Segoe UI", 10)
        ).grid(
            row=0,
            column=1,
            sticky="e",
            padx=(0, 5)
        )

        train_split_entry = ttk.Entry(
            split_config,
            textvariable=train_split_var,
            width=8
        )
        train_split_entry.grid(
            row=0,
            column=2,
            sticky="w",
            padx=(0, 12)
        )

        ttk.Label(
            split_config,
            text="Test:",
            font=("Segoe UI", 10)
        ).grid(
            row=1,
            column=1,
            sticky="e",
            padx=(0, 5),
            pady=(6, 0)
        )

        test_split_entry = ttk.Entry(
            split_config,
            textvariable=test_split_var,
            width=8
        )
        test_split_entry.grid(
            row=1,
            column=2,
            sticky="w",
            pady=(6, 0)
        )

        grouping_check = ttk.Checkbutton(
            split_config,
            text="Group-aware split/CV",
            variable=grouping_enabled_var
        )
        grouping_check.grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(7, 0)
        )

        ttk.Label(split_config, text="Group column:").grid(
            row=2, column=2, sticky="e", padx=(6, 4), pady=(7, 0)
        )
        group_column_combo = ttk.Combobox(
            split_config, textvariable=group_column_var, width=16
        )
        group_column_combo["values"] = tuple(dataset_columns)
        group_column_combo.grid(
            row=2, column=3, sticky="ew", pady=(7, 0)
        )

        # -----------------------------
        # Classification boundaries
        # -----------------------------
        boundary_config = ttk.Frame(model_panel)
        boundary_config.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(0, 8)
        )
        boundary_config.columnconfigure(0, weight=1)

        ttk.Label(
            boundary_config,
            text="Classification target mode:",
            font=("Segoe UI", 9, "bold")
        ).grid(row=0, column=0, sticky="e", padx=(0, 5))

        class_mode_combo = ttk.Combobox(
            boundary_config, textvariable=class_mode_var, width=18, state="readonly",
            values=("auto", "threshold", "categorical")
        )
        class_mode_combo.grid(row=0, column=1, sticky="w")

        ttk.Label(
            boundary_config,
            text="Lower class boundary:",
            font=("Segoe UI", 9, "bold")
        ).grid(
            row=2,
            column=0,
            sticky="e",
            padx=(0, 5),
            pady=(5, 0)
        )

        lower_boundary_entry = ttk.Entry(
            boundary_config,
            textvariable=lower_boundary_var,
            width=10
        )
        lower_boundary_entry.grid(
            row=1,
            column=1,
            sticky="w"
        )

        ttk.Label(
            boundary_config,
            text="Upper class boundary:",
            font=("Segoe UI", 9, "bold")
        ).grid(
            row=1,
            column=0,
            sticky="e",
            padx=(0, 5),
            pady=(5, 0)
        )

        upper_boundary_entry = ttk.Entry(
            boundary_config,
            textvariable=upper_boundary_var,
            width=10
        )
        upper_boundary_entry.grid(
            row=2,
            column=1,
            sticky="w",
            pady=(5, 0)
        )

        ttk.Separator(
            model_panel,
            orient="horizontal"
        ).grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(0, 8)
        )

        # -----------------------------
        # Nested-CV search-space editor
        # -----------------------------
        ttk.Label(
            model_panel,
            text="Model Search Configuration",
            font=("Segoe UI", 11, "bold")
        ).grid(
            row=3,
            column=0,
            sticky="w",
            pady=(0, 5)
        )

        ttk.Label(
            model_panel,
            text="Model:",
            font=("Segoe UI", 10, "bold")
        ).grid(
            row=4,
            column=0,
            sticky="w",
            pady=(0, 3)
        )

        model_var = tk.StringVar()

        model_select_holder = ttk.Frame(
            model_panel
        )
        model_select_holder.grid(
            row=5,
            column=0,
            sticky="ew",
            pady=(0, 8)
        )
        model_select_holder.columnconfigure(
            0,
            weight=1
        )

        model_combo = ttk.Combobox(
            model_select_holder,
            textvariable=model_var,
            state="readonly",
            postcommand=lambda: refresh_model_combo(
                select_label=model_var.get().strip()
            )
        )
        model_combo.grid(
            row=0,
            column=0,
            sticky="ew"
        )

        refresh_model_btn = ttk.Button(
            model_select_holder,
            text="Refresh YAML",
            style="Toolbar.TButton"
        )
        refresh_model_btn.grid(
            row=0,
            column=1,
            padx=(6, 0)
        )

        restore_preset_btn = ttk.Button(
            model_select_holder,
            text="Load Preset",
            style="Toolbar.TButton"
        )
        restore_preset_btn.grid(
            row=0,
            column=2,
            padx=(6, 0)
        )

        restore_all_presets_btn = ttk.Button(
            model_select_holder,
            text="Load All Presets",
            style="Toolbar.TButton"
        )
        restore_all_presets_btn.grid(
            row=0,
            column=3,
            padx=(6, 0)
        )

        ttk.Separator(
            model_panel,
            orient="horizontal"
        ).grid(
            row=6,
            column=0,
            sticky="ew",
            pady=(0, 6)
        )

        ttk.Label(
            model_panel,
            text="Hyperparameter Search Space:",
            font=("Segoe UI", 10, "bold")
        ).grid(
            row=7,
            column=0,
            sticky="w",
            pady=(0, 4)
        )

        hyper_holder = ttk.Frame(model_panel)
        hyper_holder.grid(
            row=9,
            column=0,
            sticky="nsew"
        )
        hyper_holder.columnconfigure(0, weight=1)
        hyper_holder.rowconfigure(0, weight=1)

        hyper_text = tk.Text(
            hyper_holder,
            wrap="none",
            font=("Consolas", 9),
            borderwidth=1,
            relief="solid",
            undo=True
        )
        hyper_text.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        hyper_y_scroll = ttk.Scrollbar(
            hyper_holder,
            orient="vertical",
            command=hyper_text.yview
        )
        hyper_y_scroll.grid(
            row=0,
            column=1,
            sticky="ns"
        )

        hyper_x_scroll = ttk.Scrollbar(
            hyper_holder,
            orient="horizontal",
            command=hyper_text.xview
        )
        hyper_x_scroll.grid(
            row=1,
            column=0,
            sticky="ew"
        )

        hyper_text.configure(
            yscrollcommand=hyper_y_scroll.set,
            xscrollcommand=hyper_x_scroll.set
        )

        hyper_hint_var = tk.StringVar(
            value=(
                "Candidate values searched automatically by inner CV. "
                "Nested CV selects the best combination."
            )
        )

        ttk.Label(
            model_panel,
            textvariable=hyper_hint_var,
            style="Small.TLabel",
            wraplength=390
        ).grid(
            row=10,
            column=0,
            sticky="ew",
            pady=(6, 0)
        )

        # Add panes only after all widgets exist.
        panes.add(
            left_panel,
            weight=2
        )

        panes.add(
            center_panel,
            weight=3
        )

        panes.add(
            right_panel,
            weight=3
        )

        # Give the training/search-space pane substantially more horizontal
        # room so its controls and explanatory text fit without being
        # squeezed against the right edge.
        panes.add(
            model_panel,
            weight=5
        )

        # ====================================================
        # STATUS BAR
        # ====================================================

        status_var = tk.StringVar(
            value=(
                "Select a target and feature set. "
                "Train/Test are loaded from params.yaml and must total 100%. "
                "Leave both class-boundary fields blank for automatic limits."
            )
        )

        status = ttk.Label(
            container,
            textvariable=status_var,
            style="Small.TLabel",
            anchor="w"
        )

        status.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(
                6,
                0
            )
        )

        # ====================================================
        # BOTTOM ACTION BAR
        # ====================================================

        bottom = ttk.Frame(
            container
        )

        bottom.grid(
            row=4,
            column=0,
            sticky="ew",
            pady=(
                6,
                0
            )
        )

        bottom.columnconfigure(
            0,
            weight=1
        )

        ttk.Label(
            bottom,
            text=(
                "A timestamped backup "
                "is created before saving."
            ),
            style="Small.TLabel"
        ).grid(
            row=0,
            column=0,
            sticky="w"
        )

        # ====================================================
        # STATE
        # ====================================================

        current_set_var = (
            tk.StringVar()
        )

        # Track the target that actually owns the feature set currently
        # loaded in the editor.  The target combobox changes before its
        # selection event fires, so get_target() alone is not safe when
        # committing the previous editor contents.
        current_target_var = (
            tk.StringVar()
        )

        boundary_target_var = (
            tk.StringVar()
        )

        # Stores the exact model key currently loaded in the
        # hyperparameter editor. Format: ("regression"|"classification", name)
        current_model_key = {
            "scope": "",
            "name": ""
        }

        # This flag prevents selection events from recursively
        # committing/loading while a list is being refreshed.
        state = {
            "loading": False
        }

        # ====================================================
        # LOCAL HELPER FUNCTIONS
        # ====================================================

        def _format_split_percent(value, default):
            """Convert a YAML split ratio to a clean percentage string."""
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = float(default)

            percent = number * 100.0
            if abs(percent - round(percent)) < 1e-10:
                return str(int(round(percent)))
            return f"{percent:.6g}"

        def load_split_settings():
            """Load split.train/test from params.yaml into percentage fields."""
            train_split_var.set(
                _format_split_percent(
                    working_split.get("train", 0.80),
                    0.80
                )
            )
            test_split_var.set(
                _format_split_percent(
                    working_split.get("test", 0.20),
                    0.20
                )
            )
            grouping_enabled_var.set(bool(
                working_split.get(
                    "grouping_enabled",
                    working_split.get("group_column") not in (None, "", False)
                )
            ))
            group_column_var.set(str(working_split.get("group_column") or ""))

        def commit_split_settings(*, show_errors=True):
            """
            Validate GUI percentages and store them as YAML ratios.

            Example:
                GUI:  Train=80, Test=20
                YAML: train: 0.8, test: 0.2
            """
            train_text = train_split_var.get().strip()
            test_text = test_split_var.get().strip()

            try:
                train_percent = float(train_text)
                test_percent = float(test_text)
            except (TypeError, ValueError):
                if show_errors:
                    messagebox.showwarning(
                        "Train/Test Split",
                        (
                            "Train and Test split values must be numeric "
                            "percentages, for example 80 and 20."
                        ),
                        parent=win
                    )
                return False

            if train_percent <= 0 or test_percent <= 0:
                if show_errors:
                    messagebox.showwarning(
                        "Train/Test Split",
                        "Train and Test percentages must both be greater than 0.",
                        parent=win
                    )
                return False

            total = train_percent + test_percent
            if abs(total - 100.0) > 1e-8:
                if show_errors:
                    messagebox.showwarning(
                        "Train/Test Split",
                        (
                            "Train and Test percentages must sum to 100%.\n\n"
                            f"Current total: {total:g}%"
                        ),
                        parent=win
                    )
                return False

            working_split["train"] = train_percent / 100.0
            working_split["test"] = test_percent / 100.0
            working_split.setdefault("mode", "fixed")
            grouping_enabled = bool(grouping_enabled_var.get())
            group_column = group_column_var.get().strip()
            if grouping_enabled and not group_column:
                if show_errors:
                    messagebox.showwarning(
                        "Group-aware Split",
                        "Select a group column or disable group-aware splitting.",
                        parent=win
                    )
                return False
            working_split["grouping_enabled"] = grouping_enabled
            working_split["group_column"] = group_column if grouping_enabled else None
            return True

        def model_display_items():
            """
            Return display labels mapped to their real YAML model location.

            Prefixes avoid ambiguity when a model name such as RF or KNN
            exists in both regression and classification sections.
            """
            items = []
            mapping = {}

            regression_names = []
            for name in list(working_regression_models) + list(working_regression_grids.keys()):
                name = str(name).strip()
                if name and name not in regression_names:
                    regression_names.append(name)

            classification_names = []
            for name in list(working_classification_models) + list(working_classification_grids.keys()):
                name = str(name).strip()
                if name and name not in classification_names:
                    classification_names.append(name)

            for name in regression_names:
                label = f"Regression: {name}"
                items.append(label)
                mapping[label] = ("regression", name)

            for name in classification_names:
                label = f"Classification: {name}"
                items.append(label)
                mapping[label] = ("classification", name)

            return items, mapping

        def get_model_grid(scope, name):
            if scope == "regression":
                grid = working_regression_grids.get(name, {})
            elif scope == "classification":
                grid = working_classification_grids.get(name, {})
            else:
                grid = {}

            if grid is None:
                grid = {}
            if not isinstance(grid, dict):
                grid = {}
            return grid

        def set_model_grid(scope, name, grid):
            if scope == "regression":
                working_regression_grids[name] = deepcopy(grid)
            elif scope == "classification":
                working_classification_grids[name] = deepcopy(grid)

        def hyperparameter_yaml(grid):
            """Serialize one model grid without an extra model-name wrapper."""
            if not isinstance(grid, dict) or not grid:
                return "{}\n"
            return yaml.safe_dump(
                grid,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False
            )

        def refresh_model_combo(select_label=None):
            items, mapping = model_display_items()
            model_combo["values"] = items

            if not items:
                model_var.set("")
                hyper_text.delete("1.0", tk.END)
                hyper_text.insert("1.0", "{}\n")
                hyper_text.configure(state="disabled")
                hyper_hint_var.set(
                    "No model search spaces were found in params.yaml."
                )
                current_model_key["scope"] = ""
                current_model_key["name"] = ""
                return mapping

            hyper_text.configure(state="normal")

            if select_label in items:
                model_var.set(select_label)
            elif model_var.get() not in items:
                model_var.set(items[0])

            return mapping

        def load_model_hyperparameters(label=None):
            if label is None:
                label = model_var.get().strip()

            _, mapping = model_display_items()
            scope_name = mapping.get(label)

            # Resolve first.  If the combobox is temporarily blank/unresolved,
            # keep the existing hyperparameter text exactly as it is.
            if not scope_name:
                return

            state["loading"] = True
            try:
                hyper_text.configure(state="normal")
                hyper_text.delete("1.0", tk.END)

                scope, name = scope_name
                grid = get_model_grid(scope, name)
                hyper_text.insert(
                    "1.0",
                    hyperparameter_yaml(grid)
                )

                current_model_key["scope"] = scope
                current_model_key["name"] = name
                hyper_hint_var.set(
                    f"Editing the {scope} hyperparameter search space for {name}. "
                    "These are candidate values; inner CV automatically selects "
                    "the best combination. Use valid YAML."
                )
            finally:
                state["loading"] = False

        def commit_model_hyperparameters(*, show_errors=True):
            """
            Parse the current hyperparameter editor as YAML and save it
            to the correct in-memory param_grids dictionary.
            """
            scope = current_model_key.get("scope", "")
            name = current_model_key.get("name", "")

            if not scope or not name:
                return True

            raw = hyper_text.get("1.0", tk.END).strip()

            if not raw:
                parsed = {}
            else:
                try:
                    parsed = yaml.safe_load(raw)
                except yaml.YAMLError as exc:
                    if show_errors:
                        messagebox.showerror(
                            "Invalid Hyperparameter Search Space",
                            (
                                f"Could not parse hyperparameters for {name}.\n\n"
                                "Enter valid YAML, for example:\n\n"
                                "C: [0.1, 1, 10]\n"
                                "kernel: [rbf, linear]\n\n"
                                f"YAML error:\n{exc}"
                            ),
                            parent=win
                        )
                    return False

            if parsed is None:
                parsed = {}

            if not isinstance(parsed, dict):
                if show_errors:
                    messagebox.showerror(
                        "Invalid Hyperparameter Search Space",
                        (
                            f"The hyperparameter search space for {name} must be a YAML mapping "
                            "(parameter names mapped to candidate values)."
                        ),
                        parent=win
                    )
                return False

            # Param-grid entries normally contain list candidate values, while
            # control keys such as _cv_ and _n_jobs_ may be scalars. We keep
            # the user's exact valid YAML structure rather than coercing types.
            set_model_grid(scope, name, parsed)
            return True

        def refresh_selected_model_from_yaml():
            """
            Re-read params.yaml from disk and reload the selected model's
            hyperparameter search space.

            This intentionally discards unsaved edits for the selected model
            and restores the latest values currently stored in params.yaml.
            Other in-memory feature-set/split/boundary edits are untouched.
            """
            selected_label = model_var.get().strip()
            _, mapping = model_display_items()
            scope_name = mapping.get(selected_label)

            if not scope_name:
                messagebox.showwarning(
                    "Model Required",
                    "Select a valid model before refreshing its search space.",
                    parent=win
                )
                return

            scope, name = scope_name

            try:
                latest_params = load_yaml_file(
                    PARAMS_FILE
                )
            except Exception as exc:
                messagebox.showerror(
                    "Refresh Error",
                    (
                        "Could not reload params.yaml:\n\n"
                        f"{exc}"
                    ),
                    parent=win
                )
                return

            if scope == "regression":
                latest_grids = latest_params.get(
                    "param_grids",
                    {}
                )
            else:
                latest_classification = latest_params.get(
                    "classification",
                    {}
                )
                if not isinstance(
                    latest_classification,
                    dict
                ):
                    latest_classification = {}

                latest_grids = latest_classification.get(
                    "param_grids",
                    {}
                )

            if not isinstance(
                latest_grids,
                dict
            ):
                latest_grids = {}

            if name not in latest_grids:
                messagebox.showwarning(
                    "Search Space Not Found",
                    (
                        f"No hyperparameter search space for '{name}' "
                        f"was found in params.yaml.\n\n"
                        "The current editor contents were not changed."
                    ),
                    parent=win
                )
                return

            latest_grid = latest_grids.get(
                name,
                {}
            )

            if latest_grid is None:
                latest_grid = {}

            if not isinstance(
                latest_grid,
                dict
            ):
                messagebox.showerror(
                    "Invalid Search Space",
                    (
                        f"The params.yaml entry for '{name}' is not a YAML mapping.\n\n"
                        "The current editor contents were not changed."
                    ),
                    parent=win
                )
                return

            # Replace only this model's in-memory grid with the latest YAML
            # version, then force a clean reload into the text editor.
            set_model_grid(
                scope,
                name,
                deepcopy(latest_grid)
            )

            current_model_key["scope"] = scope
            current_model_key["name"] = name

            state["loading"] = True
            try:
                hyper_text.configure(
                    state="normal"
                )
                hyper_text.delete(
                    "1.0",
                    tk.END
                )
                hyper_text.insert(
                    "1.0",
                    hyperparameter_yaml(
                        latest_grid
                    )
                )
            finally:
                state["loading"] = False

            hyper_hint_var.set(
                f"Reloaded {scope} hyperparameter search space for {name} "
                "directly from params.yaml."
            )

            status_var.set(
                f"Reloaded model search space from params.yaml: {scope} / {name}."
            )

        def restore_selected_model_preset():
            """Replace the selected model editor with its built-in preset."""
            selected_label = model_var.get().strip()
            _, mapping = model_display_items()
            scope_name = mapping.get(selected_label)
            if not scope_name:
                messagebox.showwarning(
                    "Model Required",
                    "Select a valid model before loading a preset.",
                    parent=win
                )
                return

            scope, name = scope_name
            if scope == "regression":
                preset_grid = preset_regression_grids.get(name)
            else:
                preset_grid = preset_classification_grids.get(name)

            if not isinstance(preset_grid, dict):
                messagebox.showinfo(
                    "Preset Not Available",
                    (
                        f"No predefined {scope} hyperparameter preset for '{name}' "
                        "was provided in configs/hyperparameter_presets.yaml.\n\n"
                        "You can still enter a custom YAML search space manually."
                    ),
                    parent=win
                )
                return

            set_model_grid(scope, name, deepcopy(preset_grid))
            current_model_key["scope"] = scope
            current_model_key["name"] = name
            state["loading"] = True
            try:
                hyper_text.configure(state="normal")
                hyper_text.delete("1.0", tk.END)
                hyper_text.insert("1.0", hyperparameter_yaml(preset_grid))
            finally:
                state["loading"] = False

            hyper_hint_var.set(
                f"Loaded built-in {scope} preset for {name}. "
                "Edit any candidate values here, then Save All to params.yaml."
            )
            status_var.set(f"Loaded hyperparameter preset: {scope} / {name}.")

        def restore_all_model_presets():
            """Reload every available built-in preset into the working profile."""
            if not preset_regression_grids and not preset_classification_grids:
                messagebox.showinfo(
                    "Presets Not Available",
                    "No built-in hyperparameter preset file was found.",
                    parent=win
                )
                return

            answer = messagebox.askyesno(
                "Load All Hyperparameter Presets",
                (
                    "Replace the current in-memory hyperparameter grids with the built-in "
                    "presets where presets are available?\n\n"
                    "Nothing is written to params.yaml until you click Save All."
                ),
                parent=win
            )
            if not answer:
                return

            for name, grid in preset_regression_grids.items():
                if isinstance(grid, dict):
                    working_regression_grids[name] = deepcopy(grid)
            for name, grid in preset_classification_grids.items():
                if isinstance(grid, dict):
                    working_classification_grids[name] = deepcopy(grid)

            # Ensure preset model names are visible in the model selector.
            for name in preset_regression_models:
                name = str(name).strip()
                if name and name not in working_regression_models:
                    working_regression_models.append(name)
            for name in preset_classification_models:
                name = str(name).strip()
                if name and name not in working_classification_models:
                    working_classification_models.append(name)

            selected = model_var.get().strip()
            refresh_model_combo(select_label=selected)
            if model_var.get().strip():
                load_model_hyperparameters(model_var.get().strip())

            status_var.set("Loaded all available built-in hyperparameter presets.")

        def on_model_changed(event=None):
            """
            Reload the Hyperparameter Search Space only after a real model
            item has been selected from the dropdown.

            The standard <<ComboboxSelected>> event is used because it fires
            after Tk has committed the selected popup item to model_var.
            """
            if state["loading"]:
                return

            selected_label = model_var.get().strip()
            items, mapping = model_display_items()

            # Never destroy/replace the current editor contents for a blank or
            # unresolved transient combobox value.
            if not selected_label or selected_label not in mapping:
                return

            previous_scope = current_model_key.get("scope", "")
            previous_name = current_model_key.get("name", "")
            previous_label = (
                f"{previous_scope.capitalize()}: {previous_name}"
                if previous_scope and previous_name
                else ""
            )

            # Commit edits for the previously loaded model before switching.
            if previous_scope and previous_name:
                if not commit_model_hyperparameters():
                    state["loading"] = True
                    try:
                        if previous_label:
                            model_var.set(previous_label)
                    finally:
                        state["loading"] = False
                    return

            # At this point selected_label is guaranteed to map to a real
            # regression/classification model, so reload its stored grid.
            load_model_hyperparameters(selected_label)


        def get_target():

            return (
                target_var
                .get()
                .strip()
            )

        def get_set_name():

            return (
                current_set_var
                .get()
                .strip()
            )

        def load_target_boundaries(target=None):
            """Load configured classification limits for one target."""
            if target is None:
                target = get_target()

            state["loading"] = True
            try:
                lower_boundary_var.set("")
                upper_boundary_var.set("")
                class_mode_var.set(str(working_target_modes.get(target, "auto") or "auto"))

                cfg = working_class_boundaries.get(
                    target,
                    {}
                )

                if isinstance(cfg, dict):
                    lower = cfg.get("lower")
                    upper = cfg.get("upper")

                    if lower is not None:
                        lower_boundary_var.set(str(lower))
                    if upper is not None:
                        upper_boundary_var.set(str(upper))
            finally:
                state["loading"] = False

        def commit_target_boundaries(
            target=None,
            *,
            show_errors=True
        ):
            """
            Store separate lower/upper classification boundaries.

            Leaving both inputs blank removes explicit limits for this target,
            allowing downstream code to infer them automatically.
            """
            if target is None:
                target = get_target()

            if not target:
                return True

            mode = str(class_mode_var.get() or "auto").strip().lower()
            if mode not in {"auto", "threshold", "categorical"}:
                mode = "auto"
            if mode == "auto":
                working_target_modes.pop(target, None)
            else:
                working_target_modes[target] = mode

            lower_text = lower_boundary_var.get().strip()
            upper_text = upper_boundary_var.get().strip()

            if mode == "categorical":
                working_class_boundaries.pop(target, None)
                return True

            if not lower_text and not upper_text:
                working_class_boundaries.pop(
                    target,
                    None
                )
                return True

            if not lower_text or not upper_text:
                if show_errors:
                    messagebox.showwarning(
                        "Classification Boundaries",
                        (
                            "Enter both Lower and Upper class boundaries, "
                            "or leave both fields blank for automatic limits."
                        ),
                        parent=win
                    )
                return False

            try:
                lower = float(lower_text)
                upper = float(upper_text)
            except ValueError:
                if show_errors:
                    messagebox.showwarning(
                        "Classification Boundaries",
                        "Lower and Upper class boundaries must be numeric.",
                        parent=win
                    )
                return False

            if lower >= upper:
                if show_errors:
                    messagebox.showwarning(
                        "Classification Boundaries",
                        "Lower class boundary must be smaller than Upper class boundary.",
                        parent=win
                    )
                return False

            existing = working_class_boundaries.get(
                target,
                {}
            )
            if not isinstance(existing, dict):
                existing = {}

            updated = deepcopy(existing)
            updated["lower"] = lower
            updated["upper"] = upper

            working_class_boundaries[target] = updated
            return True

        def ensure_target(target):

            if target not in working_targets:

                working_targets[target] = {}

            if not isinstance(
                working_targets[target],
                dict
            ):

                working_targets[target] = {}

            return working_targets[target]

        def all_target_names():
            """
            Targets shown in the combo include:
            - targets already in params.yaml
            - every CSV column, so a user can choose another
              target without manually typing it.
            """

            result = []

            for name in (
                list(
                    working_targets
                    .keys()
                )
                + dataset_columns
            ):

                if name not in result:
                    result.append(name)

            return result

        def target_set_names(target):

            target_sets = (
                working_targets
                .get(
                    target,
                    {}
                )
            )

            if not isinstance(
                target_sets,
                dict
            ):
                return []

            return sorted(
                target_sets.keys(),
                key=natural_sort_key
            )

        def displayed_features():

            return list(
                feature_list.get(
                    0,
                    tk.END
                )
            )

        def refresh_available():

            available_list.delete(
                0,
                tk.END
            )

            target = get_target()

            used = set(
                displayed_features()
            )

            for column in dataset_columns:

                if column == target:
                    continue

                if column in used:
                    continue

                available_list.insert(
                    tk.END,
                    column
                )

        def clear_editor():

            feature_list.delete(
                0,
                tk.END
            )

            current_set_var.set("")
            current_target_var.set("")

            set_title_var.set(
                "Feature Set Inputs"
            )

            refresh_available()

        def load_set(set_name):

            target = get_target()

            if not target or not set_name:

                clear_editor()

                return

            target_sets = ensure_target(
                target
            )

            values = target_sets.get(
                set_name,
                []
            )

            if values is None:
                values = []

            if not isinstance(
                values,
                list
            ):

                values = list(
                    values
                ) if isinstance(
                    values,
                    (
                        tuple,
                        set
                    )
                ) else []

            state["loading"] = True

            try:
                feature_list.delete(
                    0,
                    tk.END
                )

                for item in values:

                    feature_list.insert(
                        tk.END,
                        str(item)
                    )

                current_set_var.set(
                    set_name
                )
                current_target_var.set(
                    target
                )

                set_title_var.set(
                    f"Inputs: {set_name}"
                )

                refresh_available()

            finally:
                state["loading"] = False

            status_var.set(
                (
                    f"Editing {target} / "
                    f"{set_name} "
                    f"({len(values)} feature(s))."
                )
            )

        def refresh_target_combo():

            values = all_target_names()

            target_combo[
                "values"
            ] = values

            current = get_target()

            if current in values:
                return

            # Prefer the first target already defined in YAML.
            existing = list(
                working_targets.keys()
            )

            if existing:

                target_var.set(
                    existing[0]
                )

            elif values:

                target_var.set(
                    values[0]
                )

        def refresh_set_list(
            select_name=None
        ):

            target = get_target()

            names = target_set_names(
                target
            )

            state["loading"] = True

            try:
                feature_set_list.delete(
                    0,
                    tk.END
                )

                for name in names:

                    features = (
                        working_targets
                        .get(
                            target,
                            {}
                        )
                        .get(
                            name,
                            []
                        )
                    )

                    if not isinstance(
                        features,
                        list
                    ):
                        features = []

                    # Show current content directly.
                    display = (
                        f"{name}  "
                        f"[{len(features)}]  "
                        f"{', '.join(map(str, features))}"
                    )

                    feature_set_list.insert(
                        tk.END,
                        display
                    )

                if not names:

                    clear_editor()

                    status_var.set(
                        (
                            f"No feature sets "
                            f"defined for "
                            f"target '{target}'."
                        )
                    )

                    return

                if select_name in names:

                    index = names.index(
                        select_name
                    )

                else:
                    index = 0

                feature_set_list.selection_set(
                    index
                )

                feature_set_list.activate(
                    index
                )

                feature_set_list.see(
                    index
                )

            finally:
                state["loading"] = False

            load_set(
                names[index]
            )

        def selected_set_from_list():

            selected = (
                feature_set_list
                .curselection()
            )

            if not selected:
                return ""

            names = target_set_names(
                get_target()
            )

            index = selected[0]

            if index >= len(names):
                return ""

            return names[index]

        def commit_editor(
            *,
            require_features=False,
            refresh=True
        ):
            """
            Copy currently displayed feature inputs into the
            in-memory working TARGETS dictionary.
            """

            # Commit to the target that owns the loaded editor set, not
            # necessarily the target currently shown in the combobox.
            # Tk updates the combobox value before <<ComboboxSelected>> fires;
            # without this ownership tracking, switching PL -> LL can create
            # a ghost LL/fs1 entry from the old PL/fs1 editor.
            target = (
                current_target_var
                .get()
                .strip()
            )
            set_name = get_set_name()

            if not target or not set_name:
                return True

            features = [
                value
                for value
                in displayed_features()
                if value != target
            ]

            if (
                require_features
                and not features
            ):

                messagebox.showwarning(
                    "Features Required",
                    (
                        f"Feature set "
                        f"'{set_name}' must "
                        "contain at least one "
                        "input feature."
                    ),
                    parent=win
                )

                return False

            target_sets = ensure_target(
                target
            )

            target_sets[
                set_name
            ] = features

            if refresh:

                refresh_set_list(
                    select_name=set_name
                )

            return True

        def on_target_changed(
            event=None
        ):

            if state["loading"]:
                return

            commit_editor(
                require_features=False,
                refresh=False
            )

            previous_boundary_target = (
                boundary_target_var.get().strip()
            )

            if previous_boundary_target:
                if not commit_target_boundaries(
                    previous_boundary_target
                ):
                    state["loading"] = True
                    try:
                        target_var.set(
                            previous_boundary_target
                        )
                    finally:
                        state["loading"] = False
                    return

            target = get_target()

            if not target:
                clear_editor()
                lower_boundary_var.set("")
                upper_boundary_var.set("")
                boundary_target_var.set("")
                return

            refresh_set_list()
            load_target_boundaries(target)
            boundary_target_var.set(target)

        def on_set_selected(
            event=None
        ):

            if state["loading"]:
                return

            selected_name = (
                selected_set_from_list()
            )

            if not selected_name:
                return

            old_name = get_set_name()

            if (
                old_name
                and old_name
                != selected_name
            ):

                commit_editor(
                    require_features=False,
                    refresh=False
                )

            load_set(
                selected_name
            )

        def add_selected_features():

            indices = (
                available_list
                .curselection()
            )

            if not indices:
                return

            target = get_target()

            existing = set(
                displayed_features()
            )

            for index in indices:

                value = (
                    available_list
                    .get(index)
                )

                if value == target:
                    continue

                if value in existing:
                    continue

                feature_list.insert(
                    tk.END,
                    value
                )

                existing.add(value)

            refresh_available()

            status_var.set(
                (
                    "Feature selection "
                    "changed. Press Update "
                    "or Save All."
                )
            )

        def remove_selected_features():

            indices = (
                feature_list
                .curselection()
            )

            if not indices:
                return

            for index in reversed(
                indices
            ):

                feature_list.delete(
                    index
                )

            refresh_available()

            status_var.set(
                (
                    "Feature selection "
                    "changed. Press Update "
                    "or Save All."
                )
            )

        def next_feature_set_name():

            target = get_target()

            existing = set(
                target_set_names(
                    target
                )
            )

            number = 1

            while (
                f"fs{number}"
                in existing
            ):
                number += 1

            return f"fs{number}"

        def create_feature_set():

            target = get_target()

            if not target:

                messagebox.showwarning(
                    "Target Required",
                    (
                        "Select a target "
                        "before creating a "
                        "feature set."
                    ),
                    parent=win
                )

                return

            commit_editor(
                require_features=False,
                refresh=False
            )

            name = (
                simpledialog
                .askstring(
                    "New Feature Set",
                    (
                        "Enter a name for "
                        "the new feature set:"
                    ),
                    initialvalue=(
                        next_feature_set_name()
                    ),
                    parent=win
                )
            )

            if name is None:
                return

            name = name.strip()

            if not name:

                messagebox.showwarning(
                    "Invalid Name",
                    (
                        "Feature-set name "
                        "cannot be empty."
                    ),
                    parent=win
                )

                return

            target_sets = ensure_target(
                target
            )

            if name in target_sets:

                messagebox.showwarning(
                    "Already Exists",
                    (
                        f"'{name}' already "
                        "exists for "
                        f"target '{target}'."
                    ),
                    parent=win
                )

                return

            target_sets[name] = []

            refresh_set_list(
                select_name=name
            )

            status_var.set(
                (
                    f"Created {target} / "
                    f"{name}. Select one or "
                    "more input features."
                )
            )

        def duplicate_feature_set():

            target = get_target()
            source_name = (
                selected_set_from_list()
                or get_set_name()
            )

            if not target or not source_name:

                messagebox.showwarning(
                    "Select Feature Set",
                    (
                        "Select a feature set "
                        "to duplicate."
                    ),
                    parent=win
                )

                return

            commit_editor(
                require_features=False,
                refresh=False
            )

            target_sets = ensure_target(
                target
            )

            base_name = (
                f"{source_name}_copy"
            )

            new_name = base_name
            counter = 2

            while new_name in target_sets:

                new_name = (
                    f"{base_name}_{counter}"
                )

                counter += 1

            target_sets[new_name] = deepcopy(
                target_sets.get(
                    source_name,
                    []
                )
            )

            refresh_set_list(
                select_name=new_name
            )

            status_var.set(
                (
                    f"Duplicated "
                    f"{source_name} as "
                    f"{new_name}."
                )
            )

        def rename_feature_set():

            target = get_target()

            old_name = (
                selected_set_from_list()
                or get_set_name()
            )

            if not target or not old_name:

                messagebox.showwarning(
                    "Select Feature Set",
                    (
                        "Select a feature set "
                        "to rename."
                    ),
                    parent=win
                )

                return

            commit_editor(
                require_features=False,
                refresh=False
            )

            new_name = (
                simpledialog
                .askstring(
                    "Rename Feature Set",
                    "New name:",
                    initialvalue=old_name,
                    parent=win
                )
            )

            if new_name is None:
                return

            new_name = new_name.strip()

            if not new_name:

                messagebox.showwarning(
                    "Invalid Name",
                    (
                        "Feature-set name "
                        "cannot be empty."
                    ),
                    parent=win
                )

                return

            target_sets = ensure_target(
                target
            )

            if (
                new_name != old_name
                and new_name in target_sets
            ):

                messagebox.showwarning(
                    "Already Exists",
                    (
                        f"'{new_name}' "
                        "already exists."
                    ),
                    parent=win
                )

                return

            updated = {}

            for (
                name,
                values
            ) in target_sets.items():

                if name == old_name:

                    updated[
                        new_name
                    ] = values

                else:

                    updated[
                        name
                    ] = values

            working_targets[
                target
            ] = updated

            current_set_var.set(
                new_name
            )

            refresh_set_list(
                select_name=new_name
            )

            status_var.set(
                (
                    f"Renamed {old_name} "
                    f"to {new_name}."
                )
            )

        def delete_feature_set():

            target = get_target()

            set_name = (
                selected_set_from_list()
                or get_set_name()
            )

            if not target or not set_name:

                messagebox.showwarning(
                    "Select Feature Set",
                    (
                        "Select a feature set "
                        "to delete."
                    ),
                    parent=win
                )

                return

            answer = (
                messagebox
                .askyesno(
                    "Delete Feature Set",
                    (
                        f"Delete '{set_name}' "
                        f"from target "
                        f"'{target}'?\n\n"
                        "The file is not "
                        "changed until Save "
                        "All is pressed."
                    ),
                    parent=win
                )
            )

            if not answer:
                return

            target_sets = working_targets.get(
                target,
                {}
            )

            if isinstance(
                target_sets,
                dict
            ):
                target_sets.pop(
                    set_name,
                    None
                )

            target_removed = False

            # Remove a target when its last feature set is deleted.
            # This prevents empty YAML entries such as ``AC: {}``.
            if (
                not isinstance(
                    target_sets,
                    dict
                )
                or not target_sets
            ):
                working_targets.pop(
                    target,
                    None
                )
                target_removed = True

            clear_editor()

            if target_removed:
                remaining_targets = list(
                    working_targets.keys()
                )

                if remaining_targets:
                    target_var.set(
                        remaining_targets[0]
                    )
                else:
                    target_var.set(
                        ""
                    )

                refresh_target_combo()
                refresh_set_list()

                status_var.set(
                    (
                        f"Deleted {target} / "
                        f"{set_name}. Target "
                        f"'{target}' was also "
                        "removed because it has "
                        "no remaining feature sets."
                    )
                )

            else:
                refresh_set_list()

                status_var.set(
                    (
                        f"Deleted {target} / "
                        f"{set_name} in memory."
                    )
                )

        def update_current_set():

            if not get_set_name():

                messagebox.showwarning(
                    "No Feature Set",
                    (
                        "Select or create a "
                        "feature set first."
                    ),
                    parent=win
                )

                return

            if commit_editor(
                require_features=True,
                refresh=True
            ):

                status_var.set(
                    (
                        f"Updated "
                        f"{get_target()} / "
                        f"{get_set_name()} "
                        "in memory."
                    )
                )

        def add_target():

            candidates = [
                column
                for column
                in dataset_columns
                if column
                not in working_targets
            ]

            if not candidates:

                messagebox.showinfo(
                    "No New Target",
                    (
                        "Every dataset column "
                        "is already available."
                    ),
                    parent=win
                )

                return

            dialog = tk.Toplevel(win)

            dialog.title(
                "Add Target"
            )

            dialog.transient(win)
            dialog.grab_set()
            dialog.resizable(
                False,
                False
            )

            body = ttk.Frame(
                dialog,
                padding=12
            )

            body.pack(
                fill="both",
                expand=True
            )

            value_var = tk.StringVar(
                value=candidates[0]
            )

            ttk.Label(
                body,
                text=(
                    "Select a dataset column "
                    "to use as a target:"
                )
            ).pack(
                anchor="w",
                pady=(
                    0,
                    6
                )
            )

            combo = ttk.Combobox(
                body,
                textvariable=value_var,
                values=candidates,
                state="readonly",
                width=34
            )

            combo.pack(
                fill="x",
                pady=(
                    0,
                    8
                )
            )

            def accept_target():

                target = (
                    value_var
                    .get()
                    .strip()
                )

                if not target:
                    return

                # Selecting/adding a target prepares it for feature-set
                # creation, but does not persist an empty ``target: {}`` entry.
                refresh_target_combo()

                target_var.set(
                    target
                )

                refresh_set_list()
                load_target_boundaries(target)
                boundary_target_var.set(target)

                dialog.destroy()

            target_actions = ttk.Frame(
                body
            )

            target_actions.pack(
                fill="x"
            )

            ttk.Button(
                target_actions,
                text="Add Target",
                command=accept_target
            ).pack(
                side="right"
            )

        def prune_empty_target_entries():
            """Remove stale/unfinished empty entries from working TARGETS.

            Empty feature sets can be left behind if a new set is created and
            the user changes target before adding features.  They are not
            useful configuration and should never block saving a different,
            valid feature set.  Empty target mappings are removed as well.
            """

            for target in list(working_targets.keys()):
                sets = working_targets.get(target)

                if not isinstance(sets, dict):
                    continue

                for set_name in list(sets.keys()):
                    features = sets.get(set_name)
                    if isinstance(features, list) and not features:
                        sets.pop(set_name, None)

                if not sets:
                    working_targets.pop(target, None)

        def validate_all_targets():
            """
            Validate the complete in-memory TARGETS section before
            previewing or saving.
            """

            # Remove ghost/unfinished empty sets such as LL/fs1 = [] before
            # validating valid sets such as PL/fs1 = [AC, g].
            prune_empty_target_entries()

            for (
                target,
                sets
            ) in working_targets.items():

                if not isinstance(
                    sets,
                    dict
                ):

                    return (
                        False,
                        (
                            f"TARGETS -> "
                            f"{target} must be "
                            "a mapping of "
                            "feature sets."
                        )
                    )

                for (
                    set_name,
                    features
                ) in sets.items():

                    if not isinstance(
                        features,
                        list
                    ):

                        return (
                            False,
                            (
                                f"{target} / "
                                f"{set_name} must "
                                "contain a list "
                                "of features."
                            )
                        )

                    if not features:

                        return (
                            False,
                            (
                                f"{target} / "
                                f"{set_name} is "
                                "empty."
                            )
                        )

                    if target in features:

                        return (
                            False,
                            (
                                f"Target '{target}' "
                                "appears inside "
                                f"{set_name}. "
                                "This would cause "
                                "target leakage."
                            )
                        )

                    duplicates = [
                        value
                        for value
                        in set(features)
                        if features.count(
                            value
                        ) > 1
                    ]

                    if duplicates:

                        return (
                            False,
                            (
                                f"{target} / "
                                f"{set_name} has "
                                "duplicate feature"
                                f"(s): "
                                f"{duplicates}"
                            )
                        )

            return (
                True,
                ""
            )

        def active_classification_targets():
            """
            Return active classification targets from TARGETS.

            A target is active only when it has at least one non-empty
            feature set. This prevents stale entries such as
            classification.targets: [AC] while TARGETS contains only DMax.
            """
            active = []

            for target, sets in working_targets.items():
                if not isinstance(sets, dict):
                    continue

                has_usable_set = any(
                    isinstance(features, list) and bool(features)
                    for features in sets.values()
                )

                if has_usable_set:
                    active.append(str(target))

            return active

        def build_preview_params():

            latest = load_yaml_file(
                PARAMS_FILE
            )

            latest[
                "TARGETS"
            ] = deepcopy(
                working_targets
            )

            latest[
                "split"
            ] = deepcopy(
                working_split
            )

            # Keep active model lists unchanged, but save all edited grids.
            latest[
                "models"
            ] = deepcopy(
                working_regression_models
            )
            latest[
                "param_grids"
            ] = deepcopy(
                working_regression_grids
            )

            classification = latest.get(
                "classification",
                {}
            )
            if not isinstance(classification, dict):
                classification = {}

            classification[
                "class_boundaries"
            ] = deepcopy(
                working_class_boundaries
            )
            classification["target_modes"] = deepcopy(working_target_modes)

            classification[
                "targets"
            ] = active_classification_targets()

            classification[
                "models"
            ] = deepcopy(
                working_classification_models
            )
            classification[
                "param_grids"
            ] = deepcopy(
                working_classification_grids
            )

            latest[
                "classification"
            ] = classification

            return latest

        def show_yaml_preview():

            if get_set_name():

                if not commit_editor(
                    require_features=True,
                    refresh=False
                ):
                    return

            if not commit_split_settings():
                return

            if not commit_model_hyperparameters():
                return

            boundary_target = (
                boundary_target_var.get().strip()
                or get_target()
            )
            if not commit_target_boundaries(
                boundary_target
            ):
                return

            valid, error = (
                validate_all_targets()
            )

            if not valid:

                messagebox.showerror(
                    "Invalid Feature Sets",
                    error,
                    parent=win
                )

                return

            preview_data = (
                build_preview_params()
            )

            preview = tk.Toplevel(win)

            preview.title(
                "params.yaml Preview"
            )

            self._fit_child_to_screen(
                preview,
                width_ratio=0.72,
                height_ratio=0.72
            )

            preview.transient(win)
            preview.grab_set()

            frame = ttk.Frame(
                preview,
                padding=8
            )

            frame.pack(
                fill="both",
                expand=True
            )

            frame.columnconfigure(
                0,
                weight=1
            )

            frame.rowconfigure(
                0,
                weight=1
            )

            text = tk.Text(
                frame,
                wrap="none",
                font=(
                    "Consolas",
                    9
                )
            )

            text.grid(
                row=0,
                column=0,
                sticky="nsew"
            )

            yscroll = ttk.Scrollbar(
                frame,
                orient="vertical",
                command=text.yview
            )

            yscroll.grid(
                row=0,
                column=1,
                sticky="ns"
            )

            xscroll = ttk.Scrollbar(
                frame,
                orient="horizontal",
                command=text.xview
            )

            xscroll.grid(
                row=1,
                column=0,
                sticky="ew"
            )

            text.configure(
                yscrollcommand=(
                    yscroll.set
                ),
                xscrollcommand=(
                    xscroll.set
                )
            )

            text.insert(
                "1.0",
                yaml_text(
                    preview_data
                )
            )

            text.configure(
                state="disabled"
            )

            preview_actions = ttk.Frame(
                frame
            )

            preview_actions.grid(
                row=2,
                column=0,
                columnspan=2,
                sticky="e",
                pady=(
                    6,
                    0
                )
            )

            ttk.Button(
                preview_actions,
                text="Close Preview",
                command=(
                    preview.destroy
                )
            ).pack(
                side="left"
            )

        def save_all():

            if get_set_name():

                if not commit_editor(
                    require_features=True,
                    refresh=False
                ):
                    return

            if not commit_split_settings():
                return

            if not commit_model_hyperparameters():
                return

            boundary_target = (
                boundary_target_var.get().strip()
                or get_target()
            )
            if not commit_target_boundaries(
                boundary_target
            ):
                return

            valid, error = (
                validate_all_targets()
            )

            if not valid:

                messagebox.showerror(
                    "Invalid Feature Sets",
                    error,
                    parent=win
                )

                return

            try:
                # Reload immediately before writing so every unrelated
                # configuration section is preserved.
                latest_params = (
                    load_yaml_file(
                        PARAMS_FILE
                    )
                )

                backup_file = (
                    make_params_backup(
                        PARAMS_FILE,
                        PARAMS_BACKUP_DIR
                    )
                )

                latest_params[
                    "TARGETS"
                ] = deepcopy(
                    working_targets
                )

                latest_params[
                    "split"
                ] = deepcopy(
                    working_split
                )

                latest_params[
                    "models"
                ] = deepcopy(
                    working_regression_models
                )
                latest_params[
                    "param_grids"
                ] = deepcopy(
                    working_regression_grids
                )

                classification = latest_params.get(
                    "classification",
                    {}
                )
                if not isinstance(classification, dict):
                    classification = {}

                classification[
                    "class_boundaries"
                ] = deepcopy(
                    working_class_boundaries
                )

                classification[
                    "targets"
                ] = active_classification_targets()

                classification[
                    "models"
                ] = deepcopy(
                    working_classification_models
                )
                classification[
                    "param_grids"
                ] = deepcopy(
                    working_classification_grids
                )

                latest_params[
                    "classification"
                ] = classification

                PARAMS_FILE.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                with PARAMS_FILE.open(
                    "w",
                    encoding="utf-8"
                ) as stream:

                    yaml.safe_dump(
                        latest_params,
                        stream,
                        sort_keys=False,
                        allow_unicode=True,
                        default_flow_style=False
                    )

                self.write_log(
                    (
                        "\nparams.yaml feature sets, split settings, "
                        "model hyperparameters, and classification settings saved."
                    )
                )

                self.write_log(
                    (
                        "Split: Train="
                        f"{working_split.get('train', 0) * 100:g}% | "
                        "Test="
                        f"{working_split.get('test', 0) * 100:g}%"
                    )
                )

                self.write_log(
                    (
                        f"Modified: "
                        f"{PARAMS_FILE}"
                    )
                )

                if backup_file:

                    self.write_log(
                        (
                            f"Backup:   "
                            f"{backup_file}"
                        )
                    )

                backup_text = (
                    (
                        "\n\nBackup:\n"
                        f"{backup_file}"
                    )
                    if backup_file
                    else (
                        "\n\nNo previous "
                        "params.yaml existed, "
                        "so no backup was "
                        "required."
                    )
                )

                messagebox.showinfo(
                    "Saved",
                    (
                        "Feature sets were "
                        "saved successfully.\n\n"
                        "All unrelated "
                        "params.yaml sections "
                        "were preserved."
                        f"{backup_text}"
                    ),
                    parent=win
                )

                # Keep the Target and Feature Set Manager open after saving.
                # The user can continue editing and save again without
                # returning to the main SOIL MLOPS window.

            except Exception as exc:

                messagebox.showerror(
                    "Save Error",
                    (
                        "Could not save "
                        "params.yaml:\n\n"
                        f"{exc}"
                    ),
                    parent=win
                )

        # ====================================================
        # FEATURE TRANSFER BUTTONS
        # ====================================================

        arrow_bar.columnconfigure(
            5,
            weight=1
        )

        tk.Button(
            arrow_bar,
            text="➜ Add",
            font=(
                "Segoe UI",
                9,
                "bold"
            ),
            bg="#4472C4",
            fg="white",
            activebackground="#365F91",
            activeforeground="white",
            relief="flat",
            padx=10,
            pady=3,
            cursor="hand2",
            command=add_selected_features
        ).grid(
            row=0,
            column=1,
            padx=(
                3,
                2
            )
        )

        tk.Button(
            arrow_bar,
            text="⬅ Remove",
            font=(
                "Segoe UI",
                9,
                "bold"
            ),
            bg="#4472C4",
            fg="white",
            activebackground="#365F91",
            activeforeground="white",
            relief="flat",
            padx=10,
            pady=3,
            cursor="hand2",
            command=remove_selected_features
        ).grid(
            row=0,
            column=3,
            padx=(
                5,
                2
            )
        )

        # ====================================================
        # TOOLBAR BUTTONS
        # ====================================================

        buttons = [
            (
                "New Set",
                create_feature_set
            ),
            (
                "Duplicate",
                duplicate_feature_set
            ),
            (
                "Rename",
                rename_feature_set
            ),
            (
                "Delete",
                delete_feature_set
            ),
            (
                "Update Set",
                update_current_set
            ),
            (
                "Add Target",
                add_target
            ),
            (
                "YAML Preview",
                show_yaml_preview
            ),
        ]

        for (
            label,
            command
        ) in buttons:

            ttk.Button(
                toolbar,
                text=label,
                style="Toolbar.TButton",
                command=command
            ).pack(
                side="left",
                padx=(
                    0,
                    5
                )
            )

        # ====================================================
        # BOTTOM BUTTONS
        # ====================================================

        cancel_holder = ttk.Frame(
            bottom
        )

        cancel_holder.grid(
            row=0,
            column=1,
            padx=(
                4,
                0
            )
        )

        ttk.Button(
            cancel_holder,
            text="Cancel",
            command=(
                win.destroy
            )
        ).pack(
            side="left"
        )

        save_holder = ttk.Frame(
            bottom
        )

        save_holder.grid(
            row=0,
            column=2,
            padx=(
                6,
                0
            )
        )

        ttk.Button(
            save_holder,
            text=(
                "Save All to "
                "params.yaml"
            ),
            command=save_all
        ).pack(
            side="left"
        )

        # ====================================================
        # EVENTS
        # ====================================================

        target_combo.bind(
            "<<ComboboxSelected>>",
            on_target_changed
        )

        refresh_model_btn.configure(
            command=refresh_selected_model_from_yaml
        )
        restore_preset_btn.configure(
            command=restore_selected_model_preset
        )
        restore_all_presets_btn.configure(
            command=restore_all_model_presets
        )

        model_combo.bind(
            "<<ComboboxSelected>>",
            on_model_changed
        )

        # <<ComboboxSelected>> is the authoritative event.  It fires only
        # after the user has actually chosen an item from the popup, so the
        # selected model name is already committed to model_var.
        #
        # Do NOT bind <ButtonRelease-1> here: clicking the dropdown arrow or
        # popup can fire that mouse event before selection is finalized and
        # may incorrectly reload an empty grid as "{}".
        model_combo.bind(
            "<Return>",
            lambda event: (
                on_model_changed()
                if model_var.get().strip()
                else None
            ),
            add="+"
        )

        feature_set_list.bind(
            "<<ListboxSelect>>",
            on_set_selected
        )

        available_list.bind(
            "<Double-Button-1>",
            lambda event: (
                add_selected_features()
            )
        )

        feature_list.bind(
            "<Double-Button-1>",
            lambda event: (
                remove_selected_features()
            )
        )

        # Keyboard shortcuts.
        win.bind(
            "<Control-s>",
            lambda event: save_all()
        )

        win.bind(
            "<Control-p>",
            lambda event: (
                show_yaml_preview()
            )
        )

        # ====================================================
        # INITIAL LOAD
        # ====================================================

        load_split_settings()
        refresh_target_combo()

        refresh_model_combo()
        if model_var.get():
            load_model_hyperparameters(
                model_var.get()
            )

        if get_target():

            # Do not create an empty target just because it is selected in the
            # combobox. This is important when params.yaml contains TARGETS: {}.
            refresh_set_list()
            load_target_boundaries(
                get_target()
            )
            boundary_target_var.set(
                get_target()
            )

        else:

            refresh_available()
            lower_boundary_var.set("")
            upper_boundary_var.set("")
            boundary_target_var.set("")

        # Give the three panes useful starting proportions after
        # Tk has calculated their actual size.
        win.after(
            120,
            lambda: self._set_default_pane_positions(
                panes
            )
        )

    # ========================================================
    # RESPONSIVE CHILD WINDOW
    # ========================================================

    def _fit_child_to_screen(
        self,
        window,
        width_ratio=0.90,
        height_ratio=0.80
    ):
        """
        Create a child window that remains inside the screen.

        Important:
        - No huge fixed minimum size.
        - Leaves room for the Windows taskbar.
        - Works more reliably under 125%, 150%, and 175% display scaling.
        """

        window.update_idletasks()

        screen_w = (
            window
            .winfo_screenwidth()
        )

        screen_h = (
            window
            .winfo_screenheight()
        )

        width = min(
            int(
                screen_w
                * width_ratio
            ),
            max(
                760,
                screen_w - 60
            )
        )

        height = min(
            int(
                screen_h
                * height_ratio
            ),
            max(
                500,
                screen_h - 120
            )
        )

        # Never ask Tk for a size larger than the usable display.
        width = min(
            width,
            screen_w - 40
        )

        height = min(
            height,
            screen_h - 90
        )

        # On very small logical desktops, preserve usability rather than
        # enforcing a large fixed minimum.
        width = max(
            700,
            width
        )

        height = max(
            460,
            height
        )

        x = max(
            10,
            (
                screen_w
                - width
            ) // 2
        )

        y = max(
            10,
            min(
                25,
                (
                    screen_h
                    - height
                ) // 2
            )
        )

        window.geometry(
            (
                f"{width}x{height}"
                f"+{x}+{y}"
            )
        )

        window.minsize(
            680,
            440
        )

    @staticmethod
    def _set_default_pane_positions(
        panes
    ):
        """
        Set reasonable initial sash positions without preventing the
        user from dragging the panes.
        """

        try:
            width = panes.winfo_width()

            if width <= 30:
                return

            # ttk.Panedwindow uses sashpos.
            # Four panes: Available | Inputs | Feature Sets |
            # Training/Search-Space Configuration.
            panes.sashpos(
                0,
                int(width * 0.12)
            )

            panes.sashpos(
                1,
                int(width * 0.33)
            )

            # Move the left edge of the Training/Search-Space pane to the
            # left.  The far-right pane now starts at about 62% of the
            # manager width instead of 70%, giving it about 38% of the
            # available width.
            panes.sashpos(
                2,
                int(width * 0.62)
            )

        except Exception:
            # Some Tk builds may not expose sashpos consistently.
            # The weighted panes still remain functional.
            pass

    # ========================================================
    # TRAINING
    # ========================================================

    def run_training(self):

        # Before starting the pipeline, confirm that the selected MLOps
        # project contains a usable src folder and run_pipeline.py.
        src_dir = BASE_DIR / "src"

        src_missing_or_empty = (
            not src_dir.is_dir()
            or not any(src_dir.iterdir())
        )

        pipeline_missing = (
            RUN_PIPELINE is None
            or not Path(RUN_PIPELINE).is_file()
        )

        if src_missing_or_empty or pipeline_missing:
            messagebox.showwarning(
                "No Files to Run",
                "do not any file for runing",
                parent=self,
            )
            return

        dataset = Path(
            self.selected_csv.get()
        )

        if not dataset.exists():

            messagebox.showerror(
                "Dataset Missing",
                (
                    "Select a valid dataset "
                    "before training."
                )
            )

            return

        selected_tasks = []
        if self.run_regression_var.get():
            selected_tasks.append("regression")
        if self.run_classification_var.get():
            selected_tasks.append("classification")

        if not selected_tasks:
            messagebox.showwarning(
                "No Model Group Selected",
                "Select Regression, Classification, or both before running the pipeline.",
                parent=self,
            )
            return

        command = [
            sys.executable,
            str(RUN_PIPELINE),
            "--params",
            str(PARAMS_FILE),
            "--data",
            str(dataset),
            "--tasks",
            *selected_tasks,
        ]

        def start_training():
            self.run_command_async(
                command,
                (
                    "Starting SOIL MLOPS "
                    "training pipeline..."
                )
            )

        # Cleanup is offered ONLY here, after Run Training Pipeline is clicked.
        # The user can keep everything or remove any/all safe folders first.
        self.offer_startup_cleanup(start_training)

    # ========================================================
    # COMMAND EXECUTION
    # ========================================================

    def run_command_async(
        self,
        command,
        starting_message,
        on_success=None
    ):
        """
        Run one external program and stream its complete console output
        into the GUI in execution order.

        stdout and stderr are merged intentionally so their relative order
        is preserved as closely as the child process emits them.
        """

        if self._process_running:
            self.write_log(
                "[GUI][WARN] Another program is already running. "
                "Wait for it to finish before starting a new one."
            )
            messagebox.showwarning(
                "Program Already Running",
                (
                    "Another SOIL MLOPS program is currently running.\n\n"
                    "Wait for it to finish before starting another program."
                )
            )
            return

        self._process_running = True
        self._run_counter += 1
        run_id = self._run_counter

        # Make Python subprocess output line-buffered/unbuffered so console
        # messages appear in the GUI immediately.
        command_to_run = list(command)
        if (
            command_to_run
            and Path(str(command_to_run[0])).resolve()
            == Path(sys.executable).resolve()
            and "-u" not in command_to_run[1:2]
        ):
            command_to_run.insert(1, "-u")

        program_name = (
            Path(str(command_to_run[2])).name
            if len(command_to_run) > 2
            and command_to_run[1] == "-u"
            else (
                Path(str(command_to_run[1])).name
                if len(command_to_run) > 1
                else Path(str(command_to_run[0])).name
            )
        )

        self.write_log("")
        self.write_log("=" * 88)
        self.write_log(
            f"[RUN {run_id:03d}] START  |  {program_name}"
        )
        self.write_log(f"[RUN {run_id:03d}] {starting_message}")
        self.write_log(
            f"[RUN {run_id:03d}] Working directory: {BASE_DIR}"
        )
        self.write_log(
            f"[RUN {run_id:03d}] Command: "
            + subprocess.list2cmdline(command_to_run)
        )
        self.write_log("-" * 88)

        self.progress.start(12)

        try:
            self.run_training_btn.configure(
                state="disabled"
            )
        except (AttributeError, tk.TclError):
            pass

        thread = threading.Thread(
            target=self._command_worker,
            args=(
                command_to_run,
                run_id,
                program_name,
                on_success,
            ),
            daemon=True
        )

        thread.start()

    def _command_worker(
        self,
        command,
        run_id,
        program_name,
        on_success=None
    ):

        return_code = None

        try:
            env = os.environ.copy()

            # Force Python and many Python-based child tools to flush output
            # promptly through pipes.
            env["PYTHONUNBUFFERED"] = "1"
            env.setdefault("PYTHONIOENCODING", "utf-8")

            process = subprocess.Popen(
                command,
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW
                    if os.name == "nt"
                    else 0
                )
            )

            if process.stdout is not None:
                # Iterating over the pipe preserves the order in which merged
                # stdout/stderr lines arrive from the child process.
                for line in iter(process.stdout.readline, ""):
                    # Preserve blank lines and console indentation while only
                    # removing the newline already supplied by write_log().
                    console_line = line.rstrip("\r\n")
                    self.after(
                        0,
                        self.write_log,
                        console_line
                    )

                process.stdout.close()

            return_code = process.wait()

            self.after(
                0,
                self.write_log,
                "-" * 88
            )
            self.after(
                0,
                self.write_log,
                (
                    f"[RUN {run_id:03d}] END    |  {program_name}  |  "
                    f"exit code {return_code}"
                )
            )
            self.after(
                0,
                self.write_log,
                "=" * 88
            )

            if return_code == 0:
                self.after(
                    0,
                    messagebox.showinfo,
                    "Completed",
                    (
                        "The operation completed successfully.\n\n"
                        f"Exit code: {return_code}"
                    )
                )

                if on_success is not None:
                    self.after(
                        0,
                        on_success
                    )
            else:
                self.after(
                    0,
                    messagebox.showerror,
                    "Process Error",
                    (
                        "The process exited with a non-zero status.\n\n"
                        f"Exit code: {return_code}\n\n"
                        "See Training Progress / Program Output for the "
                        "complete console log."
                    )
                )

        except Exception as exc:
            self.after(
                0,
                self.write_log,
                "-" * 88
            )
            self.after(
                0,
                self.write_log,
                (
                    f"[RUN {run_id:03d}] GUI EXECUTION ERROR: {exc}"
                )
            )
            self.after(
                0,
                self.write_log,
                "=" * 88
            )

            self.after(
                0,
                messagebox.showerror,
                "Execution Error",
                (
                    f"{exc}\n\n"
                    "See Training Progress / Program Output for details."
                )
            )

        finally:
            self._process_running = False

            self.after(
                0,
                self.progress.stop
            )

            self.after(
                0,
                self._refresh_training_availability
            )

    # ========================================================
    # LOG
    # ========================================================

    def _build_log_section(
        self,
        parent
    ):

        frame = ttk.LabelFrame(
            parent,
            text=(
                "Training Progress / "
                "Program Output"
            ),
            style=(
                "Section."
                "TLabelframe"
            ),
            padding=5
        )

        frame.grid(
            row=5,
            column=0,
            sticky="nsew"
        )

        frame.rowconfigure(
            0,
            weight=1
        )

        frame.columnconfigure(
            0,
            weight=1
        )

        self.log = tk.Text(
            frame,
            height=7,
            font=(
                "Consolas",
                9
            ),
            wrap="none",
            undo=False
        )

        self.log.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        yscrollbar = ttk.Scrollbar(
            frame,
            orient="vertical",
            command=self.log.yview
        )

        yscrollbar.grid(
            row=0,
            column=1,
            sticky="ns"
        )

        xscrollbar = ttk.Scrollbar(
            frame,
            orient="horizontal",
            command=self.log.xview
        )

        xscrollbar.grid(
            row=1,
            column=0,
            sticky="ew"
        )

        self.log.configure(
            yscrollcommand=yscrollbar.set,
            xscrollcommand=xscrollbar.set
        )

        self.write_log(
            "SOIL MLOPS GUI ready. Program stdout/stderr will appear here."
        )

    def write_log(
        self,
        message
    ):

        self.log.insert(
            tk.END,
            str(message) + "\n"
        )

        self.log.see(
            tk.END
        )

    # ========================================================
    # FOLDER OPENING
    # ========================================================

    def open_folder(
        self,
        folder
    ):

        folder = Path(folder)

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        try:
            if os.name == "nt":

                os.startfile(
                    str(folder)
                )

            elif (
                sys.platform
                == "darwin"
            ):

                subprocess.Popen(
                    [
                        "open",
                        str(folder)
                    ]
                )

            else:

                subprocess.Popen(
                    [
                        "xdg-open",
                        str(folder)
                    ]
                )

        except Exception as exc:

            messagebox.showerror(
                "Folder Error",
                (
                    "Could not open "
                    "folder:\n\n"
                    f"{exc}"
                )
            )

    # ========================================================
    # FILE CHECK
    # ========================================================

    @staticmethod
    def check_required_file(
        filename
    ):

        filename = Path(
            filename
        )

        if filename.exists():
            return True

        messagebox.showerror(
            "Required File Missing",
            (
                "The following file "
                "does not exist:\n\n"
                f"{filename}"
            )
        )

        return False


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    args = parse_command_line()

    # If the user does not choose a project folder in the GUI, the folder
    # from which this command was launched is the active project root.
    configure_project_directory(Path.cwd())
    configure_params_file(args.params)

    app = SoilMLOpsApp()

    app.mainloop()