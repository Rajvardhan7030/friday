"""Interactive TUI for model-scout using Textual."""

from typing import List, Dict, Any, Optional
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable, Input, Static, Label
from textual.containers import Vertical, Horizontal
from textual.binding import Binding
from .hardware_scanner import DetailedHardwareProfile

class ModelScoutApp(App):
    """A Textual app for scouting LLM compatibility."""

    CSS = """
    Screen {
        background: $surface;
    }

    #hardware-panel {
        height: 6;
        background: $panel;
        border: solid $primary;
        margin: 1 1;
        padding: 0 1;
    }

    #search-container {
        margin: 0 1;
    }

    DataTable {
        height: 1fr;
        margin: 1 1;
        border: solid $primary-darken-2;
    }

    .best-pick {
        background: $success-darken-2;
        color: $text;
        text-style: bold;
    }

    .fit-perfect {
        color: $success;
    }

    .fit-good {
        color: $warning;
    }

    .fit-bad {
        color: $error;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("d", "show_details", "Details/Pull", show=True),
        Binding("s", "sort_score", "Sort by Score", show=True),
        Binding("n", "sort_name", "Sort by Name", show=True),
        Binding("f", "focus_search", "Filter", show=True),
    ]

    def __init__(self, profile: DetailedHardwareProfile, results: List[Dict[str, Any]]):
        super().__init__()
        self.profile = profile
        self.all_results = results
        self.filtered_results = results
        self.sort_column = "score"
        self.sort_reverse = True

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Vertical(id="hardware-panel"):
            gpu_info = f"[cyan]{self.profile.gpu_name}[/cyan] ([green]{self.profile.gpu_vram_gb:.1f} GB VRAM[/green])" if self.profile.gpu_name else "[yellow]None[/yellow]"
            yield Label(f"[b]Hardware Profile[/b]")
            yield Label(f"OS: {self.profile.os} | Arch: {self.profile.cpu_arch} | RAM: {self.profile.ram_gb:.1f} GB")
            yield Label(f"CPU: {self.profile.cpu_cores} Cores / {self.profile.cpu_threads} Threads")
            yield Label(f"GPU: {gpu_info}")

        with Horizontal(id="search-container"):
            yield Input(placeholder="Search models or tags...", id="search-input")

        yield DataTable()
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Model", "Params", "Fit", "Score", "Est. Tok/s", "Quant", "Tags")
        table.cursor_type = "row"
        self.update_table()

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        column_map = {
            0: "name",
            3: "score"
        }
        col_idx = event.column_index
        if col_idx in column_map:
            new_sort = column_map[col_idx]
            if self.sort_column == new_sort:
                self.sort_reverse = not self.sort_reverse
            else:
                self.sort_column = new_sort
                self.sort_reverse = True if new_sort == "score" else False
            self.update_table()

    def update_table(self) -> None:
        table = self.query_one(DataTable)
        table.clear()
        
        # Sort results
        if self.sort_column == "score":
            self.filtered_results.sort(key=lambda x: x["compat"]["score"], reverse=self.sort_reverse)
        elif self.sort_column == "name":
            self.filtered_results.sort(key=lambda x: x["name"], reverse=not self.sort_reverse)
        
        # Find best pick (highest score)
        best_score = -1
        best_index = -1
        for i, m in enumerate(self.filtered_results):
            if m["compat"]["score"] > best_score:
                best_score = m["compat"]["score"]
                best_index = i

        for i, m in enumerate(self.filtered_results):
            compat = m["compat"]
            fit_class = "fit-perfect" if compat["fit"] == "Perfect" else "fit-good" if compat["fit"] == "Good" else "fit-bad"
            
            if i == best_index and compat["score"] > 0:
                name_text = f"[b]⭐ {m['name']}[/b]"
            else:
                name_text = m["name"]

            table.add_row(
                name_text,
                m["params"],
                f"[{fit_class}]{compat['fit']}[/]",
                str(compat["score"]),
                f"{compat['tok_s']}",
                compat["best_quant"],
                ", ".join(m["tags"]),
                key=str(i)
            )

    def on_input_changed(self, event: Input.Changed) -> None:
        search_term = event.value.lower()
        if not search_term:
            self.filtered_results = self.all_results
        else:
            self.filtered_results = [
                m for m in self.all_results 
                if search_term in m["name"].lower() or any(search_term in t.lower() for t in m["tags"])
            ]
        self.update_table()

    def action_sort_score(self) -> None:
        if self.sort_column == "score":
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = "score"
            self.sort_reverse = True
        self.update_table()

    def action_sort_name(self) -> None:
        if self.sort_column == "name":
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = "name"
            self.sort_reverse = False
        self.update_table()

    def action_focus_search(self) -> None:
        self.query_one("#search-input").focus()

    def action_show_details(self) -> None:
        table = self.query_one(DataTable)
        cursor_row = table.cursor_row
        if cursor_row is not None:
            # Use row key for stable lookup
            row_key, _ = table.coordinate_to_cell_key((cursor_row, 0))
            index = int(row_key.value)
            model = self.filtered_results[index]
            
            detail_msg = f"Model: {model['name']}\n"
            if model.get("ollama_name"):
                detail_msg += f"Ollama: [bold]ollama pull {model['ollama_name']}[/bold]\n"
            
            # GGUF Heuristic
            clean_name = model["name"].replace(" ", "-").lower()
            detail_msg += f"GGUF: https://huggingface.co/models?search={clean_name}-GGUF"
            
            self.notify(detail_msg, title="Model Details", timeout=10)

async def run_tui(profile: DetailedHardwareProfile, results: List[Dict[str, Any]]):
    app = ModelScoutApp(profile, results)
    await app.run_async()
