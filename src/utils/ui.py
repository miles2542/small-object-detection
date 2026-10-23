"""Decoupled console rendering, structured output formatting, and progress indicators."""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()


def print_header(title: str, subtitle: str | None = None) -> None:
    """Render a clean, high-density header panel."""
    content = f"[bold]{title}[/bold]"
    if subtitle:
        content += f"\n[dim]{subtitle}[/dim]"
    console.print(Panel(content, expand=False, border_style="cyan"))


def print_info(message: str) -> None:
    """Print an informational message."""
    console.print(f"[cyan]INFO[/cyan]  {message}")


def print_success(message: str) -> None:
    """Print a success confirmation."""
    console.print(f"[green]OK[/green]    {message}")


def print_warning(message: str) -> None:
    """Print a non-fatal warning."""
    console.print(f"[yellow]WARN[/yellow]  {message}")


def print_error(message: str) -> None:
    """Print an error alert."""
    console.print(f"[bold red]ERROR[/bold red] {message}")


def print_table(
    title: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    column_alignments: Sequence[str] | None = None,
) -> None:
    """Render a dense, structured data table."""
    table = Table(title=title, show_lines=False, header_style="bold cyan")
    for i, col in enumerate(columns):
        align = "left"
        if column_alignments and i < len(column_alignments):
            align = column_alignments[i]
        table.add_column(col, justify=align)  # type: ignore[arg-type]

    for row in rows:
        table.add_row(*[str(cell) for cell in row])

    console.print(table)


def create_progress() -> Progress:
    """Create a standardized progress tracker with minimal decoration."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
