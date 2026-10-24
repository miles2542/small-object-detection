$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvSod = Join-Path $ScriptDir ".venv\Scripts\sod.exe"

if (Test-Path $VenvSod) {
    & $VenvSod @args
} else {
    Write-Warning "Virtual environment not detected at .venv. Running via uv..."
    uv run sod @args
}
