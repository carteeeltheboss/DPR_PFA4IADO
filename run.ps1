$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $RootDir "DPR_MedFusionNet\venv"
$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
$AppFile = Join-Path $RootDir "DPR_WebService\app.py"

if (-not (Test-Path $ActivateScript)) {
    throw "Virtual environment not found. Run .\run_first.ps1 first."
}

& $ActivateScript

Set-Location $RootDir
python $AppFile
