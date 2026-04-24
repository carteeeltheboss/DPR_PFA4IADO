$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $RootDir "DPR_MedFusionNet\venv"
$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
$MedRequirements = Join-Path $RootDir "DPR_MedFusionNet\requirements.txt"
$WebRequirements = Join-Path $RootDir "DPR_WebService\requirements.txt"
$AppFile = Join-Path $RootDir "DPR_WebService\app.py"

function Get-PythonCommand {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @{
            Command = "py"
            Args = @("-3")
        }
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{
            Command = "python"
            Args = @()
        }
    }

    throw "Python 3 is required but was not found on PATH."
}

if (-not (Test-Path $ActivateScript)) {
    $Python = Get-PythonCommand
    & $Python.Command @($Python.Args + @("-m", "venv", $VenvDir))
}

& $ActivateScript

python -m pip install --upgrade pip
python -m pip install -r $MedRequirements
python -m pip install -r $WebRequirements

Set-Location $RootDir
python $AppFile
