# Build the Windows installer locally (CI does the same in .github/workflows/release.yml).
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1
# Needs Python 3.12 and Inno Setup 6 (winget install JRSoftware.InnoSetup).
# Output: dist\OwlFluids-Setup-<version>.exe
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$py = if ($env:PYTHON) { $env:PYTHON } else { "py -3.12" }
if (-not (Test-Path .build-venv)) { Invoke-Expression "$py -m venv .build-venv" }
$vpy = ".build-venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip
& $vpy -m pip install -c packaging\constraints.txt -e ".[app,hdf5,synnax]" pyinstaller
if ($LASTEXITCODE) { throw "pip install failed" }

$version = & $vpy -c "import owlfluids; print(owlfluids.__version__)"
if ($env:OWLFLUIDS_VERSION_SUFFIX) { $version = "$version-$env:OWLFLUIDS_VERSION_SUFFIX" }
& $vpy -m owlfluids tables --out owlfluids\fluids\bundled
if ($LASTEXITCODE) { throw "building property tables failed" }
& $vpy -m PyInstaller --noconfirm --clean packaging\owlfluids.spec
if ($LASTEXITCODE) { throw "PyInstaller failed" }
& dist\OwlFluids\owlfluids.exe check lox-feed
if ($LASTEXITCODE) { throw "the frozen build does not run" }

$iscc = (Get-Command iscc -ErrorAction SilentlyContinue).Source
if (-not $iscc) {
    $iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe") |
        Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $iscc) { throw "Inno Setup 6 not found: winget install JRSoftware.InnoSetup" }
& $iscc "/DAppVersion=$version" packaging\owlfluids.iss
if ($LASTEXITCODE) { throw "Inno Setup failed" }
Write-Host "built dist\OwlFluids-Setup-$version.exe"
