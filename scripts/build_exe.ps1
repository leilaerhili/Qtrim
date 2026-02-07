$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (!(Test-Path $python)) {
    throw "Missing virtual environment python: $python"
}

$distDir = Join-Path $repoRoot "dist\windows"
$workDir = Join-Path $repoRoot "build\pyinstaller"
$appScriptPath = Join-Path $repoRoot "pc\app_streamlit.py"
$assetsPath = Join-Path $repoRoot "pc\assets"

New-Item -ItemType Directory -Force -Path $distDir | Out-Null
New-Item -ItemType Directory -Force -Path $workDir | Out-Null

& $python -m pip install --upgrade pip
& $python -m pip install pyinstaller

Push-Location $repoRoot
try {
    & $python -m PyInstaller `
        --noconfirm `
        --clean `
        --onefile `
        --name "QTrimDesktop" `
        --distpath $distDir `
        --workpath $workDir `
        --specpath $workDir `
        --collect-all streamlit `
        --collect-all plotly `
        --collect-all pandas `
        --collect-all qiskit `
        --collect-submodules core `
        --collect-submodules pc `
        --hidden-import uvicorn `
        --hidden-import uvicorn.logging `
        --hidden-import uvicorn.loops.auto `
        --hidden-import uvicorn.protocols.http.auto `
        --hidden-import uvicorn.protocols.websockets.auto `
        --hidden-import uvicorn.lifespan.on `
        --add-data "${appScriptPath};pc" `
        --add-data "${assetsPath};pc\assets" `
        "pc\desktop_launcher.py"
}
finally {
    Pop-Location
}

$exePath = Join-Path $distDir "QTrimDesktop.exe"
if (!(Test-Path $exePath)) {
    throw "EXE build failed. Missing artifact: $exePath"
}

Write-Host "Built EXE: $exePath"
