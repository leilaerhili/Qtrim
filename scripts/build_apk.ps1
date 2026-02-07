$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$androidRoot = Join-Path $repoRoot "phone\android_app"
$gradleWrapper = Join-Path $androidRoot "gradlew.bat"
$distDir = Join-Path $repoRoot "dist\android"

if (!(Test-Path $gradleWrapper)) {
    throw "Missing Gradle wrapper: $gradleWrapper"
}

if ([string]::IsNullOrWhiteSpace($env:JAVA_HOME)) {
    $jdkCandidates = @(
        "C:\Program Files\Android\Android Studio\jbr",
        "C:\Program Files\Android\Android Studio\jre",
        "C:\Program Files\Android\Android Studio1\jbr",
        "C:\Program Files\Android\Android Studio1\jre"
    )
    foreach ($candidate in $jdkCandidates) {
        if (Test-Path (Join-Path $candidate "bin\java.exe")) {
            $env:JAVA_HOME = $candidate
            $env:Path = "$($candidate)\bin;$($env:Path)"
            break
        }
    }

    if ([string]::IsNullOrWhiteSpace($env:JAVA_HOME)) {
        $javaExe = Get-ChildItem "C:\Program Files\Android" -Recurse -Filter java.exe -ErrorAction SilentlyContinue |
            Select-Object -First 1 -ExpandProperty FullName
        if ($javaExe) {
            $env:JAVA_HOME = Split-Path (Split-Path $javaExe -Parent) -Parent
            $env:Path = "$($env:JAVA_HOME)\bin;$($env:Path)"
        }
    }
}

if ([string]::IsNullOrWhiteSpace($env:JAVA_HOME)) {
    throw "JAVA_HOME is not set and no local JDK was auto-detected."
}

New-Item -ItemType Directory -Force -Path $distDir | Out-Null

Push-Location $androidRoot
try {
    & $gradleWrapper clean assembleDebug assembleRelease
}
finally {
    Pop-Location
}

$debugApk = Join-Path $androidRoot "app\build\outputs\apk\debug\app-debug.apk"
$releaseUnsignedApk = Join-Path $androidRoot "app\build\outputs\apk\release\app-release-unsigned.apk"

if (Test-Path $debugApk) {
    Copy-Item -Force $debugApk (Join-Path $distDir "QTrim-debug.apk")
    Write-Host "Built APK: $(Join-Path $distDir 'QTrim-debug.apk')"
}
else {
    throw "Debug APK missing: $debugApk"
}

if (Test-Path $releaseUnsignedApk) {
    Copy-Item -Force $releaseUnsignedApk (Join-Path $distDir "QTrim-release-unsigned.apk")
    Write-Host "Built APK: $(Join-Path $distDir 'QTrim-release-unsigned.apk')"
}
