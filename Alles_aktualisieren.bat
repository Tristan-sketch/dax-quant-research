@echo off
rem Downloads fresh data, reruns the study, rebuilds figures, paper and website, then opens it.
setlocal
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo uv wurde nicht gefunden. Installation: https://docs.astral.sh/uv/
  pause
  exit /b 1
)
call :pull
echo Aktualisiere Daten, Analyse, Grafiken, Paper und Website. Das dauert ein paar Minuten ...
uv run daxquant all
if errorlevel 1 (
  echo.
  echo Die Aktualisierung ist fehlgeschlagen. Die Meldungen oben zeigen den Grund.
  pause
  exit /b 1
)
uv run daxquant serve
if errorlevel 1 pause
exit /b

:pull
where git >nul 2>nul || exit /b 0
git diff --quiet >nul 2>nul
if errorlevel 1 exit /b 0
git pull --ff-only --quiet >nul 2>nul && echo Neuester Stand von GitHub geladen.
exit /b 0
