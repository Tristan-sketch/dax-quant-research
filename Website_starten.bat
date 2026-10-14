@echo off
rem Opens the research website in the browser. Double-click to run.
setlocal
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo uv wurde nicht gefunden. Installation: https://docs.astral.sh/uv/
  pause
  exit /b 1
)
call :pull
uv run daxquant serve
if errorlevel 1 pause
exit /b

:pull
rem Fetch the latest monthly update from GitHub, but only if nothing was changed locally.
where git >nul 2>nul || exit /b 0
git diff --quiet >nul 2>nul
if errorlevel 1 (
  echo Lokale Aenderungen vorhanden - der neueste Stand von GitHub wird nicht geladen.
  exit /b 0
)
git pull --ff-only --quiet >nul 2>nul && echo Neuester Stand von GitHub geladen.
exit /b 0
