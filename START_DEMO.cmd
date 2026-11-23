@echo off
rem Double-click to open the regression demo full screen (offline, no server).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_DEMO.ps1" %*
if errorlevel 1 pause
