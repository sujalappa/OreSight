@echo off
REM OreSight web app: double-click to start. The browser opens by itself once the app is ready
REM (the address is printed below). Close this window to stop the app.
cd /d "%~dp0geoscan"
"C:\Users\sujal\anaconda3\envs\eka_care\python.exe" -W ignore -m geoscan.webapp
echo.
echo OreSight has stopped. If there is an error above, send it to the developer.
pause
