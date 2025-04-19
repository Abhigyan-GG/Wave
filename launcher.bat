@echo off
cd /d "%~dp0"

start "" py -3.10 "ws_server.py"
timeout /t 2 /nobreak >nul
start "" java -jar "gesture-control-client\build\libs\gesture-control-client-1.0-SNAPSHOT.jar"
