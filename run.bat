@echo off
call .venv\Scripts\activate.bat
python lichtfeld_node.py --lfs "C:\tools\LichtFeld-Studio\bin\LichtFeld-Studio.exe" %*
