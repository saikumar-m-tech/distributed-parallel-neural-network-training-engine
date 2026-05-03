@echo off
echo Activating MSVC environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set TMP=C:\Temp
set TEMP=C:\Temp
cd /d A:\distributed-parallel-neural-network-training-engine
echo Building parallelnet_cpp...
pip install -e .
echo.
pause