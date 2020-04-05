@echo off
setlocal ENABLEDELAYEDEXPANSION
rem for /l %%p in (48, 6, 72) do for /l %%l in (0,1,3) do python prognose.py -p=%%p -l=%%l
for /l %%l in (0,3,24) do python prognose.py -s=%%l

