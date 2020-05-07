@echo off
setlocal ENABLEDELAYEDEXPANSION
rem for /l %%p in (48, 6, 72) do for /l %%l in (0,1,3) do python prognose.py -p=%%p -l=%%l
rem for /l %%l in (0,3,24) do python prognose.py -s=%%l
for /l %%l in (24,24,168) do python prognose.py -start_hour=%%l


