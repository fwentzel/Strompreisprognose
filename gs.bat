@echo off
setlocal ENABLEDELAYEDEXPANSION
rem for /l %%p in (48, 6, 72) do for /l %%l in (0,1,3) do python prognose.py -p=%%p -l=%%l
for /l %%l in (0,6,18) do python prognose.py -s=%%l
rem pause

for /f "tokens=1,2,3,4 delims=," %%a in (Results/complete_results.csv) do SET /A z= %%z &SET /A a= %%a & set /A b=%%b& set /A c=%%c
for /l %%p in (48, 24, 168) do (
   for /l %%l in (0,1,10) do (
       rem for /l %%d in (0,1,4) do (
              set "continue="
              if %%l LSS %b% if %%p EQU %a% set continue=1
              if %%d LEQ %c% if %%l EQU %b%  if %%p EQU %a%  set continue=1

              if not defined continue (
              rem python prognose.py -p=%%p -lr=%%l -d=%%d -cp=True -dp=True -mp=True
              rem python prognose.py -p=%%p -lr=%%l  -cp=True -dp=False
                    rem set "interrupted=1"
                    rem for /l %%i in (0,1,20) do (
                    rem    if defined interrupted (
                    rem        python prognose.py -p=%%p -l=%%l -d=%%d -cp=True -dp=False -mp=True
                    rem        for /f "tokens=1,2,3,4 delims=," %%e in (complete_results.csv) do SET /A e= %%e & set /A f=%%f & set /A g=%%g
                    rem        if !g! EQU %%d set "interrupted="
                    rem        ::wiederhole das Trianing und überprüfe danach erneut
                    rem        if !g! NEQ %%d set "interrupted=1" & echo interrupted
                        )
                    )
            )
       rem )
    )
)