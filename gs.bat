@echo off
setlocal ENABLEDELAYEDEXPANSION
for /l %%r in (0,1,10) do python prognose.py -lr=%%r

pause

for /f "tokens=1,2,3,4 delims=," %%a in (complete_results.csv) do SET /A a= %%a & set /A b=%%b& set /A c=%%c
for /l %%p in (%a%, 12, 168) do (
   for /l %%l in (0,1,4) do (
       for /l %%d in (0,1,4) do (
              set "continue="
              if %%l LSS %b% if %%p EQU %a% set continue=1
              if %%d LEQ %c% if %%l EQU %b%  if %%p EQU %a%  set continue=1

              if not defined continue (
                    set "interrupted=1"
                    for /l %%i in (0,1,20) do (
                        if defined interrupted (
                            python prognose.py -p=%%p -l=%%l -d=%%d -cp=True -dp=False -mp=True
                            for /f "tokens=1,2,3,4 delims=," %%e in (complete_results.csv) do SET /A e= %%e & set /A f=%%f & set /A g=%%g
                            if !g! EQU %%d set "interrupted="
                            ::wiederhole das Trianing und überprüfe danach erneut
                            if !g! NEQ %%d set "interrupted=1" & echo interrupted
                            )
                    )
            )
        )
    )
)