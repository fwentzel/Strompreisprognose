@echo off
setlocal ENABLEDELAYEDEXPANSION
set newG=-1
for /f "tokens=1,2,3,4 delims=," %%a in (complete_results.csv) do SET /A a= %%a & set /A b=%%b& set /A c=%%c
:interrupted
for /l %%p in (%a%, 12, 168) do (
   for /l %%l in (0,1,5) do (
       for /l %%d in (0,1,5) do (
              set "continue="
              if %%l LSS %b% if %%p EQU %a% set continue=1
              if %%d LEQ %c% if %%l EQU %b%  if %%p EQU %a%  set continue=1

              if not defined continue (
                python prognose.py -p=%%p -l=%%l -d=%%d
                :interrupted
                for /f "tokens=1,2,3,4 delims=," %%e in (complete_results.csv) do SET /A e= %%e & set /A f=%%f & set /A g=%%g

                if !g! EQU %%d echo alles gut
                ::wiederhole das Trianing und überprüfe danach erneut
                if !g! NEQ %%d python prognose.py -p=%%p -l=%%l -d=%%d & goto interrupted

            )
        )
    )
)