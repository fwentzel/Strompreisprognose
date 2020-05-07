# Strompreisprognose
Dieses Projekt behandelt die Prognose des Day Ahead-Strompreises für das Marktbegiebt Deutschland/Luxemburg. Für die Prognosen werden Einflussfaktoren wie Wetter, Feiertage und Tageszeit berücksichtigt.

## Einführung
Es werden standardmäßig für alleverwendeten Prognosemethoden (neuronale Netze/statistische Methoden) mit der bisher optimalsten Konfiguration für die neuronalen Netze und statistischen Methoden Prognosen berechnet und geplotted. 
Über das GUI kann eingestellt werden, ob eine bestimmte Netzart trainiert werden soll*, wofür die Konfiguration für dieses Netz geändert werden sollte über den Button "Change Net Configuration". 
Über die neue Eingabemaske können nun die Konfigurationen, die zuletzt für die Netze genutzt wurden, verändert werden. Nach der Veränderung einfach speichern.
Anschließend kann gewählt werden, welche Methoden verwendet und angezeigt werden sollen. Das einzige, was zu beachten ist, ist, dass die Wahl des "decomposed" Modells (summierte Zeitreihenkomponenten) die Auswahl der Remainderkomopnente überschreibt.
Soll nur die Remainderkomponente ausgegeben werden, sollte darauf geachtet werden, dass "decomposed" nicht genutzt wird.
Außerdem kann gewählt werden, ob die Tagesmodelle der Netze für die Prognosen genutzt werden sollen und ob diese dann auch vorher trainiert weden sollen. Sie werden nur trainiert, wenn sie auch für die Prognosen genutzt werden sollen.
Zuletzt kann angegeben werden, ob eine spezifische Prognose für einen bestimmten Zeitpunkt nach Trainingsende berechnet werden soll (0-168) oder ob eine Massenprognose mit einem Durchschnitt von einer Woche durchgeührt und dargetellt werden soll. 
![GUI zur Konfiguration der Prognosemethoden und neuronalen Netze]("Abbilungen\GUI.png")
*Wird ein Netz trainiert, wird es nicht für zukünftige Verwendungen gespeichert, um zu verhindern dass das Urspüngliche Netz überschrieben wird.
## Konfiguration der Netze
Die Eingabemaske für die Konfigurationen baut sich aus den fünf Parametern `Additional Layers`, `Input Length`, `Dropout strength`, `Epochs` und `Batch size (Training)` für die jeweiligen Netze `Price complete`, `Remainder complete`,`Price day` und `Remainder day` auf.
Einfach die Parameter für das jeweilige Netz ändern und speichern. Diese Einstellungen werde dann in einer Konfigurationsdatei gespeichert, sodass sie für die nächste Verwendung aktualisiert zur Verfügung stehen.
Eine kurze Erklärung zu den einzelnen Parameter: 

-`Additional Layers` steht für die zusätzlichen Ebenen neben der Input Ebene und der Dense Ebene

-`Input Length` steht für die Länge des Inputzeitraums für dieses Netz

-`Dropout strength` steht für die Dezimalstelle der dropout Menge in den zusätzlichen Ebenen (2 steht für 0.2=20% Dropout)

-`Epochs` beschreibt, wie viele Epochen trainiert werden sollen. Da ein Early Stopping callback genutzt wird, hat die Epochenanzahl hauptsächlich Einfluss auf die Lernrate, da sie bei einem polynomischen Lernratenplan schneller fällt, wenn es weniger Epochen gibt.

-`Batch size` beschreibt die batch_size des Trainings. Nach jeden Batch werden die internen Gewichtungen des Netzes angepasst. Sie sollte eine Potenz von 2 sein.

##Ergebnisse
In der Legende wird immer der mittlere Fehler der Prognosemethode aufgeführt. 
Bei Massenprognosen (`timestep` = -1) wird in blau der mittlere Fehler der Prognosen an diesem Zeitschritt dargestellt. 
Außerdem wird in Orange ein Moving Average des mittleren Fehlers darsgestellt für eine einfachere Einschätzung der Ergebnisse.

## Daten
Die Strompreisdaten werden nach der Creative Commonons Lizens CC BY 4.0 von der Bundesnetzagentur | SMARD.de (https://www.smard.de/home/downloadcenter/download_marktdaten) heruntergeladen und unter Daten/price.csv gespeichert.

Die Wetterdaten werden vom deutschen Wetterdienst auf dem Open Data-Server https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/ bereitgestellt und "dürfen entsprechend der "Verordnung zur Festlegung der Nutzungsbestimmungen für die Bereitstellung von Geodaten des Bundes (GeoNutzV) unter Beigabe eines Quellenvermerks ohne Einschränkungen weiterverwendet werden." (https://www.dwd.de/DE/service/copyright/copyright_node.html). Sie werden unter Daten/weather.csv gespeichert
