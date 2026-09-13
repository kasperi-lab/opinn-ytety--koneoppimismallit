# Koneoppimismallit verkkoliikenteen luokitteluun

Tässä repositoriossa ovat opinnäytetyössä käytetyt Logistic Regression-, Random Forest- ja XGBoost-mallit. Mallit käyttävät CICIDS2017-aineiston DDoS-osatiedostoa ja UNSW-NB15-aineistoa.

## Asennus

Asenna Python 3.12. Avaa PowerShell repositorion juurikansiossa ja suorita:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Windowsissa käytetään yllä näkyvää Python-polkuja. Linuxissa ja macOS:ssä vastaava polku on `.venv/bin/python`.

## UNSW-NB15-mallien ajo

Valmiit puhdistetut aineistot ovat kansiossa `UNSW_NB15`:

- `X_train_clean.csv` ja `y_train_clean.csv` sisältävät 175 341 opetushavaintoa
- `X_test_clean.csv` ja `y_test_clean.csv` sisältävät 82 332 testihavaintoa

Siirry ensin mallien kansioon ja käynnistä haluamasi ohjelma:

```powershell
Set-Location UNSW_NB15
..\.venv\Scripts\python.exe LR-Model-UNSW_NB15.py
..\.venv\Scripts\python.exe Random_Forest_UNSW_NB15.py
..\.venv\Scripts\python.exe XGBoost-Model-UNSW_NB15.py
```

Ohjelma kysyy malliversion, parametrihaun, feature importancen ja piirteiden poiston valinnat. Vastaa jokaiseen kysymykseen `baseline` tai `parannettu` sekä `Y` tai `N` ohjelman näytössä olevan kysymyksen mukaisesti.

Parametrihaussa käytetään kolmiosaista ositettua ristiinvalidointia, F1-arvoa valintamittarina ja satunnaistilaa 42.

## Kahden lisäpiirteen poistaminen

UNSW:n lisäkokeessa poistetaan tavallisen poistolistan lisäksi `sjit` ja `dtcpb`. Tee muutos käsin tiedostossa `UNSW_NB15/Piirteiden_poisto_lista.py` lisäämällä nimet `features_to_be_removed`-listaan:

```python
features_to_be_removed = [
    "attack_cat", "id", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_ltm", "ct_dst_src_ltm", "ct_srv_dst",
    "ct_ftp_cmd", "ct_flw_http_mthd", "sjit", "dtcpb",
]
```

Aja tämän jälkeen sama parannettu malli uudelleen. Tarkista tulosteesta, että piirteiden määräksi tulee 29. Palauta nimet myöhemmin pois listasta, jos ajat tavallisen 31 piirteen version.

## CICIDS2017-mallien ajo

Palaa repositorion juurikansioon ja suorita haluamasi ohjelma:

```powershell
Set-Location ..
..\.venv\Scripts\python.exe CICIDS2017/LR-CICIDS2017_VALMIS.py
..\.venv\Scripts\python.exe CICIDS2017/RF-CICIDS2017_VALMIS.py
..\.venv\Scripts\python.exe CICIDS2017/XGBoost-CICIDS2017_VALMIS.py
```

Ohjelmat lataavat aineiston Kagglehubilla. Ensimmäinen ajo tarvitsee verkkoyhteyden ja mahdollisesti Kaggle-tunnistautumisen. CIC-aineiston jako on 80/20, jako on ositettu ja satunnaistila 42.

## Aineistojen muodostaminen uudelleen

Valmiita CSV-tiedostoja ei tarvitse muodostaa uudelleen. Jos ajat UNSW-esikäsittelyn, se korvaa nykyiset puhdistetut CSV-tiedostot:

```powershell
Set-Location UNSW_NB15
..\.venv\Scripts\python.exe Data-processing-UNSW_NB15.py
```

Tarkista tämän jälkeen havaintomäärät ja sarakkeet ennen mallien ajoa. Säilytä tarvittaessa kopio vanhoista CSV-tiedostoista.

## Tulosten tulkinta

Sekaannusmatriisin järjestys on `[[TN, FP], [FN, TP]]`. `PR-AUC` lasketaan ohjelmissa `average_precision_score`-funktiolla. Tallenna käytetty Git-commit, Python- ja kirjastoversiot sekä tieto siitä, käytettiinkö 31 vai 29 piirrettä.

Testiaineistosta laskettu permutation importance ja sen perusteella tehty piirrekarsinta ovat tutkiva lisäkoe. Testiaineisto ei silloin ole täysin riippumaton piirteiden valinnasta.

## Tiedostot

- `CICIDS2017/`: CIC-mallien ohjelmat.
- `UNSW_NB15/`: UNSW-mallien ohjelmat, esikäsittely, piirrelista ja CSV-aineistot.
- `check_data.py`: aineistojen rakenteen ja havaintomäärien tarkistus.
- `requirements.txt`: Python-riippuvuudet.

Koodin ja aineistojen jatkojakelussa noudata alkuperäisten aineistojen ehtoja. Repositorioon ei ole määritelty koodilisenssiä.
