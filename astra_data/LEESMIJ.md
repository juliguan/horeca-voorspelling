# Vijf synthetische horeca-datasets

De CSV’s bevatten synthetische kassa-orderregels voor de kalenderperiode 1 september 2023 tot en met 31 augustus 2026 (1.096 kalenderdagen). Iedere zaak heeft een eigen vraagproces, eigen standaardseed en eigen willekeurige trekkingen. De generatoren zijn ontworpen zonder informatie over jullie voorspelmodel, features of resultaten. De technische exportstructuur is gedeeld; de vraagprocessen zijn verschillend. Dit zijn beredeneerde scenario’s, geen op echte kassadata gekalibreerde steekproeven en geen garantie op modelneutraliteit.

## Uitvoeren

Vereisten: Python 3.10 of nieuwer, numpy en pandas. Er worden geen API’s aangeroepen en geen externe data ingelezen. Elk script bevat alle benodigde code en kan los van de andere bestanden worden gebruikt.

```bash
pip install numpy pandas
python buurtcafe.py
python kantoorlunchroom.py
python hotelrestaurant.py
python grandcafe.py
python avondrestaurant.py
```

Elk script schrijft standaard de gelijknamige CSV naast het script. Een bestaand uitvoerbestand wordt overschreven. Alternatief:

```bash
python buurtcafe.py --seed 20260908 --output nieuwe_trekking/buurtcafe.csv
```

Dezelfde seed geeft met dezelfde Python/numpy/pandas-omgeving dezelfde uitvoer. Andere seeds leveren nieuwe trekkingen binnen hetzelfde zaakscenario; de vijf verschillende scripts leveren verschillende scenario’s. De meegeleverde CSV’s zijn met de standaardseeds gemaakt. Geteste omgeving: numpy 2.3.5, pandas 2.2.3.

## Korte beschrijvingen zonder formules

### 1. Buurtcafé — buurtcafe.py / buurtcafe.csv

Lokaal bezoek concentreert zich in de late dagdelen en het weekend. Gesimuleerd weer blijft soms meerdere dagen hangen en beïnvloedt het terras, met grenzen aan het voordeel van warmte. Binnenbezoek, vaste sociale activiteiten en incidentele drukte reageren anders. Er zijn vaste sluitingsmomenten, een vakantieperiode, veranderende consumptiekeuzes en beperkte capaciteit.

### 2. Kantoorlunchroom — kantoorlunchroom.py / kantoorlunchroom.csv

Een smalle lunchpiek wordt gevoed door de aanwezigheid van werknemers in omliggende kantoren. Hybride werkweken, vakanties, veranderingen bij huurders en incidentele bijeenkomsten beïnvloeden aantallen en mandjes. Het weekend is vrijwel altijd gesloten. Weer is geen vraagdriver in dit scenario.

### 3. Hotel-ontbijt/restaurant — hotelrestaurant.py / hotelrestaurant.csv

Ontbijt hangt samen met een aanhoudende kamerbezetting, verblijfsduur en de verhouding tussen zakelijke en recreatieve gasten. Seizoen en meerdaagse bijeenkomsten werken door in verschillende dagdelen. Het restaurant ontvangt daarnaast gasten van buiten het hotel. Een tijdelijke beperking van de keuken raakt niet alle dagdelen; lokaal weer stuurt de omzet niet rechtstreeks. Ontbijt wordt hier als aparte kassaverkoop geregistreerd, ook voor hotelgasten.

### 4. Toeristisch grand café — grandcafe.py / grandcafe.csv

Toeristenstromen volgen het seizoen, vakantieachtige drukte en fictieve evenementen met uiteenlopende duur en omvang. Groepsbezoek, aanhoudende drukke of rustige periodes en incidentele verstoringen maken het patroon onregelmatig. Weekdagen spelen een kleinere rol dan bij een lokale lunchzaak. De beschikbare capaciteit begrenst uitschieters en de consumptiemix verandert met het bezoek.

### 5. Avondrestaurant met bar — avondrestaurant.py / avondrestaurant.csv

Dinerreserveringen, wisselende opkomst en spontane barinloop vormen verschillende vraagbronnen. Weekendbezoek en bijzondere eetgelegenheden beïnvloeden zowel het volume als de bestedingen. Alcohol en extra rondes dragen substantieel bij aan omzetvariatie. Langzaam veranderende populariteit, vakantiesluitingen en een capaciteitsgrens maken het patroon minder regelmatig.

Alle vijf scenario’s bevatten prijsaanpassingen per product, groepsafhankelijke aantallen, niet-normale toevalsvariatie en incidenteel onvolledige kassa-export. De beschrijvingen onthullen geen exacte formules of effectsterktes. De volledige scripts maken deze noodzakelijkerwijs wel inzichtelijk: laat ze bij een aparte testbeheerder als jullie de evaluatie blind willen houden.

## Gegevenscontract

Exact deze kolommen, in deze volgorde:

```text
order_id,tijdstip,dagdeel,aantal_gasten,item,categorie,aantal,stukprijs,regelbedrag
```

- CSV met komma als scheidingsteken, UTF-8, punt als decimaalteken, geen indexkolom.
- `order_id`: unieke bestelling binnen een dataset; een zaakprefix voorkomt ook overlap tussen de vijf datasets. ID’s kunnen gaten bevatten en zijn geen tijdsvolgorde.
- Elke bestelling bevat 2–5 verschillende producten/orderregels, dus binnen het gevraagde bereik van 1–5 en altijd meerdere regels. Herhaalde consumpties van hetzelfde product zijn verwerkt in `aantal`.
- `tijdstip`: `YYYY-MM-DD HH:MM:SS`, lokaal en zonder tijdzone. Alle regels van één order hebben hetzelfde tijdstip. Er worden geen transacties na middernacht aan de voorgaande dag toegewezen.
- `dagdeel`: ochtend 06:00–10:59, lunch 11:00–13:59, middag 14:00–16:59, diner 17:00–20:59, avond 21:00–23:59. Buiten de opening van de betreffende zaak ontstaan geen transacties; niet elk dagdeel komt bij elk type voor.
- `aantal_gasten`: groepsgrootte van de bestelling, herhaald op elke regel. Tel deze waarde dus eenmaal per unieke order als je gasten per bestelling wilt aggregeren. Het is geen unieke bezoekerstelling over meerdere bestellingen.
- `item` en `categorie`: herkenbare producten en productgroepen; assortimenten verschillen tussen zaken.
- `aantal`: positief geheel aantal verkochte eenheden; kan bij meerdere drankrondes groter zijn dan de groepsgrootte.
- `stukprijs` en `regelbedrag`: EUR inclusief btw met twee decimalen. De berekening gebruikt hele centen: `regelbedrag = aantal * stukprijs` zonder afrondingsverschillen op centniveau. Bij inlezen als floats kunnen gebruikelijke binaire representatieverschillen ontstaan.
- Alleen positieve verkopen; geen retouren, tips of aparte kortingsregels.

## Kalenderdagen en ontbrekende data

Elk script simuleert alle 1.096 kalenderdagen, inclusief 29 februari 2024. Een gesloten dag of volledig ontbrekende kassa-export levert geen orderregels op. Bij een gedeeltelijk ontbrekende export ontbreken hele bestellingen. Daarom hoeft de eerste of laatste transactiedatum niet gelijk te zijn aan de grens van de simulatieperiode, en staan niet alle kalenderdatums in de CSV.

Binnen het vaste schema is een echte sluitingsdag niet te onderscheiden van een dag zonder waargenomen orders door een exportstoring. Voeg bij dagaggregatie zelf de volledige kalender toe, maar vul ontbrekende dagen niet zonder expliciete aanname met nul. De CSV bevat bewust geen verborgen vraagdrivers, weerskolommen, evenementlabels of statuskolommen. Het gesimuleerde weer en de evenementen zijn geen reconstructie van werkelijk weer of echte evenementen op die datums.

## Blinde evaluatie

De meegeleverde bestanden zijn vaste testtrekkingen. Bekijk tijdens modelontwikkeling bij voorkeur alleen de gegevensbeschrijving; bewaar scripts en aanvullende seeds bij een testbeheerder. Deze scenario’s toetsen uiteenlopende omstandigheden, maar vervangen geen evaluatie op echte, later waargenomen omzet.
