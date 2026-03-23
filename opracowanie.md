# Projekt OSiOSN

## Etap 1

Celem projektu jest stworzenie modelu pełniącego rolę klasyfikatora śmieci. Model ma być częścią większego
systemu do inteligentnego sortowania odpadów. Jego zadaniem będzie analizowanie zdjęć przedmiotów wrzucanych do pojemnika i przypisywanie obiektu do klasy śmieci nadających się do recyklingu lub nie oraz elektrośmieci.

Błędy ludzkie przy segregacji prowadzą do zanieczyszczenia części odpadów co czyni je niezdatnymi do recyklingu. Inteligentne kosze na śmieci mają za zadanie zredukować wpływ tego problemu u źródła, obniżając koszty sortowania.

Platforma uruchomieniowa modelu jest prostym urządzeniem wbudowanych o ograniczonych zasobach, zatem głównym celem projektu jest zmniejszenie rozmiaru modelu oraz skrócenie czasu wnioskowania przy utrzymaniu akceptowalnej dokładności.

## Etap 2

Wybrano zbiór CIFAR-100. Dane zawierają klasy odpowiadające powyższym założeniom.
<!-- - uzasadnienie wyboru danych i modelu, -->
<!-- scharakteryzowanie oryginalnego modelu: rozmiar, liczba parametrów, czas wnioskowania, 
czas treningu pojedynczej epoki, wymagania pamięciowe (samodzielnie wykonać test uczenia 
jednej epoki na podzbiorze minibach o zadanym rozmiarze – zaraportowanie zajętości pamięci i 
czasu treningu), -->
<!-- krótkie porównanie z innymi dostępnymi zbiorami, podobnymi pod względem dziedziny 
(nie   ma   konieczności   uruchamiania   treningu,   testu.   Odwołać   można   się   do   porównań 
dostępnych w publikacjach) -->

## Etap 3

Wprowadzono nowe klasy
jednej z trzech zdefiniowanych klas:

- Recykling
- Bio
- Elektrośmieci

Przygotowanie danych obejmowało scalenie klas zgodnie z tabelą

łączone klasy | nowa klasa
--------------|-----------
bottle, bowl, can, cup, plate | recyclable
apple, mushroom, orange, pear, sweet_pepper | bio
computer_keyboard, clock, telephone, television | electrical_waste

Procedura zaszumiania etykiet:
- losowe podmienianie etykiet z $\mathcal{P}(y^{gt} \ne y^{label}) = 0.5$
- szum gaussowski z $\sigma=0.03$
- zamiana obrazka na zdjęcie psa z $\mathcal{P}(x="dog" / x^{gt})=0.1$



<!-- Dokumentacja powinna zawierać opisane:
- dane dla nowego zadania - lista rozpoznawanych klas, z krótkim uzasadnieniem, dlaczego 
określony na początku cel wymaga takich klas,
- sposób przygotowania danych: jakie klasy łączono, usuwano, zmieniono nazwy, dodano 
ręcznie, jakie nowe próbki nowych klas dostarczono i opisano samodzielne, jak prowadzono 
procedurę zaszumiania etykiet,
-  uwaga:  wszystkie operacje realizowane w sposób losowy (np. szum) powinny w kodzie 
wykorzystywać ręczne zadawanie ziarna dla generatora liczb pseudo-losowych, aby zapewnić 
powtarzalność/replikowalność wyników.
- uwaga:  w tym etapie można wykorzystać wiedzę zdobytą podczas realizacji przedmiotu 
„Zaawansowane   przygotowanie   danych   w   uczeniu   maszynowym”,   jednak   każdy   etap 
odpowiednio udokumentować, bez stosowania „skrótów myślowych”, w sposób taki, aby nie 
było potrzeby odwoływania się do materiałów z tego przedmiotu.
Pomiary modelu odniesienia (etap 4)
Proces douczania w tym etapie należy wykonać dla oryginalnego modelu. Poprzez takie 
douczenie bez optymalizacji uzyskane zostaną dane, które wykorzystać należy w dalszych 
etapach pracy do oceny modelu optymalizowanego.
4. Douczenie modelu odniesienia (baseline)
Typowo w prostych podejściach, które nie mają na celu optymalizacji a zapewnić mają 
dostosowanie do nowego zagadnienia (ang. Transfer learning, transfer wiedzy), stosuje się 
douczanie oryginalnego modelu nowymi danymi, zwykle bez zmiany jego architektury. Adaptacja  -->
