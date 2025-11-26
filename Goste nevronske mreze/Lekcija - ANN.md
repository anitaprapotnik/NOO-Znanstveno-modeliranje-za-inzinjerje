
# Umetna inteligenca in nevronske mreže
Umetna inteligenca je zelo širok pojem. Evropska unija jo opredeljuje kot »sposobnost stroja, da posnema človeške sposobnosti, kot so logično razmišljanje, učenje, načrtovanje in ustvarjalnost.«

Leta 1956 je umetna inteligenca z ustanovitvijo samostojnega raziskovalnega področja postala del znanstvene sfere.

Pojem umetne inteligence zajema najrazličnejše strojne funkcije in algoritme, ki med seboj niso nujno povezani.

Velik preboj na področju umetne inteligence se je zgodil v osemdesetih letih z razvojem umetnih ***nevronskih mrež***. Gre za algoritme, ki računalniku omogočajo strojno učenje iz velike količine podatkov.

Poznamo več vrst nevronskih mrež, ki jih lahko razvrščamo po različnih merilih – na primer glede na način učenja ali vrsto podatkov, ki jih obdelujejo.

Posebna podzvrst nevronskih mrež, prilagojena razpoznavanju in ustvarjanju besedila, so tako imenovani ***veliki jezikovni modeli*** (large language models), med katerimi je najbolj znan ChatGPT.

Čeprav danes večina ljudi ob omembi umetne inteligence najprej pomisli na ChatGPT ali drug veliki jezikovni model, so ti v resnici le ena izmed podkategorij nevronskih mrež, te pa spadajo le med eno izmed podkategorij umetne inteligence.

V nadaljevanju bomo spoznali princip delovanja nevronskih mrež in se naučili, kako v Pythonovi knjižnici Keras najprej sestaviti osnovno (gosto) nevronsko mrežo, nato pa še nevronsko mrežo za razpoznavanje slik.

# Kako delujejo nevronske mreže?
Nevronska mreža je računalniški algoritem za strojno učenje. Kot že samo ime pove, je bila inspiracija za razvoj nevronskih mrež povzeta po delovanju živčnih celic (nevronov) v človeških možganih.

Tipičen problem, ki ga nevronske mreže rešujejo, lahko ponazorimo na naslednjem primeru:

Predstavljajmo si tabelo s petimi stolpci in velikim številom vrstic.
V prvih treh stolpcih (A, B in C) so vneseni neodvisni podatki, v zadnjih dveh stolpcih (D in E) pa podatki, ki so odvisni od vrednosti stolpcev A, B in C.

Osnovna ideja delovanja nevronskih mrež je v tem, da računalniku podamo takšno tabelo in mu omogočimo, da se iz podatkov nauči, kako sta stolpca D in E odvisna od stolpcev A, B in C.
Ko se mreža tega nauči, lahko od nje pričakujemo, da bo ob novih vrednostih stolpcev A, B in C (ki jih prej ni videla) znala napovedati pripadajoče vrednosti stolpcev D in E.

Število neodvisnih in odvisnih stolpcev je seveda poljubno in je odvisno od konkretnega problema.

Nekaj primerov uporabe:

V neodvisne stolpce lahko zapišemo osnovne zdravstvene podatke osebe (na primer višino, težo, krvni tlak, podatke o krvni sliki, ali je oseba kadilec ali ne …), v odvisne stolpce pa, ali je 
posameznik v življenju zbolel za določeno boleznijo. Računalnik se lahko na podlagi teh podatkov nauči napovedovati verjetnost, da bo oseba v prihodnosti zbolela za določeno boleznijo.

Drug primer je področje prometne varnosti.
V neodvisne stolpce lahko zapišemo podatke o cesti, kot so gostota prometa, vidljivost, prisotnost in vrsta signalizacije ter geometrijske lastnosti ceste, v odvisni stolpec pa število prometnih 
nesreč na določenem križišču. Računalnik se lahko na podlagi teh podatkov nauči oceniti nevarnost križišča, torej verjetnost, da na njem pride do nesreče.

# Arhitektura nevronske mreže
Arhitekturo nevronske mreže sestavljajo vozlišča ali nevroni (ang. nodes / neurons), ki so na sliki prikazani s krogi. Vozlišča so razporejena v stolpce, ki jih imenujemo plasti (ang. layers).
Prvo plast imenujemo vhodna plast (input layer), zadnjo izhodna plast (output layer), vmesne pa skrite plasti (hidden layers). 
Vsaka nevronska mreža ima torej vhodno plast, izhodno plast ter eno ali več skritih plasti.

Vozlišča so med seboj povezana s povezavami. Pri osnovni (gosti) arhitekturi je vsako vozlišče v posamezni plasti povezano z vsemi vozlišči v sosednjih plasteh, kot je prikazano na zgornjih 
slikah. Vsaki povezavi je pripisana številčna vrednost, imenovana utež (ang. weight, na sliki označena z w). Vsem vozliščem, razen tistim v vhodni plasti, pripišemo še številko, imenovano odmik ali 
pristranskost (ang. bias, na sliki označeni z b). Uteži in odmiki so na začetku določeni kot naključno izžrebane vrednosti med 0 in 1.

Podatki iz tabele skozi nevronsko mrežo prehajajo vrstico za vrstico.
Kolikor neodvisnih stolpcev ima tabela, toliko vozlišč mora imeti vhodna plast, saj vsako vozlišče predstavlja podatek enega stolpca.

# Potovanje podatkov skozi nevrosnko mrežo

Podatki iz vhodne plasti nato potujejo po povezavah do vozlišč v naslednji plasti. Vsaka povezava pomeni množenje podatka z ustrezno utežjo. V vsakem vozlišču se vsi dohodni podatki seštejejo,
vsoti pa se prišteje še vrednost odmika (ang. bias) tega vozlišča.

Izhodna plast predstavlja podatke, ki jih želimo napovedati. V izhodni plasti mora biti torej toliko vozlišč, kolikor je izhodnih spremenljivk — v našem primeru dve vozlišči, ki predstavljata stolpca D in E.

Po prvem prehodu podatkov bodo vrednosti na izhodu seveda popolnoma različne od pričakovanih. Ideja učenja je v tem, da z zaporednim prilagajanjem uteži in odmikov mreža postopoma izboljšuje svoje napovedi, 
tako da se izhodni podatki čim bolj približajo pričakovanim vrednostim.

***Pomembno:***
Nevronska mreža mora imeti:

 - v vhodni plasti toliko vozlišč, kolikor je vhodnih (neodvisnih) podatkov,

 - v izhodni plasti toliko vozlišč, kolikor je izhodnih (odvisnih) podatkov, ki jih želimo napovedati.

Število skritih plasti in vozlišč v posamezni skriti plasti pa je poljubno in se določi glede na zahtevnost problema.

 # Aktivacijska funkcija
Prej opisani postopek ima eno pomembno pomanjkljivost, ki bi jo matematiki hitro opazili: linearna kombinacija dveh linearnih kombinacij je namreč ponovno linearna kombinacija.

Na primeru bomo razložili, kaj to pomeni.
Predpostavimo, da imamo nevronsko mrežo z naslednjo arhitekturo: v vhodni plasti sta dve vozlišči, v skriti plasti prav tako dve vozlišči, 
v izhodni plasti pa eno vozlišče. Uteži med prvo in drugo plastjo označimo z $w_1$, uteži med drugo in tretjo plastjo pa z $w_2$.

Vrednosti vozlišč v skriti plasti sta torej:

$a^1=w^1_{1,1}I_1+w^2_{1,2}I_2+b_1$   in $a^2=w^1_{2,1}I_1+w^1_{2,2}I_2+b_2$.

Vrednost v izhodnem vozlišču pa je torej:

$o=w^2_{1,1}a^1+w^2_{1,2}a_2+b_0=w^2_{1,1}(w^1_{1,1}I_1+w^1_{1,2}I_2+b_1)+w^2_{1,2}(w^1_{2,1}I_1+w^1_{2,2}I_2+b_2)=$

$(w^2_{1,1}w^1_{1,1}+w^2_{1,2}w^1_{2,1})I_1+(w^2_{1,1}w^1_{1,2}+w^2_{1,2}w^1_{2,2})I_2+(w^2_{1,1}b_1+w^2_{1,2}b_2+b_0)=w_1I_1+w_2I_2+b$

Vidimo torej, da je skrita plast brez pomena, saj bi z ustrezno izbiro uteži enak rezultat dobili že, če bi vhodno plast neposredno povezali z izhodno.

Da to preprečimo, na vrednosti v vsakem vozlišču uporabimo nelinearno funkcijo, ki jo imenujemo aktivacijska funkcija. Najbolj pogosto uporabljane aktivacijske funkcije so zbrane v priloženi datoteki tabele.pdf

 # Priprava podatkov
Čeprav nevronske mreže lahko obdelujejo različne vrste podatkov, kot so slike, videoposnetki in besedilo, se bomo tukaj osredotočili le na pripravo preprostih številskih podatkov in kategorij.

Če kot vhod v nevronsko mrežo uporabljamo zvezne podatke, jih je najprej smiselno normalizirati. To pomeni, da podatke preslikamo v interval med 0 in 1 oziroma med –1 in 1, da imajo vsi vhodni 
podatki primerljivo velikost in da se izognemo numeričnim težavam pri učenju modela.

Če pa imamo opravka z diskretnimi podatki – kategorijami (na primer, želimo razlikovati med psom in mačko, prepoznati barvo las ali ugotoviti, ali bo oseba zbolela ali ne) – imamo dve možnosti, 
kako jih pretvoriti v števila.

Prva možnost je uporaba celih števil, kjer vsako kategorijo označimo z ustrezno celo številko, na primer:

 rdeča → 1, črna → 2, rumena → 3.
Čeprav je ta pristop enostaven, ima resno pomanjkljivost – umetno nakazuje zaporedje ali velikostni odnos, ki v kategorijah v resnici ne obstaja (npr. da je rumena “več” kot črna, ali črna “več” kot rdeča).

Da se temu izognemo – in s tem tudi težavam pri učenju modela – uporabljamo t. i. “one-hot” notacijo. Pri tem vsako kategorijo predstavimo z vektorjem dolžine N, kjer je N število kategorij.
Če je na primer možen rezultat rdeča, modra ali rumena, potem:

rdeča → (1, 0, 0)

modra → (0, 1, 0)

rumena → (0, 0, 1)

Cela števila pa uporabljamo le takrat, kadar med podatki res obstaja vrstni red ali hierarhija (npr. končana osnovna, srednja ali visoka šola).

# Ocena napake - stroškovna funkcija
V naslednjem koraku mora model nevronske mreže prilagoditi uteži in odmike, da se izhod čim bolje prilagodi pričakovanim vrednostim. Da bi to dosegli, potrebujemo najprej mero napake - 
tako imenovano stroškovno funkcijo, ki je odvisna od vrste podatkov, s katerimi delamo. Poglejmo, katere napake uporabljamo v primeru zveznih podatkov in v primeru kategorialnih podatkov.
Zvezni podatki

V primeru zveznih podatkov uporabljamo varianco oziroma povprečno kvadratno napako.

Enačba:

$𝑀𝑆𝐸=\frac{1}{𝑁}\sum^𝑁_{𝑖=0}(𝑦_𝑖−\hat{𝑦}_𝑖)^2$,

kjer je $𝑦_𝑖$ dejanska vrednost, $\hat{𝑦}_𝑖$ napoved modela in N število vzorcev. 

Opomba: varianca je občutljiva na izstopajoče vrednosti. Če želimo zmanjšati to občutljivost, lahko izberemo drug način ocenjevanja napake.
Klasifikacija

V primeru klasifikacije, ko je rezultat podan z one-hot zapisom, uporabljamo križno entropijo (ang. cross entropy).

Enačba:

$𝐻=−\sum^N_{𝑖=1} \sum^𝐶_{𝑐=1} 𝑦_{𝑖,𝑐} \log(\hat 𝑦_{𝑖,𝑐})$,

kjer je  N število vzorcev, C število razredov, $𝑦_{𝑖,𝑐}$ dejanska vrednost in $\hat 𝑦_{𝑖,𝑐}$ napoved modela.

Primer:
Recimo, da je pravi razred druga kategorija — v one-hot zapisu torej $𝑦=[0,1,0]$.

Če model napove verjetnosti razredov $\hat 𝑦 =[0.7,0.2,0.1]$, ima križna entropija vrednost:

$𝐻=−[0 \cdot 𝑙𝑜𝑔 (0.7)+1 \cdot 𝑙𝑜𝑔 (0.2)+0\cdot 𝑙𝑜𝑔 (0.1)]=− 𝑙𝑜𝑔 (0.2)=1.609$

Če pa model napove verjetnosti $\hat{y} =[0.1,0.8,0.1]$, dobimo:

$𝐻=−[0\cdot 𝑙𝑜𝑔 (0.1)+1\cdot 𝑙𝑜𝑔 (0.8)+0\cdot 𝑙𝑜𝑔 (0.1)]=− 𝑙𝑜𝑔 (0.8)=0.223$

Večja kot je verjetnost, ki jo model pripiše pravemu razredu, manjša je vrednost križne entropije.

Druga možnost je uporaba redke križne entropije, ki je matematično enaka križni entropiji, vendar preverja le verjetnost pravega razreda — torej ni potrebno uporabiti celotnega one-hot zapisa.

Enačba se poenostavi v:  $𝐻=− 𝑙𝑜𝑔 (\hat y)$.

V prvem primeru, ko je napoved $\hat 𝑦 =[0.7,0.2,0.1]$, preverjamo samo verjetnost za pravi razred (0.2) in dobimo:

$𝐻=− 𝑙𝑜𝑔 (0.2)=1.609$,

Katero možnost — križno entropijo ali redko križno entropijo — bomo uporabili v programu, je odvisno od načina, kako imamo podane (zapisane) prave oziroma pričakovane vrednosti kategorij.

Podrobnejši seznam stroškovnih funkcij je podan v priloženi datoteki tabele.pdf.


# Proces učenja - optimizacija
Ko enkrat določimo oceno napake (t. i. stroškovno funkcijo), lahko pristopimo k optimizaciji. Osnovna ideja optimizacijskega procesa je poiskati minimum napake. To storimo tako, da izračunamo (parcialne) 
odvode napake po utežeh in odmikih ter tako dobimo smer, v katero se moramo premakniti, da se približamo minimumu.



Velikost koraka, s katerim se premikamo v tej smeri, imenujemo hitrost učenja (ang. learning rate). Ta mora biti skrbno izbrana: 
premajhen korak pomeni počasno učenje mreže, prevelik korak pa lahko povzroči preskakovanje minimuma in s tem nestabilno učenje.

Opisano metodo imenujemo gradientni spust (GD – gradient descent). Ker je v svoji osnovni obliki razmeroma počasna, v praksi pogosteje uporabljamo njene izboljšane različice.

- Metoda naključnega gradientega spusta (SGD, ang. Stochastic Gradient Descent): Namesto da bi uteži prilagodili po prehodu vseh podatkov, jih posodabljamo po vsakem posameznem vzorcu. Metoda omogoča hitro učenje ij no uporabljamo kadar 
imamo zelo veliko podatkov ali želimo hitro približno rešitev.

 - Mini-batch Gradient Descent: Mešanica med SGD in GD - namesto enega vzorca ali vseh podatkov izbere vmesno pot in uteži posodablja po prehodu skupka vzorcev (takoimenovanega batch-a). 
Združuje prednosti obeh pristopov – večjo stabilnost posodabljanja kot SGD in hitrejše učenje kot klasični gradientni spust.

 - SGD z zagonom (ang. GSD with mometum). Za smer premika kombinira smer trenutnega odvoda in prejšnjih odvodov - torej nekako ohranja informacijo prejšnjih odvodov. Hitreje premikamo skozi dolge,
ravne doline in zmanjšamo nihanje v bližini minimuma. Uporablja se pri kompleksnih modelih, kjer je funkcija napake zelo valovita.

 - AdaGrad prilagaja velikost koraka za vsako utež posebej — pogoste spremembe dobijo manjše korake, redke pa večje. Dobro deluje pri redkih podatkih (npr. pri obdelavi besedil), vendar se koraki sčasoma preveč
zmanjšajo, kar lahko upočasni učenje.

 - RMSProp je AdaGrad z momentom: ne premikamo se v smeri zadnejga gradienta ampak v smeri povprečne vrednosti nekaj zaporedih gradientov (t.i. tekoče povprečje ali moving average).
Deluje zelo dobro pri nepostojnih problemih, kot so nevronske mreže, kjer se gradienti s časom spreminjajo.
 
 - AdaDelta je izboljšana RMSProp metoda, ki na pameten način določi začetni korak z uporabo relativnih sprememb. S tem omogoča stabilno učenje brez natančnega nastavljanja parametrov.
 
  - Adam (Adaptive Moment Estimation) – izračuna povprečje gradientov in njihovih kvadratov, kar omogoča hitrejše in stabilnejše učenje. Je ena najpogosteje uporabljenih metod danes, saj dobro deluje v večini
primerov brez veliko prilagajanja.

Natančnejši opisi optimizacijskih funkcij so zbrani v tabele.pdf.

 # Učenje in validacija
Pri procesu učenja nevrosnka mreža prejema podatke in glede na izbrano metodo optimizacije prilagaja uteži. Ves čas spremljamo napako (stroškovno funkcijo). Ko opazimo, da napaka doseže plato in se ne zmanjšuje, 
lahko sklepamo, da se je model naučil.

***Parametri procesa učenja: ***

 - Število epoh učenja – določa, kolikokrat model preide skozi celoten nabor podatkov (ena epoha pomeni en prehod podatkov). Običajno ni dovolj, da model podatke "vidi" le enkrat; za uspešno učenje jih
 moramo večkrat prevleči skozi mrežo. Po vsakem prehodu je priporočljivo podatke naključno premešati.

 - Batch size – velikost vzorca, ki ga model obravnava "naenkrat". To je število vzorcev, ki gre skozi model, preden se uteži ponovno prilagodijo.

 - Metrika – način, s katerim merimo natančnost modela. Pogosto uporabimo isto metodo kot za stroškovno funkcijo, ni pa nujno. Razlika je v tem, da se stroškovna funkcija uporablja v procesu učenja,
 medtem ko metrika le ocenjuje natančnost modela.

Po končanem učenju je obvezna validacija naučenega modela. To pomeni, da podatke, ki jih model vidi prvič (na katerih se ni učil), spustimo skozi mrežo. Za validacijo je običajno na začetku del podatkov, 
namenjenih učenju, ločen na stran (približno 10 % podatkov). Po prehodu teh podatkov ocenimo natančnost z izbrano metriko.

Včasih bo natančnost pri validaciji nižja kot med učenjem. Razlog je, da se nevrosnka mreža lahko bolj "napamet nauči" rezultate, namesto da bi pogruntala dejanske povezave med podatki. Temu pojavu pravimo overfitting. 
Problem rešujemo tako, da prilagodimo arhitekturo (število plasti, število vozlišč) in uporabimo funkcijo dropout, ki po prehodu določi del podatkov kot nepomemben.

Natančnost natrenirane mreže skoraj nikoli ni 100 %. Naš cilj je, da se ji čim bolj približamo. To dosežemo s prilagajanjem arhitekture. Pravega pravila ni — običajno moramo preizkusiti več možnosti in oceniti, 
katera prinese najboljše rezultate.

Natančnejši seznam možnih metrik je zbran v tabele.pdf.
