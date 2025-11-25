# Uporaba naključno generiranih števil v znanosti
 
Uporaba računalniško generiranih naključnih števil je v znanosti zelo razširjena. Spodaj je navedenih nekaj značilnih primerov njihove uporabe:

 1. ***Monte Carlo metode***
    
Gre za metode, pri katerih s pomočjo velikega števila naključno generiranih števil in statistične analize rešujemo probleme, ki so sicer deterministične narave, a analitično težko rešljivi.
Na primer, z Monte Carlo metodo lahko določimo površino nepravilnega lika tako, da naključno izbiramo koordinate v omejenem območju in štejemo, kolikokrat točke padejo znotraj lika.
V nadaljevanju bomo spoznali Monte Carlo metodo za določanje števila π.

 2. ***Simulacije***
    
Z naključnimi števili lahko simuliramo in raziskujemo naravne pojave, pri katerih ima naključnost ključno vlogo — na primer jedrske razpade (kdaj in v katero smer se sprosti delec), kvantnomehanske procese, vremenske pojave ipd.
Naključna števila pogosto uporabljamo tudi v prometnem inženirstvu, kjer simuliramo realno obnašanje voznikov (hitrost, reakcijski čas in pospeški so naključno porazdeljeni) ali prometne tokove (npr. koliko vozil zapelje v križišče in v katero smer zavijejo).

 3. ***Optimizacija***

Pri optimizacijskih problemih lahko z naključnim iskanjem rešitev poiščemo optimalno kombinacijo parametrov.
Ker je »slepo« naključno iskanje neučinkovito, ga pogosto kombiniramo z bolj sistematičnimi pristopi.
Na primer: pri trenutni rešitvi izvedemo majhno naključno spremembo in jo sprejmemo le, če izboljša rezultat.

 4. ***Analiza meritev in napak***

Pri zahtevnih eksperimentih, kjer uporabljamo več merilnih instrumentov, je težko neposredno oceniti skupno napako meritve, ki je sestavljena iz prispevkov posameznih instrumentov.
S pomočjo simulacij, v katerih meritvam dodamo naključni šum, ki ustreza znanim meritvenim napakam, lahko realneje ocenimo skupno negotovost in pričakovano natančnost meritev.

# Osnovni algoritem
Osnovni algoritem, ki ga vsebujejo skoraj vsi programski jeziki, omogoča generiranje (psevdo)naključnih števil na intervalu med 0 in 1 po enakomerni porazdelitvi.
Enakomerna porazdelitev pomeni, da je verjetnost, da izžrebamo naključno število v kateremkoli podintervalu izbranega intervala, enaka za vse enako široke podintervale.
Na primer, verjetnost, da izžrebamo število na intervalu med 0,1 in 0,2, je enaka verjetnosti, da izžrebamo število na intervalu med 0,35 in 0,45.

V Pythonovi knjižnici numpy.random lahko z ukazom numpy.random.rand(N)
ustvarimo array z N naključnimi števili. Če potrebujemo le eno število, ukaz pokličemo brez argumenta: numpy.random.rand().

Ker je računalnik determinističen sistem, si ne more »izmisliti« naključnih števil v pravem pomenu besede. Algoritem za generiranje naključnih števil deluje tako, da iz neke
začetne vrednosti (t. i. semena) s kompleksnimi računskimi operacijami izračuna prvo naključno število. To število nato uporabi kot seme za naslednje, in tako naprej. Dober 
algoritem ustvari števila, ki so porazdeljena po enakomerni porazdelitvi in med seboj nekorelirana, torej brez ponavljajočih se vzorcev.

Kot seme računalnik običajno uporabi trenutni čas (npr. odčitek notranje ure v milisekundah) ali kombinacijo različnih signalov strojne opreme (tipkovnica, miška ipd.). Če želimo,
lahko seme določimo tudi ročno z ukazom: numpy.random.seed(N), kjer je N poljubno število – naše seme.

To je posebej uporabno pri testiranju in preverjanju delovanja programov, saj računalnik ob vsakem zagonu, če uporabimo isto seme, vedno ustvari enaka naključna števila. 
Ko smo s programom zadovoljni, ukaz numpy.random.seed() odstranimo – in ob vsakem novem zagonu bo program ustvaril nova, unikatna naključna števila.

# Generiranje ostalih porazdelitev
Ko enkrat znamo generirati enakomerno porazdeljena naključna števila na intervalu med 0 in 1, lahko iz njih pridobimo naključna števila drugih porazdelitev. 
Najpogosteje pri tem uporabimo inverzno transformacijsko metodo, pri kateri poiščemo inverz kumulativne porazdelitvene funkcije (CDF). Če v ta inverz vstavimo enakomerno 
porazdeljena naključna števila, dobimo naključna števila, porazdeljena po izbrani porazdelitvi.

Pythonova knjižnica numpy.random ta postopek zelo poenostavi, saj že vsebuje vgrajene funkcije za generiranje naključnih števil po najpogosteje uporabljenih porazdelitvah:

 - numpy.random.normal(povp, std, N)
generira naključna števila, porazdeljena po Gaussovi (normalni) porazdelitvi;

 - numpy.random.poisson(povp, N)
generira naključna števila, porazdeljena po Poissonovi porazdelitvi;

 - numpy.random.binomial(n, p, N)
generira naključna števila, porazdeljena po binomski porazdelitvi;

 - numpy.random.exponential(povp, N)
generira naključna števila, porazdeljena po eksponentni porazdelitvi.

Lastnosti in uporaba posameznih porazdelitev so prikazane v datoteki porazdelitve.pdf.

 ## Dodatni koristni ukazi
Knjižnica numpy.random ponuja še nekaj uporabnih ukazov za delo z naključnimi števili:

- numpy.random.randint(min, max, N)
generira N celih naključnih števil med min in max (min je vključen, max pa izključen);

 - numpy.random.shuffle(niz)
premeša vrstni red elementov v danem nizu (arrayu ali seznamu);

 - numpy.random.multivariate_normal(povp, cov, N)
generira N večrazsežnih (npr. dvo- ali večdimenzionalnih) naključnih števil, porazdeljenih po večspremenljivčni normalni porazdelitvi, 
pri čemer so posamezne spremenljivke med seboj korelirane (odvisne glede na podano kovariančno matriko cov).
