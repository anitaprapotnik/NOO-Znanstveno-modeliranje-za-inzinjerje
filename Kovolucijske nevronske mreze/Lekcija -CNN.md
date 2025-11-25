
# Kaj je to konvolucijska nevronska mreža?

V osnovni obliki umetnih nevronskih mrež je vsak nevron iz ene plasti povezan z vsemi nevroni iz sosednjih plasti. Takšna ureditev je primerna, kadar med vhodnimi
podatki ni izrazite urejenosti ali medsebojnih povezav. Če pa vhodni podatki vsebujejo določeno strukturo – na primer informacijo o tem, kateri podatki so si med seboj “bližji” – 
z uporabo gosto povezanih mrež del te informacije izgubimo.

Tak primer so večdimenzionalni podatki, kot so slike ali videoposnetki. Sliko lahko računalnik predstavi kot matriko, kjer število vrstic in stolpcev ustreza številu pikslov v navpični
in vodoravni smeri. Vsak piksel v matriki predstavlja eno polje, katerega vrednost določa barvo tega piksla. Podatek o tem, kje posamezni piksel leži, je zelo pomemben – če bi uporabili 
gosto povezano mrežo, kjer so vsi vhodi povezani z vsemi vozlišči v prvi skriti plasti, bi prostorsko razporeditev pikslov izgubili.

Ta problem rešujejo konvolucijske nevronske mreže (CNN), ki upoštevajo geometrijsko obliko slike ter medsebojno oddaljenost in usmerjenost pikslov. Mreža je zasnovana tako, da se najprej 
osredotoča na podrobnosti – ozko območje okoli posameznega piksla, kjer prepoznava robove in kontraste – ter postopno napreduje k prepoznavanju večjih struktur, kot so oblike in predmeti.

Konvolucijske nevronske mreže to dosežejo z uporabo treh dodatnih vrst plasti, ki bodo opisane v nadaljevanju.

 - Konvolucijska plast (Convolutional layer)

 - Plast za združevanje (Pooling layer)

 - Sploščevalna plast (Flatten layer)

# Konvolucijska plast
Konvloucijska plast prestavlja drsene majhnega filtra dimenzij n x n po sliki. Vrednosti slike, ki jih prekriva filter se preslikajo v naslednjo plast,
tako da se pomnožijo z utežmi filtra. Filter ima dve pomembne lasnosti:

Lokalna povezanost: le nevroni, ki jih prekriva filter so povezani preslikanimi nevroni v sosednji plasti
Deljenje uteži: Ko se filter premika po sliki, v vsakem položaju uporablja enake uteži.
Delovanje konvolucijske plasti najlažje prikažemo s sliko:

Prvi del slike najprej prekrijemo s 3x3 filtrom. Uteži filtra so predstavljene v sredinskem  3x3 kvadratu, odmik pa kod številka pod njim. 

| 1 | 2 | 4 | 3 | 1 |   | 1 | 0 | 1 |   | 11 |
|---|---|---|---|---|---|---|---|---|---|----|
| 0 | 2 | 1 | 1 | 5 |   | -1 | 1 | 1 |   |    |
| 3 | 2 | 0 | 3 | 2 |   | 0  | 0 | 1 |   |    |
| 3 | 2 | 1 | 5 | 2 |   |    |   |   |   |    |
| 1 | 2 | 5 | 1 | 0 |   |    | 3 |   |   |    | 

V vozlišče v naslednji plasti se zapiše vrednost:

1*1+2*0+4*1+0*(-1)+2*1+1*1+3*0+2*0+0*1+3=11

 Nato se filter premakne po sliki za n polj na desno ali navzdol. Vrednost n imenujemo korak (stride). V primeru n=2 Naslednja koraka zgledata takole:

| 1 | 2 | 4 | 3 | 1 |   | 1 | 0 | 1 |   | 11 | 15 |
|---|---|---|---|---|---|---|---|---|---|----|----|
| 0 | 2 | 1 | 1 | 5 |   | -1 | 1 | 1 |   |    |    |
| 3 | 2 | 0 | 3 | 2 |   | 0  | 0 | 1 |   |    |    |
| 3 | 2 | 1 | 5 | 2 |   |    |   |   |   |    |    |
| 1 | 2 | 5 | 1 | 0 |   |    | 3 |   |   |    |    | 	 	 
 	 	 	 	 	 	 	 	 	 	 	 

| 1 | 2 | 4 | 3 | 1 |   | 1 | 0 | 1 |   | 11 | 15 |
|---|---|---|---|---|---|---|---|---|---|----|----|
| 0 | 2 | 1 | 1 | 5 |   | -1 | 1 | 1 |   | 4  |    |
| 3 | 2 | 0 | 3 | 2 |   | 0  | 0 | 1 |   |    |    |
| 3 | 2 | 1 | 5 | 2 |   |    |   |   |   |    |    |
| 1 | 2 | 5 | 1 | 0 |   |    | 3 |   |   |    |    |

 # Dodajanje robov
Če uporabimo konvolucijsko plast, se dimenzija naslednje plasti zmanjša. Prav tako so pri drsenju filtra po sliki robni piksli uporabljeni oziroma preslikani manj pogosto 
kot preostali del slike, kar pomeni, da so obravnavani z manjšo težo. Da to preprečimo, običajno okoli robov vhodnih podatkov dodamo dodatne piksle – temu pravimo padding. Glede na strategijo ločimo:

 - Brez paddinga / valid padding: Strategija, ki običajno povzroči, da se izhod skrči.

 - Same padding: Katerekoli metode, ki zagotavljajo, da je velikost izhoda enaka velikosti vhoda.

 - Full padding: Metode, ki zagotovijo, da je vsak vhodni element konvolucijsko obdelan enako število krat.

Glede na način dodajanja pikslov pa ločimo naslednje možnosti:

 - Zero padding: Dodajanje ničelnih vrednosti okoli robov vhoda.

 - Mirror / reflect / symmetric padding: Zrcaljenje vhodnega niza ob robu.

 - Circular padding: Ciklični vhodni niz, ki se »zavije« na nasprotni rob, podobno kot na torusu.

# Plast za združevanje

Namen plasti za zduževanje (pooling layer) je  zmanjšati količino podatkov, hkrati pa ohraniti najpomembnejše informacije. Pri konvolucijski mreži običajno uporabljamo takoimenovani lokalni pooling. 

Pri lokalnem poolingu se združujejo majhne skupine podatkov. Tipičen primer je mreža 2 × 2, ki se premika po sliki in na vsakem koraku izračuna eno vrednost. Na ta način se slika postopoma zmanjšuje,
hkrati pa ohranja najpomembnejše značilnosti.

Najpogosteje uporabljamo dve vrsti poolinga:

 - Max pooling: izbere največjo vrednost v posamezni skupini. To pomeni, da izpostavi najmočnejši signal ali najbolj izrazito značilnost v tistem delu slike.

 - Average pooling: izračuna povprečno vrednost skupine. Ta pristop bolj izravna podatke in zajame povprečno informacijo o tistem območju.

# Sploščevalna plast in gosta plast

Sploščevalna plast se uporablja proti koncu konvolucijskega modela. Njena naloga je, da podatke iz matrične oblike pretvori v enodimenzionalen vektor.

Po sploščevalni plasti običajno sledijo še gosto povezane plasti (plasti, v katerih je vsako vozlišče povezano z vsemi vozlišči naslednje plasti), ki vodijo do končnega izhoda. Ta nam – podobno kot pri navadnih gostih mrežah – poda verjetnost, da vhodna slika pripada eni izmed določenih kategorij (npr. ali je na sliki pes ali mačka).

Osnovna sestava konvolucijske nevronske mreže je torej:

 1. Izmenično si sledita konvolucijska plast in plast za združevanje
 2. Sploščevalna plast
 3. Goste plasti

 # VGG convolucijski modeli
VGG modeli so konvolucijske nevronske mreže za razpoznavanje slik, ki jih je razvila Visual Geometry Group na Univerzi v Oxfordu.

Najbolj znana modela sta VGG16 in VGG19, ki sta leta 2014 bila uvrščena med zmagovalne modele na ImageNet tekmovanju. Na tekmovanju so razvijali modele, 
ki so morali prepoznati objekte na več kot milijon slikah, razvrščenih v 1000 kategorij.

Ob zmagi na tekomvanju, se VGG modela odlikujeta tudi po uporabi preprostih gradnikov, pogostosti uporabe v praksi in globoki arhitekturi.

Arhitektura VGG modelov:

Konvolucijski del:

 - zaporedja 3×3 konvolucijskih plasti, vedno s korakom 1 in paddingom “same”,
 - vsaki 2–3 konvolucijski plasti sledi max pooling 2×2, ki prepolovi velikost slike,
 - število filtrov se postopoma povečuje: 64 → 128 → 256 → 512 → 512.

Sploščevalna plast (Flatten): Pretvori 3D tenzor v enodimenzionalni vektor.

Gosto povezane plasti:

 - dve gosti plasti po 4096 nevronov,
 - ReLU aktivacija,
 - dropout (včasih),
 - zadnja plast: softmax za klasifikacijo v 1000 kategorij (ImageNet).

Modela VGG16 in VGG19 lahko uvozimo v Keras in jih preoblikujemo za svoje potrebe. Postopek poteka takole:

 1. Odstranimo zadnjo plast in jo nadomestimo s plastjo oblikovano glede na število kategorij.
 2. Izbišemo uteži nekaj zadnjih plast.
 3. Mrežo ponovno streniramo na naših podatkih.

Tako na zelo enostaven način pridobimo učinkovito CNN mrežo za razpoznavanje slik za lastne potrebe.


 # Podatkovni nabori
Dobro je, če imamo pripravljeno obsežno bazo slik, na katerih lahko treniramo in razvijamo naše modele. Na srečo je na voljo kar nekaj podatkovnih naborov, 
ki vsebujejo veliko število slik z različnih področij.

Nekatere izmed teh naborov ponuja kar Keras oziroma TensorFlow, do drugih pa lahko dostopamo preko različnih spletnih strani.

TensorFlow Datasets (TFDS)
TFDS je knjižnica podatkovnih naborov, do katere lahko dostopamo neposredno iz Kerasa.

Knjižnica ponuja različne podatkovne nabore med najbolj znanimi pa so:

 - MNIST je podatkovni nabor črno-belih, ročno pisanih števk od 0 do 9 velikosti 28 × 28 pikslov. Gre za preprost in zelo primeren začetniški nabor. Slike lahko preprosto uvozimo v Keras s pomočjo ukaza:

from tensorflow.keras.datasets import mnist

(x_train,y_train),(y_test,y_test)=mnist.load_data()

 - CIFAR-10 in CIFAR-100 sta podatkovna nabora 32 × 32 pikslov velikih slik, razporejenih v 10 oziroma 100 kategorij. Uvozimo ju lahko iz Kerasa z ukazom:

from tensoflow.keras.datasets import cifar10

(x_train,y_train),(y_test,y_test)=cifar10.load_data()

Podatke  ostalih podatkovih naborov iz knjižnice TFDS najdete na povezavi: https://keras.io/api/datasets/

 
 - ImageNet
je ogromen podatkovni nabor, ki vsebuje več kot 14 milijonov slik in več kot 20.000 kategorij.
Zaradi licenčnih omejitev podatkovni nabor ni več prosto dostopen preko Kerasa, lahko pa ga prenesemo z uradne spletne strani (potrebna je registracija):
https://image-net.org/download.php

Podatkovni nabori na temo vozil/prometa
 - KITTI Dataset. Namenjen je učenju zaznavanja vozil in drugih objektov v prometu. Dostopen na:
https://www.cvlibs.net/datasets/kitti/.
 - Udacity Self-Driving Car Dataset vsebuje slike avtomobilov, tovornjakov, pešcev, prometnih znakov in semaforjev. Na voljo je na:
https://github.com/udacity/self-driving-car.
Vehicle Classification Dataset (VD2) – Kaggle. Različni podatkovni nabori, namenjeni klasifikaciji vozil.

# Transformerji

Na koncu še nekaj besed o transformerjih. Ti sodijo med najsodobnejše modele nevronskih mrež, ki pogosto prekašajo goste in konvolucijske modele ter se običajno uporabljajo za obdelavo in razumevanje jezika.

Njihova posebnost je t. i. plast pozornosti. To so plasti, ki samodejno zaznajo, katere besede v stavku so med seboj povezane oziroma katera beseda mora biti pozorna na katero drugo.
Na primer: v stavku »Marko kosi travo, Mateja pa kosi juho« plast pozornosti ugotovi, da se prvi »kosi« nanaša na besedo »travo«, drugi »kosi« pa na »juho«, kar omogoča pravilno interpretacijo pomena.
Za transformer modele so značilne tudi plasti večglave pozornosti, kjer se posamezne glave osredotočajo na različne naloge, npr. na slovnico, pomenske odnose ali časovni kontekst.

Transformerji so modeli, na katerih temeljijo veliki jezikovni modeli, kot je npr. ChatGPT (črka »T« v GPT pomeni transformer).

Keras omogoča tudi sestavljanje transformer modelov, vendar to presega namen tega učnega modula.

Prav tako python omogoča vključevanje velikih jezikovnih modelov kot del programa v obliki agentov s pomočjo knjižnice ANGO. 

Če se kdo želi še bolj poglobiti v raziskovanje nevrosnkih mrež s pomočjo pythona - sta knjižnica ANGO in transformerji dobra smer za naprej.  
