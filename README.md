# Empilement optimal

Version minimale et figée de l'optimiseur d'empilement.

Le script `empilement.py` cherche la plus petite taille de conteneur permettant
de placer `n` objets unitaires, dans quatre configurations :

| Mode | Contenu |
|---|---|
| `1` / `sq_in_sq` | carrés dans un carré |
| `2` / `circ_in_circ` | cercles dans un cercle |
| `3` / `circ_in_sq` | cercles dans un carré |
| `4` / `sq_in_circ` | carrés dans un cercle |

## Dépendances

- Python 3
- `numpy`
- `matplotlib`
- `scipy`
- `numba`
- `tqdm`

Aucun framework de deep learning externe n'est utilisé.

Les parties coûteuses sont compilées avec numba (`cache=True`) : entraînement
Deep-Set/Adam, polissage et ajustement des contacts. Le repli numpy reste
disponible si numba est absent.

## Utilisation

```bash
python3 empilement.py -n 11
python3 empilement.py -n 17 --pack 1
python3 empilement.py -n 20 --pack 3 --time-budget 20
python3 empilement.py --range 10-50
python3 empilement.py --pack 1 --gradcheck
```

Le mode par défaut est `1` (`sq_in_sq`).

Une solution acceptée produit une seule figure :

```text
resultats/figures/empilement_n<n>_<mode>.png
```

Les solutions refusées ne produisent aucun fichier. Si une ancienne figure du
même nom subsiste, un avertissement l'indique explicitement afin d'éviter de la
confondre avec le résultat courant.

Motifs de refus :

- côté entier en mode carré dans carré ;
- record DS7 primitif pour lequel une construction exacte certifiée existe,
  lorsque la solution la dépasse et que cette construction n'est pas
  disponible (par exemple avec `--no-rotate`, car toutes les constructions
  exactes concernées sont inclinées).

En revanche, si `n` est un record DS7 primitif et qu'une construction exacte
certifiée existe, le programme n'abandonne plus : il enregistre cette
construction (si les rotations sont autorisées), avec la figure
correspondante. Une borne publiée purement numérique n'est jamais imposée
sans construction certifiée.

## Constructions et bornes (DS7)

En mode `sq_in_sq`, le programme exploite les constructions exactes du
recensement DS7 d'Erich Friedman :

- Göbel généralisé pour `n = k²+1` (côté `k + √2/2`) ;
- bande diagonale de Göbel : `n = q²+q+3+⌊(q−1)√2⌋` (côté `q+1+1/√2`) ;
- Göbel à bloc `b×b` tourné pour `n = 2q²+2q+b²` (côté `q+1+b/√2`) ;
- variante décentrée pour `n = 2q²+4q+b²+1` (côté `q+3/2+b/√2`) ;
- Trump (`n=11`), Bidwell (`n=17`), Hämäläinen (`n=18`),
  Wainwright (`n=19`), Stenlund (`n=66`) ;
- famille √7 généralisée : côté `(m−1/2)+√7/2`, avec `m=6 → n=39`,
  `m=8 → n=68`, `m=9 → n=86`. Les escaliers sont fixes ; l'angle vaut
  `atan((4−√7)/3)` et la bande inclinée est placée par contacts exacts,
  en balayant les phases au pas de `1/64` et en gardant la phase qui
  maximise le nombre de carrés ;
- record `n=37` de Cantrell (bande diagonale modifiée, angles ≈42.086° et
  45°) : petite table de coordonnées certifiée, car aucune formule close
  publiée n'est disponible. Le côté enregistré est `6.598648`, celui du
  certificat obtenu — légèrement au-dessus de la valeur publiée, jamais
  imposée sans preuve. L'augmentation en « L » en déduit `n=50`
  (côté `7.598648`) ;
- augmentation en « L » : `n' = n + 2⌊s⌋ + 1` pour un côté `s+1`.

Ces configurations sont des **certificats directs** : elles sont vérifiées
géométriquement puis utilisées comme majorants. Les amorces des records
purement numériques restent heuristiques : elles ne sont retenues qu'après
certification.

Le repli DS7 est factorisé : la recherche ne peut dépasser la meilleure
construction exacte certifiée disponible pour `n` ; si elle le fait, cette
construction est enregistrée à la place et son nom est affiché.

La borne basse de la bissection utilise les minorants publiés de DS7 (et non
plus la seule borne d'aire). La console affiche donc un encadrement :

```text
encadrement : 5.391800 <= C(26) <= 5.621375   (écart 4.258 %)
```

Un balayage d'orientations globales affine le certificat final, et le mode
`--range` réutilise les augmentations en « L » d'un `n` déjà résolu pour
amorcer les `n` suivants.

## Robustesse des options

- `--range` accepte `11;15`, `11-15`, `11:15`, `11,15`, `11..15` ou une
  valeur seule, et inverse automatiquement des bornes données à l'envers.
  Une entrée non numérique produit une erreur `argparse` lisible, sans
  traceback.
- `--out` crée le dossier parent manquant.
- Les paramètres numériques sont validés avant le calcul : `--runs >= 1`,
  `--tol > 0`, `--iters >= 1`, `--agents >= 1`, `--polish >= 1`,
  `--anneal-iters >= 0`, `--kicks >= 0`, `--workers >= 1`,
  `--max-slots >= 1`, `--time-budget >= 0`.
- Une solution refusée n'écrit rien et signale une éventuelle figure
  homonyme laissée par un run précédent.

## Certification

Chaque résultat renvoyé est un majorant certifié de la taille optimale :

- ajustement des contacts ;
- dilatation vers une configuration faisable ;
- resserrement au support réel.

La géométrie exacte est revérifiée avant l'enregistrement de la figure.

## Vérification interne

La commande suivante contrôle les gradients analytiques des quatre modes :

```bash
python3 empilement.py --gradcheck
```
