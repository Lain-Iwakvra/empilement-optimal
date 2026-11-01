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

Les solutions refusées ne produisent aucun fichier :

- côté entier en mode carré dans carré ;
- record DS7 primitif pour lequel une construction exacte certifiée existe,
  lorsque la solution la dépasse et que cette construction n'est pas
  disponible (par exemple avec `--no-rotate`).

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
  Wainwright (`n=19`), Stenlund (`n=66`), famille √7 (`n=86`) ;
- augmentation en « L » : `n' = n + 2⌊s⌋ + 1` pour un côté `s+1`.

Ces configurations sont des **certificats directs** : elles sont vérifiées
géométriquement puis utilisées comme majorants. Les amorces des records
purement numériques restent heuristiques : elles ne sont retenues qu'après
certification.

La borne basse de la bissection utilise les minorants publiés de DS7 (et non
plus la seule borne d'aire). La console affiche donc un encadrement :

```text
encadrement : 5.391800 <= C(26) <= 5.621375   (écart 4.258 %)
```

Un balayage d'orientations globales affine le certificat final, et le mode
`--range` réutilise les augmentations en « L » d'un `n` déjà résolu pour
amorcer les `n` suivants.

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
