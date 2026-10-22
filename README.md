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
- cas `n = k² + 1` au-dessus de la borne `k + √2/2`.

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
