# graph-cut

To reproduce it, use the commit `5770445365f0068b2823a54c85d8f4f001a61afc`.

# logbook

## diagnostic

there are `9484` calls to `disorientation`

these calls take `24.6 sec` out of `26.6 sec` of the call

the total number of adjacencies is `(101 - 1) * 101 + 101 * (101 - 1) = 20200`

the number of calls is roughly half of the number of calls to cover all adjcencies (without duplicate)

this matches the duration of the scikit-image method being twice as long

there are two options:

(a) accelerate `disorientation`

(b) reduce the number of calls to it


Obs: `9484 / 20200 = 47%`

HP's method is already avoiding 53% of all the possible adjacency computations.

## one idea for `(b)`

S'il y a un angle d'euler qui est "trop" différent entre les des orientations, on peut en déduire que la misorientation ne sera pas plus petite qu'une certaine valeur.

Example

Disons qu'on a une tolerance `tol`, on cherche ah calculer des valeurs `(tol_a, tol_b, tol_c)` telles que, pour les orientations (en euler) `o1=(a1, b1, c1)` et `o2=(a2, b2, c2)`, si `|a1-a2| > tol_a` ou `|b1-b2| > tol_b` ou `|c1-c2| > tol_c` donc `misorientation(o1, o2) > tol`.

Avec une condition de ce type tu peux eviter quelques calculs de misorientation.

En fait l'approche peut se faire avec n'importe quelle representation, le calcul Orientation.from_euler est relativement cheap.

## about accelerating (option (a))

One issue with this algorithm is that it decides wich adjacencies to compute on the fly, meaning that we do not know in advance which ones will be computed.

This makes it harder - and inefficient - to parallelize. If one computes all of them then it becomes easier.

In other words, accelerating the method `Orientation.disorientation` will allow further acceleration for bigger images.

### how much can it accelerate with that?

about 15% for this image

this means that instead for computing `47%` of the adjacencies, it'd compute (in the best case scenario) `47% - 15% = 32%`

the improvement in time would be something like `32 / 47 = 68% = 32% of reduction`

this wont change the method's duration scale