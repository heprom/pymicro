from math import pi, cos, sin
import numpy as np
import random
from matplotlib import pyplot as plt, colors as mcol
from pymicro.view.vol_utils import compute_affine_transform

colors = 'brgmk'
my_gray = (0.8, 0.8, 0.8)
random.seed(13)
n = 5
ref_points = np.empty((n, 2))
for i in range(n):
    ref_points[i, 0] = random.randint(0, 10)
    ref_points[i, 1] = random.randint(0, 10)
    print(ref_points[i])
    plt.plot(ref_points[i, 0], ref_points[i, 1], 'o', color=colors[i], markersize=10, markeredgecolor='none',
             label='reference points' if i == 0 else '')
plt.grid()
plt.axis([0, 20, 0, 20])
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend(numpoints=1)
plt.savefig('pointset_registration_1.png', format='png')

# compute the affine transform by composing R, S and T
s = 1.4
angle = 22 * pi / 180.
tx, ty = 5, 3
S = np.array([[s, 0.0, 0.0],
              [0.0, s, 0.0],
              [0.0, 0.0, 1.0]])
R = np.array([[cos(angle), -sin(angle), 0.0],
              [sin(angle), cos(angle), 0.0],
              [0.0, 0.0, 1.0]])
T = np.array([[1.0, 0.0, tx],
              [0.0, 1.0, ty],
              [0.0, 0.0, 1.0]])
A = np.dot(T, np.dot(S, R))
print('full affine transform:\n{:s}'.format(A))

# transform the points
tsr_points = np.empty_like(ref_points)
for i in range(n):
    p = np.dot(A, [ref_points[i, 0], ref_points[i, 1], 1])
    tsr_points[i, 0] = p[0]
    tsr_points[i, 1] = p[1]
    # tsr_points[i] = T[:2] + np.dot(np.dot(S[:2, :2], R[:2, :2]), ref_points[i])
    print(tsr_points[i])
    plt.plot(tsr_points[i, 0], tsr_points[i, 1], 's', color=colors[i], markersize=10, markeredgecolor='none',
             label='transformed points' if i == 0 else '')
# overwrite reference points in light gray
plt.plot(ref_points[:, 0], ref_points[:, 1], 'o', color=my_gray, markersize=10, markeredgecolor='none',
         label='reference points' if i == 0 else '')
# draw dashed lines between reference and transformed points
for i in range(n):
    plt.plot([ref_points[i, 0], tsr_points[i, 0]], [ref_points[i, 1], tsr_points[i, 1]], '--', color=colors[i])
plt.legend(numpoints=1)
plt.savefig('pointset_registration_2.png', format='png')

# compute the affine transform from the point set
translation, transformation = compute_affine_transform(ref_points, tsr_points)
invt = np.linalg.inv(transformation)
offset = -np.dot(invt, translation)
ref_centroid = np.mean(ref_points, axis=0)
tsr_centroid = np.mean(tsr_points, axis=0)
new_points = np.empty_like(ref_points)
for i in range(n):
    new_points[i] = ref_centroid + np.dot(transformation, tsr_points[i] - tsr_centroid)
    print('point %d will move to (%3.1f, %3.1f) to be compared with (%3.1f, %3.1f)' % (
        i, new_points[i, 0], new_points[i, 1], ref_points[i, 0], ref_points[i, 1]))
    plt.plot(new_points[i, 0], new_points[i, 1], 'x', color=colors[i], markersize=12,
             label='new points' if i == 0 else '')
plt.legend(numpoints=1)
plt.savefig('pointset_registration_3.png', format='png')
plt.show()
