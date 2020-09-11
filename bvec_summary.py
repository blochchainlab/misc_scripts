#!/usr/bin/env python3

import numpy as np
import pylab as pl
from scipy.spatial.distance import pdist, squareform   


eps_norm = 1e-6
# remove norm 0 vectors
def remove_zero_norm(v, eps=eps_norm):
	norms = np.linalg.norm(v, axis=1)
	norm_n0_idx = norms > eps
	return v[norm_n0_idx]

# # approximate upper bound on angle for optimal distribution 
# def voronoi_circle_approx(N):
# 	# surface of sphere R=1 is (4 pi), therefore (4 pi / N) area for each point
# 	# the radius of a circle of area (4 pi / N) is (2 / sqrt(N))
# 	# We ignore sphere curvature and find the angle of a triangle of sides 1 1 and 2 times the circle radius (4 / sqrt(N))
# 	# by Triangle cosine rule, angle = arccos(1 - 8/N)
# 	tmp = np.arccos(1-(8/float(N)))
# 	# in degrees
# 	return (180/np.pi)*tmp

# approximate upper bound on angle for optimal distribution 
def voronoi_circle_approx(N):
	# surface of sphere R=1 is (4 pi), therefore (4 pi / N) area for each point
	# the radius of a circle of area (4 pi / N) is (2 / sqrt(N))
	# We compute the angle of an arc of length 2*(2 / sqrt(N)) of a circle of radius 1
	tmp = 4/np.sqrt(N)
	# in degrees
	return (180/np.pi)*tmp

# compute time-neighbors angular distance
def get_neighbors_distance(v):
    # vectorize neighbors distance in radian
    tmp = np.arccos(np.clip(np.sum(v[:-1] * v[1:], axis=1), -1, 1))
    # antipodal symmetry
    tmp2 = np.minimum(tmp, np.pi-tmp)
    # in degrees
    return tmp2*(180/np.pi)

# append antipodally symmetrical directions
def apply_antipodal(v):
	return np.concatenate((v, -v), axis=0)

def compute_dist_matrix(v):
	# in degrees
	return np.arccos((1-squareform(pdist(v, 'cosine'))))*(180/np.pi)

# sort each row of the distance matrix
def sort_dist_matrix(Ma):
	M = Ma.copy()
	for i in range(M.shape[0]):
		M[i] = sorted(M[i])
	return M


def main(dirfname):

	# try to load directions assuming space delimiter
	directions = np.genfromtxt(dirfname)
	if np.any(np.isnan(directions)):
		# try to load directions assuming coma delimiter
		print('There was some NaNs assuming space as delimiter, trying coma')
		directions = np.genfromtxt(dirfname, delimiter=',')
		if np.any(np.isnan(directions)):
			# unknown file formatting
			print('There was still some NaNs, fix the file format')
			print('Quitting...')
			return None

	# We want a shape of (N,3) for N gradients
	if directions.shape[0] == 3:
		print('shape was {}, transposing'.format(directions.shape))
		directions = directions.T

	# remove norm 0 vectors	
	N_init = directions.shape[0]
	directions = remove_zero_norm(directions)
	N = directions.shape[0]
	if N_init - N > 0:
		print('Removed {} norm-0 directions'.format(N_init - N))

	# normalizing vector
	directions = directions / np.linalg.norm(directions, axis=1)[:,None]

	# double scheme with antipodal symmetry
	directions2 = apply_antipodal(directions)

	# approximate optimal distance
	angle_approx = voronoi_circle_approx(directions.shape[0])
	angle_approx2 = voronoi_circle_approx(directions2.shape[0])

	# distance matrix
	distM = compute_dist_matrix(directions)
	distM2 = compute_dist_matrix(directions2)
	distM_s = sort_dist_matrix(distM)
	distM2_s = sort_dist_matrix(distM2)

	# angular distance between temporal neighbors
	temporal_distance = get_neighbors_distance(directions)


	### Check if the global optimization produced something decent
	# We look at distances on the antipodally augmented directions
	# The 2 closest points
	fullsphere_closest = distM2_s[:,1].min()
	# The mean distance to M closest neighbors for each gradient
	Ms = [1,3,5]
	fullsphere_meanMclosest = np.array([distM2_s[:,1:M+1].mean(axis=1) for M in Ms])
	# The mean distance to M closest neighbors across the whole scheme
	fullsphere_meanmeanMclosest = fullsphere_meanMclosest.mean(axis=1)
	print('\nQuality of the overall distribution with antipodal symmetry')
	print('N = {} directions'.format(directions2.shape[0]))
	print('Approximate best distance {:.2f} degrees'.format(angle_approx2))
	print('Closest pair of directions {:.2f} degrees'.format(fullsphere_closest))
	for iM,M in enumerate(Ms):
		print('Mean distance (over {} neighbors) {:.2f} degrees'.format(M, fullsphere_meanmeanMclosest[iM]))
	pl.figure()
	for iM,M in enumerate(Ms):
		pl.subplot(1,len(Ms),iM+1)
		pl.hist(fullsphere_meanMclosest[iM], int(np.round(directions2.shape[0]/4)))
		pl.title('{} neighbors mean distance'.format(M))
	pl.show()


	### Check if the "EDDY" optimization produced something decent
	# We look at distances without the antipodal symmetry
	# The 2 closest points
	fullsphere_closest = distM_s[:,1].min()
	# The mean distance to M closest neighbors for each gradient
	Ms = [1,3,5]
	fullsphere_meanMclosest = np.array([distM_s[:,1:M+1].mean(axis=1) for M in Ms])
	# The mean distance to M closest neighbors across the whole scheme
	fullsphere_meanmeanMclosest = fullsphere_meanMclosest.mean(axis=1)
	print('\nQuality of the hemisphere flipping procedure ("EDDY")')
	print('N = {} directions'.format(directions.shape[0]))
	print('Approximate best distance {:.2f} degrees'.format(angle_approx))
	print('Closest pair of directions {:.2f} degrees'.format(fullsphere_closest))
	for iM,M in enumerate(Ms):
		print('Mean distance (over {} neighbors) {:.2f} degrees'.format(M, fullsphere_meanmeanMclosest[iM]))
	# count directions per "hemisphere"
	print('Hemisphere count')
	print('-/+ X   {} | {}'.format((directions[:,0]<=0).sum(), (directions[:,0]>=0).sum()))
	print('-/+ Y   {} | {}'.format((directions[:,1]<=0).sum(), (directions[:,1]>=0).sum()))
	print('-/+ Z   {} | {}'.format((directions[:,2]<=0).sum(), (directions[:,2]>=0).sum()))
	pl.figure()
	for iM,M in enumerate(Ms):
		pl.subplot(1,len(Ms),iM+1)
		pl.hist(fullsphere_meanMclosest[iM], int(np.round(directions.shape[0]/4)))
		pl.title('{} neighbors mean distance'.format(M))
	pl.show()


	### Check if the "DUTYCYCLE" optimization produced something decent
	# We look at distances between consecutive directions
	print('\nQuality of the direction shuffling procedure ("DUTYCYCLE")')
	print('N = {} directions'.format(directions.shape[0]))
	print('Closest pair of neighbors {:.2f} degrees'.format(temporal_distance.min()))
	print('Mean distance of neighbors {:.2f} degrees'.format(temporal_distance.mean()))
	pl.figure()
	pl.subplot(1,2,1)
	pl.plot(temporal_distance)
	pl.title('Distance to temporal neighbors')
	pl.subplot(1,2,2)
	pl.hist(temporal_distance, int(np.round(directions.shape[0]/4)))
	pl.title('Mean = {:.2f}'.format(temporal_distance.mean()))
	pl.show()

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])

