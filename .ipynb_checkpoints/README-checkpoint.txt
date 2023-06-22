This is the README file for the code and data repositories supporting the paper:
"A Computational Topology-based Spatiotemporal Analysis Technique for Honeybee
Aggregation" by Gharooni-Fard et al.

======================== Dataset files ============================

All the data files can be found under datasets/, organized as follows: 

- datasets/
	- datasets/movies/: Sped-up movies of the experimental and simulation
			    recordings.  

	- datasets/expt_images/: Images generated from the experimental movie data using 
                            ffmeg. Due to data limitations, images for only one dataset 
                            (C0128)are provided under https://github.com/vrd1243/tda_bees_dataset                                   (Used for generating Fig. 7 in the paper.)

	- datasets/sim_txt/: Text file containing the data for the simulation run. (used for generating
			      Fig. 4 in the paper.)

        - datasets/crockers/: Generated crocker matrices of the six experimental datasets.

======================== Code files ============================

All the code files can be found under code/. We provide a short pipeline execution for
extracting the CROCKER plots from sequences of images, compress CROCKER plots using norms/PCA,
and use the resulting time-series data to detect changepoints. 

STEP Ia: Generate crocker plots for experiments from image data. Use code/generate_crocker_matrix.py
This can be skipped and the user may directly use the CROCKER files provided in datasets/crockers/.

Usage: python generate_crocker_matrix.py <input_frame_dir> <output_prefix>
Output: Generates a CROCKER matrix file in results/ directory

STEP Ib: Generate crocker plots for simulation data. Use code/generate_crocker_matrix_sim.py

Usage: python generate_crocker_matrix_sim.py <input_sim_file> <output_prefix>
Output: Generates a CROCKER matrix file in results/ directory

STEP II: Compress crocker plots using PCA. The first principal component of each time-point
is returned. Use the file code/pca.R (to be used for Step IVb)

Usage: Rscript pca.R <input_crocker_file> <output_pca_file>
Output: Generates a file with the first principal component as a function of time.

STEP IIIa: Compress the crocker plots using norms and run clustering based phase-change detection. Use file 
code/clustering_using_norms.py.

	  Four clustering variants are used: a) Agglomerative clustering on crocker norm. 
          b) Agglomerative on tuple (crocker norm, time)
          c) Kmeans on crocker norm
          d) Kmeans on (norm, time) tuple.

Usage: python clustering_using_norms.py <input_crocker_file> <output_prefix>
Output: Generates the following files in results/: a) <prefix>_norms.txt contains the CROCKER norms. 
b) <prefix>_agg.txt contains the clustering labels using the agglomerative clustering on the norm time-series. 
c) <prefix>_agg_with_time.txt contains the clustering labels using the agglomerative clustering on (norm,time) tuple.
d) <prefix>_kmeans.txt contains clustering labels using kmeans on the norm time-series.
e) <prefix>_kmeans_with_time.txt contains clustering labels using kmeans on (norm, time) tuple.

STEP IIIb: Run clustering based phase-change detection on the PCA representation. Use file code/clustering_using_norms.py. 
          Requires PCA compressed file generated in Step III.

Usage: python clustering_using_pca.py <input_pca_file> <output_prefix>
Output: Generates equivalent files of STEP IVa:(b)-(e).

STEP IVa: Run clustering-based phase-change algorithm to determine phase-change locations from files generated in STEP IIIa. Use file code/find_changepoint.py 

Usage: Usage: python find_changepoint.py <input_representation_file>
Output: Returns changepoint locations. This will by default provide the first change-point. To get the next change-point, set the variable first_changepoint to the value of the first change-point.

STEP IVb: Run change-point (R) based phase-change detection on the norms and PCA representation. Use code/changepoint.R 

Usage: Rscript changepoint.R <input_representation_file> {pca|norm}
Output: Returns the changepoint location. To get values of n changepoints, set Q=n in the code.
