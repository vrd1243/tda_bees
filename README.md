This is the README file for the code and data repositories supporting the paper:
"A Computational Topology-based Spatiotemporal Analysis Technique for Honeybee
Aggregation" by Gharooni-Fard et al.

======================== Dataset files ============================

All the data files can be found under datasets/, organized as follows: 

- datasets/
	- datasets/movies/: Original movies for the experimental and simulation
			    recordings. For the experimental recordings, you can
			    split the movies into images using the ffmpeg tool. 

	- datasets/expt_images/: Images generated from the experiemntal movie data using 
                            ffmeg. Images for only one dataset are provided (Used for 
			    generating Fig. 7 in the paper.)

	- datasetts/sim_txt/: Text file containing the data for the simulation run. (used for generating
			      Fig. 4 in the paper.)

======================== Code files ============================

All the code files can be found under code/. We provide a short pipeline execution for
extracting the CROCKER plots from sequences of images, compress CROCKER plots using norms/PCA,
and use the resulting time-series data to detect changepoints. 

STEP I: Image Generation from Videos

For a movie in datasets/movies, run the following script: 

Usage: ffmpeg -i <movie file> -vf fps=30 datasets/expt_images/<image folder>/img%04d.png
Output: Generates individual images for the movie file.

STEP II: Generate crocker plots. Use code/generate_crocker_matrix.py

Usage: python generate_crocker_matrix.py <input_frame_dir> <output_prefix>
Output: Generates a CROCKER matrix file in results/ directory

STEP III: Compress crocker plots using PCA. Use the file code/pca.R (to be used for Step IVb)

Usage: Rscript pca.R <input_crocker_file>
Output: Generates a pca file in results/ directory

STEP IVa: Compress the crocker plots using norms and run clustering based phase-change detection. Use file code/clustering_using_norms.py.
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

STEP IVb: Run clustering based phase-change detection on the PCA representation. Use file code/clustering_using_norms.py. 
          Requires PCA compressed file generated in Step III.

Usage: python clustering_using_pca.py <input_pca_file> <output_prefix>
Output: Generates equivalent files of STEP IVa:(b)-(e).

STEP IVc: Run change-point based phase-change detection on the norms and PCA representation. Use code/changepoint.R 

Usage: Rscript changepoint.R <input_representation_file>
Output: Returns the time-stamp of change-point. 
          


