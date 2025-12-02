# AI_CW2 Group A114

## Structure
1. Install Dependencies and test MediaPipe on one Image
2. Run Feature extraction over all images to produce a CSV where each row = instance + columns = id, x0,y0,z0 ... x20,y20,z20. Save raw features as a CSV file "features_raw.csv"
3. Data Cleaning/ preprocessing:
- Remove rows where MediaPipe failed (no landmarks) OR mark as noise and remove/ impute.
- Remove duplicate IDs.
- Optionally normalise coordinates, scale z, or derive extra features (angles, distance between certain key points).
4. Exploratory data analysis (class balance, missing value counts, feature distributions).
5. Supervised learning:
- Implement kNN from scratch.
- Train Decision Tree and a third classifier (recommend RandomForest or SVM).
- Use 5-fold CV to tune at least two hyperparameters per classifier (include defaults as baseline).
- Choose best hyperparameters, retrain on full training set, evaluate on test set; produce confusion matrices and metrics.

6. Unsupervised learning:
- Remove labels, run KMeans and Hierarchical clustering.
- Evaluate clustering vs true labels (Adjusted Rand Index, Purity), silhouette scores, and compare to classifier outputs.

7. Write results, include visuals, discuss limitations (occlusion, inter-observer error, MediaPipe noise).