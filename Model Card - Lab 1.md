Model Card – Lab 1



1\. Intended Use



We created the model as part of Lab 1 to showcase a reproducible, leakage-safe machine learning pipeline. The goal of this assignment was to design a workflow that includes dataset documentation, data-quality auditing, leakage prevention, baseline comparison, and explicit threshold selection under a defined cost assumption.



2\. Metrics



The primary evaluation metric selected for this assignment was Accuracy \& F1 score.



Performance was evaluated using:

&nbsp;	- A single leakage-safe 80/20 train/test split

&nbsp;	- A fixed random seed for reproducibility

&nbsp;	- A scikit-learn Pipeline to ensure preprocessing was fit only on training data

&nbsp;	- Identical preprocessing and split for both baseline and SVM models



The following were reported in the modeling notebook:

&nbsp;	- Primary test-set metric

&nbsp;	- Precision and recall

&nbsp;	- Confusion matrix at a threshold of 0.5 and 0.15

&nbsp;	- Performance at a chosen cost-sensitive threshold



Threshold selection was based on a predefined 2×2 cost matrix that assigned higher cost to false negatives than false positives. Metrics were reported at both thresholds to explicitly show the trade-off between recall and precision.



3\. Limitations



\- The model was evaluated using a single train/test split; no external validation dataset was used.



\- Only one modeling family (SVM) was explored beyond the baseline; no ensemble or tree-based methods were used.



4\. Risks



Misclassification Risk: False positives and false negatives carry different consequences. Although a cost-sensitive threshold was selected, a different threshold choice in another context could increase decision cost.



Bias Risk: If certain subgroups are underrepresented in the dataset, the model may systematically underperform for those groups.



