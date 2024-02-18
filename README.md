Repository for the *Detecting Interpretable Subgroup Drifts* paper, currently under submission at KDD 2024. 

Note: the "build a classification model" steps are separate for adult and CelebA (since they use different models). The processing of the results is the same for both datasets (and is found in the src/adult folder for the time being -- some restructuring will occur later on!)

## Build a classification model
First, we need to create a classification model to be used for the following deployment & drift detection
- `python models.py --checkpoint=[checkpoint name] [config]`


Next subgroups are extracted on the training set and stored for later use
- `python precompute.py --checkpoint=[checkpoint name] [config] --minsup=[min support]`

Finally, multiple experiments are run (both positive and negative) -- simulating the deployment mode
- Positive experiments (noise frac may be varied based on the desired injection): `python inject.py --checkpoint=[...] --frac-noise=0.5 [config]`
- Negative experiments: `python inject.py --checkpoint=[...] --frac-noise=0 [config]` 

### Results:
The main results presented in the paper are extracted in:
- src/adult/overall.py and src/adult/overall.ipynb: for the results concerning the global drift detection
- src/adult/ranking.py and src/adult/ranking.ipynb: for the results on the ranking of the subgroups (based on the ground truth injections)
- src/adult/qualitative.ipynb: for the qualitative results
