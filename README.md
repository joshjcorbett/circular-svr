# circular-svr
Barebones code for running a circular support vector regression ('circular-svr.py') that predicts orientations based on some data, using scikit-learn. Also contains a Jupyter notebook file ('check_results.ipynb') which computes the average angular difference between actual and predicted orientations.

Currently designed to work with orientations in the range [0, 180), but can be easily modified to work with other ranges (e.g., 0-360).

Only requires [scikit-learn](https://scikit-learn.org/stable/index.html) (tested with v1.1.3) and [NumPy](https://numpy.org/) (tested with v1.22.4).

## Detailed description

circular-svr.py will first load the pre-processed BOLD data* from 'data/neuro_data.csv' (data has dimensions n_trials x n_features) and the presented orientations (with length n_trials) from 'data/orientations.csv'.

After converting orientations from the range [0, 180) to [-pi, pi), two linear support vector regressions are trained on the BOLD data to predict the sine and cosine of the angular data. The arctangent of these two angles is then used to predict the presented orientation, and then converted back to the original range.

Training and testing is conducted over 10 cross-validated folds of the data, preserving the order of trials.

The results are saved to 'results.csv', where the first column is the actual orientation and the second column is the predicted orientation.

'check_results.ipynb' loads the 'results.csv' file and computes the mean angular difference between actual and predicted orientations.

*Any data in the format n_trials x n_features will work.