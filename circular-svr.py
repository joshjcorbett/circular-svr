import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVR

data='data/bold_data.csv'
oris='data/orientations.csv'
output='results.csv'

# `X` is the feature matrix (n_trials x n_voxels) and `y` holds the orientation targets (n_trials,)
X = np.genfromtxt(data, delimiter=',')
y = np.genfromtxt(oris, delimiter=',')

# Convert orientations to radians for trigonometric transformation
y_rad = (y*2 - 180)*np.pi/180 # adjust this line if data is not in range [0, 180)

# Convert orientations into sine and cosine components
y_sin = np.sin(y_rad)
y_cos = np.cos(y_rad)

# Initialize models for sine and cosine predictions
svr_sin = SVR(kernel='linear')
svr_cos = SVR(kernel='linear')

# Prepare cross-validation, respecting the order of the rows and the block structure of the data 
# The example data has 180 trials made up of 10 blocks of 18 trials (hence 10 CV folds)

kf = KFold(n_splits=10)

# Lists to store predictions and actual orientation values
predictions = []
actuals = []

for train_index, test_index in kf.split(X):
    # Splitting the data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_sin_train, y_sin_test = y_sin[train_index], y_sin[test_index]
    y_cos_train, y_cos_test = y_cos[train_index], y_cos[test_index]
    
    # Train the models
    svr_sin.fit(X_train, y_sin_train)
    svr_cos.fit(X_train, y_cos_train)
    
    # Predict
    y_sin_pred = svr_sin.predict(X_test)
    y_cos_pred = svr_cos.predict(X_test)
    
    # Convert predictions back to angles
    y_pred_angles = np.arctan2(y_sin_pred, y_cos_pred)
    y_pred_degrees = ((y_pred_angles*180/np.pi) + 180)/2  # adjust this line if data is not in range [0, 180)
    
    # Convert actuals back to angles
    y_test_angles = np.arctan2(y_sin_test, y_cos_test)
    y_test_degrees = ((y_test_angles*180/np.pi) + 180)/2 # adjust this line if data is not in range [0, 180)

    # Store predictions and actual values
    predictions.extend(y_pred_degrees)
    actuals.extend(y_test_degrees)

ntrials = len(predictions)

final = np.empty((ntrials,2))
final[:,0] = np.array(actuals)
final[:,1] = np.array(predictions)

np.savetxt(output, final, delimiter=',')
