from sklearn.ensemble import RandomForestRegressor


def train_RF(npy_X, npy_y, npy_test):
    RF = RandomForestRegressor(n_estimators=200, n_jobs=20, verbose=True)
    RF.fit(npy_X, npy_y)
    y_pred = RF.predict(npy_test)
    return y_pred
