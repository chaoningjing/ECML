from read_data import *
from preprocess import *
from extract_feature import*
from train import*
import numpy as np
import pandas as df


# np_X = np.random.rand(10, 6 + 300 * 55)
# np_y = np.random.rand(10, 55)
# np_test = np.random.rand(8, 6 + 300 * 55)

print("1 Begin to read data.")
df_X, df_y, df_test = read_and_save()
np_X, np_y, np_test = np.array(df_X.drop(columns='name').values), np.array(df_y.drop(columns='name').values), np.array(df_test.drop(columns=['name']).values)
print("1 Done with shape(train_X): " + str(np_X.shape) + ", shape(train_y): " 
    + str(np_y.shape) + ", shape(test_X): " + str(np_test.shape) + ".")

print("2 Begin to preprocess.")
np_X, np_y, np_test = preprocess_X(np_X), preporcess_y(np_y), preprocess_X(np_test)
print("2 Done with shape(train_X): " + str(np_X.shape) + ", shape(train_y): " 
    + str(np_y.shape) + ", shape(test_X): " + str(np_test.shape) + ".")

print("3 Begin to extract feature.")
X_feature, test_feature =  combine_feature(np_X), combine_feature(np_test)
print("3 Done with shape(train_X): " + str(X_feature.shape) + ", shape(train_y): " 
    + str(np_y.shape) + ", shape(test_X): " + str(test_feature.shape) + ".")

print("4 Begin to extract feature.")
np_output = train_RF(X_feature, np_y, test_feature)
np_output = np_output.reshape((-1, 55))
print("4 Done with shape(test_y): " + str(np_output.shape) + ".")

df_output = df.DataFrame(np_output)
df_output["name"] = df_test["name"]
df_output = df_output.sort_values("name")
df_output.drop(columns=['name']).to_csv("output.txt", sep='\t',float_format='%.12f', header=False, index=False)
