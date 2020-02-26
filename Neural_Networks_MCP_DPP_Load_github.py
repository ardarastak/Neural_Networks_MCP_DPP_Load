# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 20:49:54 2020

@author: orastak
"""
#%%
import pandas as pd
from matplotlib import pyplot
import statsmodels as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import statistics

import pandas as pd
from sqlalchemy import create_engine
engine = create_engine(engine = create_engine("mssql+pyodbc://user:password@servername/tablename?driver=SQL+Server+Native+Client+11.0"))
mcp_dpp_load= pd.read_sql('SELECT * from MCP_DPP_Load order by date',engine)
#%% excel version
import pandas as pd
mcp_dpp_load= pd.io.excel.read_excel(r"C:\Users\orastak\Desktop\excels for PTF forecast\df_mcp_dpp_load.xlsx", sheetname=0)

#%%
X=mcp_dpp_load[["date","wind","geothermal","dammed_hydro","biomass","river","lep"]]
X=X.set_index("date")
y=mcp_dpp_load[["date","mcp"]]
y=y.set_index("date")
#%%
from sklearn.preprocessing import  MinMaxScaler
sc_x=MinMaxScaler()
X = pd.DataFrame(sc_x.fit_transform(X), index=X.index, columns=X.columns)
sc_y=MinMaxScaler()
y = pd.DataFrame(sc_y.fit_transform(y), index=y.index, columns=y.columns)
#%%
from sklearn.model_selection import train_test_split
X_train=X[:-10000]
X_test=X[-10000:]
y_train=y[:-10000]
y_test=y[-10000:]
#%%
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from xgboost import XGBRegressor

NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()
#%%
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]
NN_model.fit(X_train, y_train, epochs=400, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)
#%%
"""
# Load wights file of the best model :
wights_file = 'Weights-490--18738.19831.hdf5' # choose the best checkpoint 
NN_model.load_weights(wights_file) # load it
"""
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
#%%
predictions = NN_model.predict(X_test)
#%%
MAE = mean_absolute_error(y_test, predictions)
print('XGBoost validation MAE = ',MAE)
y_predict_unscaled=sc_y.inverse_transform(predictions)
y_test_unscaled=sc_y.inverse_transform(y_test)
MAE = mean_absolute_error(y_test_unscaled, y_predict_unscaled)
print('XGBoost validation MAE = ',MAE)
#%%
y_predict_df=pd.DataFrame(y_predict_unscaled)
y_test_df=pd.DataFrame(y_test_unscaled)
#%% use index as x label
import matplotlib.pyplot as plt
y_test_df.plot(use_index=True,figsize=(400,6),color="blue")
y_predict_df.plot(use_index=True,figsize=(400,6),color="red")
plt.title("mcp analysis",fontsize=12)
plt.ylabel("mcp_TL",fontsize=12)
plt.show()
#%% use index as x label
import matplotlib.pyplot as plt
plt.figure(figsize=(200,30), dpi=20)
plt.plot(y_test_df.index[-100:], y_test_df[0][-100:], color="blue",linewidth=1)
plt.plot(y_predict_df.index[-100:],y_predict_df[0][-100:],color="red",linewidth=1)
plt.title("mcp analysis",fontsize=12)
plt.ylabel("mcp_TL",fontsize=12)
plt.show()
#%%
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(NN_model, random_state=1).fit(X_train,y_train)
eli5.show_weights(perm, feature_names = X_train.columns.tolist())
#%%
y_test_df[0].plot(kind='hist', orientation='horizontal')
y_predict_df.plot(kind='hist', orientation='horizontal')
y_test_df[0]

