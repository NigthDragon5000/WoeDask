#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Created on Sat Nov  3 10:13:22 2018
@author: Jair Condori
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

class woe:
    def __init__(self,bins=None,nbreaks=10,
                 stat=None,name=None):
        self.bins=bins if bins is None else bins
        self.stat=stat 
        self.iv=None
        self.nbreaks=nbreaks
        self.name=name
        self.woe=None
        self.df=None
        self.per_NA=None


    def fit(self,x,y):
        '''Fitting Information'''
        if not isinstance(x, pd.Series):
            x = pd.Series(x.compute())
        if not isinstance(y, pd.Series):
            y = pd.Series(y.compute())
            
        self.name=x.name
                           
        df = pd.DataFrame({"X": x, "Y": y, 'order': np.arange(x.size)})
        
        if self.bins is None:
           breaks=pd.qcut(df["X"],self.nbreaks,duplicates='drop',retbins=True)[1]
           breaks=breaks[1:-1]
           bins=[]
           bins.append(-float('Inf'))
           for i in breaks:
               bins.append(i)
           bins.append(float('Inf'))
           self.bins=bins
           
        q = pd.cut(df['X'], bins=self.bins,
                   labels=np.arange(len(self.bins)-1).astype(int))
        df['labels']=q.astype(str)
       # q = pd.cut(df['X'], bins=self.bins)
       # df['range']=q.astype(str)
        col_names = {'count_nonzero': 'bad', 'size': 'obs'}
        #self.stat = df.groupby(["labels","range"])['Y'].agg([np.mean, np.count_nonzero, np.size]).rename(columns=col_names).copy()  
        self.stat = df.groupby(["labels"])['Y'].agg([np.mean, np.count_nonzero, np.size]).rename(columns=col_names).copy()
        self.stat['bad_perc']=self.stat['bad']/sum(self.stat['bad'])
        self.stat['good']=self.stat['obs']-self.stat['bad']
        self.stat['good_perc']=self.stat['good']/sum(self.stat['good'])
        self.stat['woe'] = np.log(self.stat['good_perc'].values/self.stat['bad_perc'].values)
        self.stat['iv']= (self.stat['good_perc']-self.stat['bad_perc'])*self.stat['woe']               
        self.stat['per']= self.stat['obs']/sum(self.stat['obs'])
        self.stat['index'] = self.stat.index
        NA=self.stat[self.stat['index'] =='nan']
        self.stat=self.stat[self.stat['index'] !='nan']
        self.stat['index'] = pd.to_numeric(self.stat['index'])
        self.stat=self.stat.sort_values('index')
        self.stat['breaks']=self.bins[1:len(self.bins)]
        self.stat=pd.concat([self.stat,NA],sort=True)
        self.iv=sum(self.stat['iv']) 
        self.df=df
        self.per_NA=sum(df['X'].isnull())/len(df)
        self.stat['z']=self.name
    
    def fit_categorical(self,x,y):
        if not isinstance(x, pd.Series):
            x = pd.Series(x.compute())
        if not isinstance(y, pd.Series):
            y = pd.Series(y.compute())
            
        self.name=x.name
                           
        df = pd.DataFrame({"X": x, "Y": y, 'order': np.arange(x.size)})
        col_names = {'count_nonzero': 'bad', 'size': 'obs'}
        self.stat = df.groupby(["X"])['Y'].agg([np.mean, np.count_nonzero, np.size]).rename(columns=col_names).copy()
        self.stat['bad_perc']=self.stat['bad']/sum(self.stat['bad'])
        self.stat['good']=self.stat['obs']-self.stat['bad']
        self.stat['good_perc']=self.stat['good']/sum(self.stat['good'])
        self.stat['woe'] = np.log(self.stat['good_perc'].values/self.stat['bad_perc'].values)
        self.stat['iv']= (self.stat['good_perc']-self.stat['bad_perc'])*self.stat['woe']               
        self.stat['per']= self.stat['obs']/sum(self.stat['obs'])
        self.stat['index'] = self.stat.index
        #NA=self.stat[self.stat['index'] =='nan']
        #self.stat=self.stat[self.stat['index'] !='nan']
        #self.stat['index'] = pd.to_numeric(self.stat['index'])
        #self.stat=self.stat.sort_values('index')
        #self.stat['breaks']=self.bins[1:len(self.bins)]
        #self.stat=pd.concat([self.stat,NA],sort=True)
        self.iv=sum(self.stat['iv']) 
        self.df=df
        self.per_NA=sum(df['X'].isnull())/len(df)        
        
            
    
    def deploy(self,df):
        ''' Deploy of bins '''

        if not isinstance(df[self.name], pd.Series):
            x = pd.Series(df[self.name].compute())
            if x.isnull().values.any():
                labels = pd.cut(x,bins=self.bins,
                        labels=self.stat['woe'].tolist()[0:-1])
                labels=labels.astype(float)
                labels[labels.isnull()] =self.stat['woe'].tolist()[-1]
            else:                
                labels = pd.cut(x,bins=self.bins,
                        labels=self.stat['woe'].tolist())
                labels=labels.astype(float)
        if isinstance(df[self.name], pd.Series):
            if self.df['X'].isnull().values.any():
                labels = pd.cut(df[self.name],bins=self.bins,
                        labels=self.stat['woe'].tolist()[0:-1])
                labels=labels.astype(float)
                labels[labels.isnull()] =self.stat['woe'].tolist()[-1]
            else:                
                labels = pd.cut(df[self.name],bins=self.bins,
                        labels=self.stat['woe'].tolist())
                labels=labels.astype(float)
            
        return labels  
    
    def plot(self):
        #self.stat['index'] = self.stat.index
        ''' Plot in function of bad rate'''
        return self.stat.plot(kind='bar',x='breaks',y='mean',color='blue')
    
    def optimize(self,depth=2,criterion='gini',
                 samples=1,max_nodes=None,seed=None):
        clf = DecisionTreeClassifier(criterion='gini',random_state=seed,
                                     max_depth=depth,
                                     min_samples_leaf=samples,
                                     max_leaf_nodes=max_nodes)
        df=self.df.dropna()
        name=self.name
        df=self.df.dropna()
        clf.fit(df['X'][:, None],df['Y'])
        breaks=clf.tree_.threshold[clf.tree_.threshold>-2]
        breaks=sorted(breaks)
        bins=[]
        bins.append(-float('Inf'))
        for i in breaks:
            bins.append(i)
        bins.append(float('Inf'))
        self.bins=bins
        self.fit(self.df['X'],self.df['Y'])
        self.name=name
        self.stat['z']=self.name
       

    def auto_optimize(self,criterion='gini',
                 samples=1,seed=None):
        

        df=self.df.dropna()
        name=self.name
        df=self.df.dropna()

        
        #parameters1 = {'max_leaf_nodes':[2,3,4,5,6], 'max_depth':[2,3,4]}
        par1 = {'max_leaf_nodes':[2,3,4,5,6]}
        par2 = {'max_leaf_nodes':[2,3,4,5]}
        par3 = {'max_leaf_nodes':[2,3,4]}
        par4 = {'max_leaf_nodes':[2,3]}
        par5 = {'max_leaf_nodes':[2]}
        
        par_list = [par1,par2,par3,par4,par5]
       
        for parameters in par_list:

            dt = DecisionTreeClassifier(criterion=criterion,random_state=seed,
                                         min_samples_leaf=samples)

            clf = GridSearchCV(dt, parameters,n_jobs =-1,cv=5)
            clf.fit(df['X'][:, None],df['Y'])

            tree= clf.best_estimator_

            breaks=tree.tree_.threshold[tree.tree_.threshold>-2]

            breaks=sorted(breaks)
            bins=[]
            bins.append(-float('Inf'))
            for i in breaks:
                bins.append(i)
            bins.append(float('Inf'))
            self.bins=bins
            self.fit(self.df['X'],self.df['Y'])
            self.name=name
            self.stat['z']=self.name
            if self._checkMonotonic():
                break
    
    def massive(self,df,y_name,plot=False, deploy = False,train=None,                 test=None,len_samples=[0.05,0.10,0.20,0.25],                 nodes=[2,3,4,5],min_iv=0.02):
     
     '''Return a tuple with index 0 that have IV and index 1 that have tables of woes '''
     frames=[]
     names=[]
     iv=[]
     monotonic=[]
     per_NA = []
     depth_arbol=[]
     tablas=[]
     for i in df.columns.tolist():
         try :
            # w=woe(nbreaks=10)
             self.bins=None
             self.name=None  
             self.stat=None
             self.nbreaks=5
             self.fit(df[i],df[y_name])
             iv_prev=0
             tablas.append('')
             for j in list(range(1,4)):
                 for sample in len_samples:
                     for node in nodes:
                         self.optimize(depth=j,samples=int(round(len(df)*sample)),max_nodes=node,seed=0) # Minimo por leaf min_sample_leaf
                         if self._checkMonotonic() and self.iv != float('Inf') and self.iv >min_iv and self.iv>iv_prev:
                             frames.append(self.stat)
                             names.append(i)
                             iv.append(self.iv)
                             monotonic.append(self._checkMonotonic())
                             per_NA.append(self.per_NA)
                             depth_arbol.append(j)
                             tablas[-1] = self.stat
                             iv_prev=self.iv
                             if deploy:
                                 train[str(i+'_binned')]=self.deploy(train)
                                 test[str(i+'_binned')]=self.deploy(test) 

         except KeyboardInterrupt:
             raise Exception('Stop by user')
         except:
             pass
    
     dm =  pd.DataFrame({'Names':names, 'IV':iv, 'Monotono' : monotonic                         ,'per_NA': per_NA,'depth':depth_arbol})

    # Seleccionando los maximos IV con minima profundad de arbol
    
     some_values=dm.groupby('Names')['IV'].max()
     g=dm.loc[dm['IV'].isin(some_values)]
     h=g.groupby('Names')[['depth','Names']].min()
     dm=pd.merge(g,h, on=['depth','Names'])
     
   # Filtrando campos vacios en tablas
     contenedor=[]
     for tabla in tablas:
        try:
            if tabla.empty==False:
                contenedor.append(tabla)
        except:
            pass

     
   # Ploteando
     if plot :
         dm.plot(kind='bar',x='Names',y='IV',color='red')

     return(dm,contenedor)
 
     
    def _checkMonotonic(self):
        bins=np.asarray(self.stat['mean'].values)
        if self.df['X'].isnull().values.any():
            bins=bins[0:len(bins)-1]
        return np.all(np.diff(bins) > 0) or  np.all(np.diff(bins) < 0)
    
    
    def deploy_frame(self,frame,df):
        pre_bins=list(frame['breaks'])
        if pre_bins[-1]==0 or str(pre_bins[-1])=='nan':
            pre_bins.pop()
        bins=[-float('Inf')]
        bins.extend(pre_bins)
        name=frame.iloc[0,11]
        
        self.df = pd.DataFrame()       
        self.bins=bins
        self.name=name
        self.stat=frame
        self.df['X']=df[self.name]

        return self.deploy(df)
    
    @staticmethod
    def merge(obj,bin1,bin2):
            binn =  pd.DataFrame({'bad': [obj.iloc[bin2,0]+obj.iloc[bin1,0]],
                    'bad_perc' :0,
                    'breaks':obj.iloc[bin1,2],
                     'good': [obj.iloc[bin2,3]+obj.iloc[bin1,3]],
                    'good_perc' :0,
                    'index':obj.iloc[bin1,5],
                    'iv':0,
                    'mean':0,
                    'obs':0,
                    'per':0,
                    'woe':0,
                    'z':obj.iloc[bin1,11]
                    })
            for i in list(range(12)):
                obj.iloc[bin1,i]=binn.iloc[0,i]
            obj.iloc[bin2,0:10]=0
            obj['obs']=obj['good']+obj['bad']
            obj['mean']=obj['bad']/obj['obs']
            obj['good_perc']=obj['good']/sum(obj['good'])
            obj['bad_perc']=obj['bad']/sum(obj['bad'])
            obj['woe'] = np.log(obj['good_perc'].values/obj['bad_perc'].values)
            obj['iv']= (obj['good_perc']-obj['bad_perc'])*obj['woe']               
            obj['per']= obj['obs']/sum(obj['obs'])
            obj['woe'][bin2]=obj['woe'][bin1]
            return obj


    def massive2(self,df,y_name,plot=False, deploy = False,train=None,                 test=None,len_samples=[0.05,0.10,0.20,0.25],min_iv=0.02):
     
     '''Return a tuple with index 0 that have IV and index 1 that have tables of woes '''
     frames=[]
     names=[]
     iv=[]
     monotonic=[]
     per_NA = []
     depth_arbol=[]
     tablas=[]
     for i in df.columns.tolist():
         try :
            # w=woe(nbreaks=10)
             self.bins=None
             self.name=None  
             self.stat=None
             self.nbreaks=5
             self.fit(df[i],df[y_name])
             iv_prev=0
             tablas.append('')
             for j in list(range(1,4)):
                 for sample in len_samples:
                     self.optimize(depth=j,samples=int(round(len(df)*sample)),max_nodes=5,seed=0) # Minimo por leaf min_sample_leaf
                     if self.iv >min_iv and self.iv>iv_prev:
                         frames.append(self.stat)
                         names.append(i)
                         iv.append(self.iv)
                         monotonic.append(self._checkMonotonic())
                         per_NA.append(self.per_NA)
                         depth_arbol.append(j)
                         tablas[-1] = self.stat
                         iv_prev=self.iv

         except KeyboardInterrupt:
             raise Exception('Stop by user')
         except:
             pass
    
     dm =  pd.DataFrame({'Names':names, 'IV':iv, 'Monotono' : monotonic                         ,'per_NA': per_NA,'depth':depth_arbol})

    # Seleccionando los maximos IV con minima profundad de arbol
    
     some_values=dm.groupby('Names')['IV'].max()
     g=dm.loc[dm['IV'].isin(some_values)]
     h=g.groupby('Names')[['depth','Names']].min()
     dm=pd.merge(g,h, on=['depth','Names'])
     
   # Filtrando campos vacios en tablas
     contenedor=[]
     for tabla in tablas:
        try:
            if tabla.empty==False:
                contenedor.append(tabla)
        except:
            pass

     
   # Ploteando
     if plot :
         dm.plot(kind='bar',x='Names',y='IV',color='red')

     return(dm,contenedor)


# In[2]:


import numpy as np
import pandas as pd

from dask_ml.datasets import make_classification

from dask_ml.model_selection import train_test_split


from dask_ml.linear_model import LogisticRegression

import dask.array as da, dask.dataframe as dd


import joblib as joblib

from sklearn.ensemble import RandomForestClassifier

import sklearn.datasets



#from dask.distributed import Client, progress
#client = Client(n_workers=2, threads_per_worker=2, memory_limit='1GB')
#client




def f(x,woe):
    if np.isnan(x):
         y= woe
    else :
         y = x
    return y




df=dd.read_csv(r'C:\Users\JAIR\train_titanic.csv')


df.columns



y=df['Survived']



X=df.drop('Survived',1)




X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=123)



train = X_train
test = X_test



min_per = 0.05
seed = 123
cuadros=[]
variables_finales = list()
variables_originales=train.columns
for variable in variables_originales:
    if (str(train[variable].dtype) == 'float64' or str(train[variable].dtype) == 'int64'):
        w=woe()
        w.fit(X[variable],y)
        w.auto_optimize(samples = round(min_per*len(train)),seed=seed)
        cuadros.append(w.stat)

        if (sum(X[variable].isna()) ==0):
            script = 'train= train.assign('+'V'+str(variable)+'_binned'+'=train["'+str(variable)+'"].map_partitions(pd.cut, w.bins,True,w.stat["woe"].tolist()).astype(float))'
            exec(script)
            script = 'test= test.assign('+'V'+str(variable)+'_binned'+'=test["'+str(variable)+'"].map_partitions(pd.cut, w.bins,True,w.stat["woe"].tolist()).astype(float))'
            exec(script)
            script = 'df= df.assign('+'V'+str(variable)+'_binned'+'=df["'+str(variable)+'"].map_partitions(pd.cut, w.bins,True,w.stat["woe"].tolist()).astype(float))'
            exec(script)
                
                
        else: #En este caso es nulo

            script = 'train= train.assign('+'V'+str(variable)+'_binned'+'=train["'+str(variable)+'"].map_partitions(pd.cut, w.bins,True,w.stat["woe"][:-1].tolist()).astype(float))'       
            #Rellenando los nulos
            scriptna =  'train = train.assign('+'V'+str(variable)+'_binned'+'= train["'+'V'+str(variable)+'_binned'+'"].apply(f,args=(w.stat["woe"][-1],)))'    
            exec(script)
            exec(scriptna)

            script = 'test= test.assign('+'V'+str(variable)+'_binned'+'=test["'+str(variable)+'"].map_partitions(pd.cut, w.bins,True,w.stat["woe"][:-1].tolist()).astype(float))'       
            #Rellenando los nulos
            scriptna =  'test = test.assign('+'V'+str(variable)+'_binned'+'= test["'+'V'+str(variable)+'_binned'+'"].apply(f,args=(w.stat["woe"][-1],)))'    
            exec(script)
            exec(scriptna)

            script = 'df= df.assign('+'V'+str(variable)+'_binned'+'=df["'+str(variable)+'"].map_partitions(pd.cut, w.bins,True,w.stat["woe"][:-1].tolist()).astype(float))'       
            #Rellenando los nulos   
            scriptna =  'df = df.assign('+'V'+str(variable)+'_binned'+'= df["'+'V'+str(variable)+'_binned'+'"].apply(f,args=(w.stat["woe"][-1],)))'    

            exec(script)
            exec(scriptna)


        variables_finales.append(str('V'+str(variable)+'_binned'))
            #break       

            
            



print(variables_finales)


# In[171]:


from sklearn.ensemble import RandomForestClassifier


# In[174]:

'''
with joblib.parallel_backend("dask"):
    forest = RandomForestClassifier()
    forest.fit(train[variables_finales].values, y_train)


# In[175]:


print(forest.feature_importances_)
'''

