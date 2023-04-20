# --------------------------------------------------------------------------------
# FUNCTIONS
# --------------------------------------------------------------------------------

# Libreries
from .libreries import *

# Definitions
colors_classification_models={

                'LogisticRegression' : 'ghostwhite',
                'KNeighborsClassifier' : 'snow',
                'DecisionTreeClassifier' : 'ivory',
                'ExtraTreeClassifier' : 'whitesmoke',
                'RandomForestClassifier' : 'darkseagreen',
                'BaggingClassifier' : 'mediumaquamarine', 
                'AdaBoostClassifier' : 'honeydew',
                'GradientBoostingClassifier' : 'teal',
                'SVC' : 'floralwhite',
                'XGBClassifier' : 'aquamarine',
                'VotingClassifier': 'lightcyan',
                'LinearDiscriminantAnalysis': 'seashell'
            }


# Functions
def save_file(file, head, content, dir = 'data', sep = ';'):
    '''
    Objective: 
    ---

        Archive data in a .csv or .txt from a list.

    args:
    ----
    file: str; file name.
    head: str; header/columns of the file separated by ';'.
    content: list; content to store.

    returns:
    ----
    Does not return anything, just performs the saving.

    '''
    # Check if the file/directory exists
    # print(os.getcwd())
    dir_path = os.path.join(os.getcwd(), dir)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f'{file}')

    # If the file already exists, append the content to it
    if os.path.exists(file_path):
        with open(file_path, mode='a', newline='\n') as out:
            # Save the information
            # print(type(content))
            [out.write(cont+sep) if (content[-1]!=cont) else out.write(cont) for cont in content]
            out.write('\n')

    # If the file does not exist, create it and write the header and content to it
    else:   
        with open(file_path, mode='w') as out:
            out.write(head+'\n')
            print(type(content))

            # Save the information
            [out.write(cont+sep) if (content[-1]!=cont) else out.write(cont) for cont in content]
            out.write('\n')


def dict4save(dict, name_file, dirf, addcols=False, cols='new cols', vals='values cols added', sep=';'):
    '''
    Objective:
    ---
        
        Save the values obtained as a dictionary in the .csv or .txt file.

    args:
    ---
    dict: dict; dictionary with the values to be saved.
    name_file: str; name of the file where the data will be saved.
    dirf: str; name of the directory or relative path where the file is located.
    addcols: bool; 
                True: if we want to add extra columns to those in the dictionary.
                False: otherwise.
    cols: str; name of the column(s). (optional; use separator if there are more than one)
    vals: str; values of the new column(s). (optional; use separator if there are more than one)
    sep: str; separator used in the .csv or .txt file.

    returns:
    ---
    print: 'Saved' when finished.
    '''

    str_dict = ''

    for str_k in list(dict.keys()):
        str_dict = str_dict + str_k + sep
    
    values = [str(val) for val in dict.values()]
    
    if addcols:
        values.insert(0, vals)
        save_file(name_file, cols+sep+str_dict[:-1], values, dir=dirf)
    else:
        save_file(name_file, str_dict[:-1], values, dir=dirf)
    
    print('Saved')


def eval_metrics(y_pred, y_test, clf=True, c_matrix=False):
    '''
    Objective: 
    ---
    Evaluate the model with the corresponding metrics.

    args:
    ---
    y_pred: the model's prediction. 
    y_test: the actual test result. 
    clf: bool; True: if it is a classification. (by default)
               False: if it is a regression.
    c_matrix: bool; True: obtain confusion matrix.
                    False: do not obtain matrix. (by default)

    returns:
    ---
    dict; result of the metrics.

    * Except if c_matrix True and clf True:
        dict, array; metric results, confusion matrix.
    '''

    if clf:
        
        clf_metrics = {
            'ACC': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            #'ROC': roc_auc_score(y_test, y_pred),
            'Jaccard': jaccard_score(y_test, y_pred)
        }
        
        if c_matrix:
            confusion_mtx = confusion_matrix(y_test, y_pred)
            return clf_metrics, confusion_mtx
        else:
            return clf_metrics

    else:

        reg_metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }   

        return reg_metrics  


def baseline(X, y, base_model = None, clf = True, file_name = 'metrics.csv', dir_file = 'model/model_metrics', tsize = 0.2, random = 77):
    '''
    Objective: 
    ---
    Create a rough initial model.

    args:
    ----
    X: pd.DataFrame; the complete dataset, with numerical values.
    y: pd.DataFrame; the target column, the dependent variable.
    base_model: estimator to be used. By default, it uses RandomForest(). (optional)
    clf: True/False; if it is a classification dataset (True) or a regression dataset (False). (optional)
    tsize: float; test size [0.0,1.0]. (optional)
    random: int; random state, seed. (optional)

    returns:
    ----
    Evaluation metrics of the model and the pack: 
        model_pack = {

            'trained_model' : estimator,
            'Xytest' : [X_test, y_test],
            'Xytrain' : [X_train, y_train],
            'ypred' : y_pred
        }
    '''

    if base_model == None:
        if clf:
            base_model = RandomForestClassifier()
        else:
            base_model = RandomForestRegressor()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = tsize, random_state = random)

    estimator=base_model.fit(X_train,y_train)
    y_pred=estimator.predict(X_test)

    model_pack = {

        'trained_model' : estimator,
        'Xytest' : [X_test, y_test],
        'Xytrain' : [X_train, y_train],
        'ypred' : y_pred
    }

    metrics = eval_metrics(y_pred,y_test,clf)
    model_str = str(base_model)[0:str(base_model).find('(')]

    dict4save(metrics, file_name, dir_file, addcols=True, cols='model', vals=model_str,sep=';')
    
    return metrics, model_pack

def choose_params(model,clf = True):
    '''
    Objective:
    ---
    
        Choose the parameters to test for a specific model.
        
    args:
    ----
    model: model for which parameters are wanted.
    clf: bool; True: if it is a classification model.

    returns:
    ----
    dict; with the parameters to test.

    '''

    if clf :

        clf_params = {

            'LogReg' : {

                #'penalty' : ['l1','l2','elasticnet','none'],
                #'penalty' : ['l2'],
                #'class_weight' : ['none','balanced'],
                #'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                #'solver' : ['newton-cg', 'liblinear', 'sag', 'saga'],
                #'max_iter' : [50,75,100]
                
            #},
            #{
                'penalty': ['l1', 'l2'],
                'C': [0.01, 0.1, 1, 10],
                'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
                'max_iter': [100, 500, 1000]
            }
            #]
        ,

            'KNNC' : {

                'n_neighbors' : [3,5,7,9,11,13,15],
                'weights' : ['uniform','distance'],
                'algorithm' : ['ball_tree','kd_tree','brute','auto'],
                'leaf_size' : [20,30,40],
                'p' : [1,2]

            },

            'DTC' : {
                
                'criterion' : ['log_loss','gini','entropy'],
                'splitter' : ['best','random'],
                'max_depth' : [7,9,11,13,None],
                'max_features': ['log2','sqrt','auto'],
                'class_weight' : [None,'balanced']

            },

            'ETC' : {
                #'n_estimators': np.linspace(10,80,10).astype(int),
                'criterion': ['gini','entropy'],
                'max_depth' : [7,9,11,13,None],
                'max_features': ['log2','sqrt',None],
                'class_weight' : [None,'balanced'],
                'max_leaf_nodes' : [None,3,7,11]
            },

            'RFC' : {
                'n_estimators': np.linspace(10,150,10).astype(int),
                'criterion': ['gini','entropy'],
                'max_depth' : [7,9,11,13,None],
                'max_features': ['log2','sqrt',None],
                'class_weight' : [None,'balanced']
            },

            'BagC' : {
                #'base_estimator__class_weight': ['balanced'],
                #'base_estimator__criterion': ['gini'],
                #'base_estimator__max_depth': [7], 
                #'base_estimator__max_features': ['log2'], 
                #'base_estimator__splitter': ['best'],
                'n_estimators' : [10, 20, 30, 50, 100],
                'max_samples' : [0.05, 0.1, 0.2, 0.5]
            },
            'AdaBC' : {
                #'base_estimator__class_weight': ['balanced'],
                #'base_estimator__criterion': ['gini'],
                #'base_estimator__max_depth': [7], 
                #'base_estimator__max_features': ['log2'], 
                #'base_estimator__splitter': ['best'],
                'n_estimators' : [10, 20, 30, 50, 100]
            
            },

            'GBC' : [{
                #'base_estimator__class_weight': ['balanced'],
                #'base_estimator__criterion': ['gini'],
                #'base_estimator__max_depth': [7], 
                #'base_estimator__max_features': ['log2'], 
                #'base_estimator__splitter': ['best'],
                'n_estimators' : [10, 20, 30, 50, 100],
                'max_depth' : [7,9,11,13,None],
                'criterion': ['friedman_mse','mse'],
                'loss': ['log_loss','exponential']
            },
            {
              'loss' : ["log_loss"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01, 0.001],
              'max_depth': [4, 8,16],
              'min_samples_leaf': [100,150,250],
              'max_features': [0.3, 0.1]
              }
            ],

            'SVC' : [
                #{'C' : [1,10,50],'kernel' : ['poly','sigmoid','precomputed'],'degree' : [3,4],'class_weight' : [None,'balanced']},
                {'C': [1, 10, 100, 1000], 'kernel': ['linear'], 'class_weight' : [None,'balanced']},
                {'C': [1, 10, 100, 1000],  'kernel': ['rbf'],'class_weight' : [None,'balanced']}
            ],

            'XGBC' : {
                'nthread':[4], #when use hyperthread, xgboost may become slower
                'objective':['binary:logistic'],
                'learning_rate': [0.05], #so called `eta` value
                'max_depth': [4,5,6,7],
                'min_child_weight': [1, 5, 10, 11],
                'subsample': [0.6,0.8,1.0],
                'colsample_bytree': [0.6,0.7,1.0],
                'n_estimators': [5,50,100], #number of trees, change it to 1000 for better results
                'missing':[-999],
                'seed': [1337]
            },
        }

        return clf_params[model]

    else :

        reg_params = {

            'LinReg' : {},
            'KNNR' : {},
            'GNBR' : {},
            'BNBR' : {},
            'ENR' : {},
            'DTR' : {},
            'ETR' : {},
            'RFR' : {},
            'BagR' : {},
            'AdaBR' : {},
            'GBR' : {},
            'SVR' : {},
            'XGBR' : {}
        }

        return reg_params[model]

def choose_models(model, params, clf = True):
    '''
    Objective:
    ---
    
        Choose the corresponding model or models.

    args:
    ----
    model: str; model to be selected. 
            'all': selects all models. 

    returns:
    ----
    The selected model or models.

    '''
    
    if clf :
        if params == None:

            classification_models={

                'LogReg' : LogisticRegression(),
                'KNNC' : KNeighborsClassifier(),
                'DTC' : DecisionTreeClassifier(),
                'ETC' : ExtraTreeClassifier(),
                'RFC' : RandomForestClassifier(),
                'BagC' : BaggingClassifier(), 
                'AdaBC' : AdaBoostClassifier(),
                'GBC' : GradientBoostingClassifier(),
                'SVC' : SVC(),
                'XGBC' : XGBClassifier(),
                'VC': VotingClassifier(estimators=[('RFC',RandomForestClassifier())]),
                #'MulNB': MultinomialNB(params),
                #'LDA': LinearDiscriminantAnalysis()
            }

        else:
            classification_models={

                'LogReg' : LogisticRegression(params),
                'KNNC' : KNeighborsClassifier(params),
                'DTC' : DecisionTreeClassifier(params),
                'ETC' : ExtraTreeClassifier(params),
                'RFC' : RandomForestClassifier(params),
                'BagC' : BaggingClassifier(params), 
                'AdaBC' : AdaBoostClassifier(params),
                'GBC' : GradientBoostingClassifier(params),
                'SVC' : SVC(params),
                'XGBC' : XGBClassifier(params),
                'VC': VotingClassifier(params),
                #'MulNB': MultinomialNB(params),
                #'LDA': LinearDiscriminantAnalysis(params)
            }


        if model == 'all' and params == None:
            return classification_models

        else:
            return classification_models[model]

    else : 

        if params == None:

            regression_models={

                'LinReg' : LinearRegression(),
                'KNNR' : KNeighborsRegressor(),
                'GNBR' : GaussianNB(),
                'BNBR' : BernoulliNB(),
                'ENR' : ElasticNet(),
                'DTR' : DecisionTreeRegressor(),
                'ETR' : ExtraTreeRegressor(),
                'RFR' : RandomForestRegressor(),
                'BagR' : BaggingRegressor(), 
                'AdaBR' : AdaBoostRegressor(),
                'GBR' : GradientBoostingRegressor(),
                'SVR' : SVR(),
                'XGBR' : XGBRegressor()
                
            }

        else:

            regression_models={

                'LinReg' : LinearRegression(params),
                'KNNR' : KNeighborsRegressor(params),
                'GNBR' : GaussianNB(params),
                'BNBR' : BernoulliNB(params),
                'ENR' : ElasticNet(params),
                'DTR' : DecisionTreeRegressor(params),
                'ETR' : ExtraTreeRegressor(params),
                'RFR' : RandomForestRegressor(params),
                'BagR' : BaggingRegressor(params), 
                'AdaBR' : AdaBoostRegressor(params),
                'GBR' : GradientBoostingRegressor(params),
                'SVR' : SVR(params),
                'XGBR' : XGBRegressor(params)
                
            }

        if model == 'all'and params == None:
            return regression_models

        else:
            return regression_models[model]

def save_model(model, dirname):
    '''
    Objective: 
    ---
    
        Save the model in the chosen folder.

    args:
    ---
    model: model to be saved.
    dirname: str; relative path to the folder where the model will be saved.

    returns:
    ---
    Prints a message indicating that the input model has been saved.

    Returns the relative path of the model.
    '''
    model_str = str(model)
    model_str = model_str[0:model_str.find('(')]
    path_dir = os.path.join(os.getcwd(), dirname)
    
    os.makedirs(path_dir, exist_ok=True)
    path_file = os.path.join(path_dir, f'{model_str}.pkl')
    
    if os.path.exists(path_file):
        for i in range(1, 99):
            path_file = os.path.join(path_dir, f'{model_str}_{i}.pkl')
            if os.path.exists(path_file):
                x='another try'
            else:
                pickle.dump(model, open(path_file, 'wb'))
                the_path = os.path.join(dirname, f'{model_str}_{i}.pkl')
                break
    else:
        pickle.dump(model, open(path_file, 'wb'))
        the_path = os.path.join(dirname, f'{model_str}.pkl')

    print(f'Model {model_str} saved')
    
    return the_path 

def train_predict_best_model(data, target, model, params, scoring, tsize=0.2, random=77, scaling=False, balancing=False, scaler=None, Xy=True):
    '''
    Objective: 
    ---
        
        Train the model with the best introduced parameters and make predictions using it.

    args:
    ---
    data: Complete dataset.
    target: str; target variable.
    model: Model to use.
    params: dict; set of parameters to modify and test through GridSearchCV.
    scoring: dict; metric(s) to optimize in GridSearchCV.
    tsize: float; size of the test set as a fraction of the total dataset. (Default: 0.2)
    random: int; parameter chosen for randomizing. (Default: 77)
    scaling: bool; True to scale the data and False to not scale it. 
    scaler: None if the scaling is done using the StandardScaler generated/trained in the function, or the trained scaler if a pre-trained scaler is to be used.
    balancing: bool; True to balance the data and False if it is not required.
    Xy: True if data is the column we want to introduce directly and target is the label column.

    returns:
    ---
    estimator, X_test, y_test, X_train, y_train, y_pred
    '''

    # Data separation
    if (Xy):
        X = data
        y = target
    else:
        X = data.drop([target], axis=1)
        y = data[target].copy()
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tsize, random_state=random)
    
    # Scaling:
    if scaling:
        if scaler == None:
            scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Balancing:
    if balancing:
        sm = SMOTEENN(random_state=random) 
        X_train, y_train = sm.fit_resample(X_train, y_train.ravel()) 
        
    # Training the model: 
    estimator = GridSearchCV(model, params, scoring=scoring, refit='AUC', return_train_score=True)
    estimator.fit(X_train,y_train)

    # Predicting with the best estimator
    y_pred = estimator.best_estimator_.predict(X_test)

    return estimator, X_test, y_test, X_train, y_train, y_pred


def save_all(model, estimator, params, metrics, file_name='metrics.csv', dir_file='model/model_metrics', dir_model_file='model'):
    '''
    Objective: 
    ---
    
    '''
    model_str = str(model)[0:str(model).find('(')]
    
    file2save = {'model':model_str, 'params_tried': str(params), 'best_params':str(estimator.best_params_)}
    file2save.update(metrics)
    
    # Save model:
    model_path = save_model(estimator.best_estimator_, dir_model_file)

    file2save.update({'model_path' : model_path})

    # Save file:
    dict4save(file2save, file_name, dir_file, addcols=False, sep=';')


def models_generator(data, target, model=None, params=None, clf=True, scaling=True, scaler=None, scoring={"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score)}, balancing=False, file_name='metrics.csv', dir_file='model/model_metrics', dir_model_file='model', tsize=0.2, random=77):
    '''
    Objective: 
    ---
    
        Train a model, evaluate its metrics, and save the results in the indicated file.

    args:
    ----
    data: pd.DataFrame; the complete dataset, with numerical values.
    target: str; name of the target column, dependent variable.
    model: estimator to be used. By default, it uses RandomForest(). (optional)
    params: parameters that are tested to obtain the best model through GridSearchCV. (optional)
    clf: True/False; if it is a classification dataset (True) if it is a regression dataset (False). (optional)
    tsize: float; test size [0.0, 1.0]. (optional)
    random: int; random state, seed. (optional)
    scaling: bool; True to scale and False not to scale the data.
    scaler: None if the scaling is generated/trained in the function itself, and the trained scaler if a specific pre-trained scaler is intended to be used.
    scoring: dict; metric(s) to optimize in GridSearchCV.
    balancing: bool; True to balance the data, and False if it is not required.
    file_name: str; the name of the file where the results will be saved. (optional)
    dir_file: str; the directory where the file will be saved. (optional)
    dir_model_file: str; the directory where the model will be saved. (optional)


    returns:
    ----
    model_pack = {

        'trained_model': estimator,
        'Xytest': [X_test, y_test],
        'Xytrain': [X_train, y_train],
        'ypred': y_pred,
        'metrics': metrics
    }

    '''

    # Default model:
    if model is None:
        if clf:
            model = RandomForestClassifier()
        else:
            model = RandomForestRegressor()

    # Trained estimator and prediction: 
    estimator, X_test, y_test, X_train, y_train, y_pred = train_predict_best_model(data, target, model, params, scoring, tsize=tsize, random=random, scaling=scaling, balancing=balancing, scaler=scaler)

    # Evaluate metrics:
    metrics = eval_metrics(y_pred,y_test,clf)
    
    # Save model and obtained metrics:
    save_all(model, estimator, params, metrics, file_name = file_name, dir_file = dir_file, dir_model_file = dir_model_file)

    # Output variable: 
    model_pack = {

        'trained_model' : estimator,
        'Xytest' : [X_test, y_test],
        'Xytrain' : [X_train, y_train],
        'ypred' : y_pred,
        'metrics' : metrics
    }

    return model_pack


def take_params(str_params):
    '''
    Objective:
    ---
        
        Transform parameters stored as string to dictionary with values readable by GridSearchCV()
    '''

    dict_params = eval(str_params)
    params = dict_params.copy()

    for k, v in dict_params.items():
        params[k] = [v]

    return params


def models_duration(data, target, model, params, scoring={"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score)}):
    '''
    Objective:
    ---
    
        Measure the training and prediction time of a model
    '''

    start = time.time()
    train_predict_best_model(data, target, model, params, scoring)
    end = time.time()

    return end - start


def add_duration(data, target, csv):
    '''
    Objective:
    ---
    
        Add duration to the csv with model data
    '''
    times = []
    csv2 = csv.reset_index(drop=True)

    for i in range(len(csv)):
        name = csv2['model'][i]
        models_keys = ['LogReg', 'KNNC', 'DTC', 'ETC', 'RFC', 'BagC', 'AdaBC', 'GBC', 'SVC', 'XGBC', 'MLPC']
        model = [mk for mk in models_keys if name[0] == mk[0]][0]
        if model == 'MLPC':
            times.append(models_duration(data, target, MLPClassifier(), take_params(csv2['best_params'][i])))
        else:
            times.append(models_duration(data, target, choose_models(model, params=None), take_params(csv2['best_params'][i])))

    csv2['duration [s]'] = times

    return csv2

def clf_models_comparation(csv,metric='Precision',xlabel='Precision',ylabel='Models',title='Models comparation',del_last=True):
    '''
    Objective:
    ---
    Compare the metrics obtained with different classification models through a histogram.
    '''
    saved_metrics = csv.copy()

    cv_means = saved_metrics[metric]
    lista = saved_metrics['model']

    if del_last:
        cv_means = cv_means[:-1]
        lista = lista[:-1]

    selected_colors=[colors_classification_models[model] for model in lista]   


    cv_frame = pd.DataFrame(
        {
            "CrossValMeans":cv_means.astype(float),
            "Models": lista
        })

    cv_plot = sns.barplot("CrossValMeans","Models", data = cv_frame, palette=selected_colors)

    cv_plot.set_xlabel(xlabel,fontweight="bold")
    cv_plot.set_ylabel(ylabel,fontweight="bold")
    cv_plot = cv_plot.set_title(title,fontsize=16,fontweight='bold')

def models_comparation(csv,metric='Precision',xlabel='Precision',ylabel='Models',title='Models comparation',del_last=True, selected_colors='Blues'):
    '''
    Objective:
    ---
    Compare the metrics obtained with different classification models through a histogram.
    '''

    saved_metrics = csv.copy()

    cv_means = saved_metrics[metric]
    lista = saved_metrics['model']

    if del_last:
        cv_means = cv_means[:-1]
        lista = lista[:-1]  

    cv_frame = pd.DataFrame(
        {
            "CrossValMeans":cv_means.astype(float),
            "Models": lista
        })

    cv_plot = sns.barplot("CrossValMeans","Models", data = cv_frame, palette=selected_colors)

    cv_plot.set_xlabel(xlabel,fontweight="bold")
    cv_plot.set_ylabel(ylabel,fontweight="bold")
    cv_plot = cv_plot.set_title(title,fontsize=16,fontweight='bold')


def nuwe_prediction_file(id,prediction,rel_path = None,folder = "/data/processed/" ,file = "prediction_transf_1.json", colab = True): 
  '''
  Objective:
  ---

    Create the prediction.json file with the prediction values
  
  args.
  ---

  ret.
  ---
  print "json saved"
  '''
  
  prediction_json = {'target':{str(id[i]):int(prediction[i]) for i in range(len(id))}}

  if colab: 
    with open(rel_path + folder + file, "w") as f:
        json.dump(prediction_json, f)
  else:  
    with open(folder+file, "w") as f:
        json.dump(prediction_json, f)
  
  print('json saved')




#######
# NLP #
#######


# Preprocesamiento de los datos de texto
def preprocess_text(text):
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Eliminar signos de puntuación
    text = text.replace(',', '').replace('.', '').replace('!', '').replace('?', '').replace(':', '').replace(';', '').replace('-', '').replace('_', '').replace('"', '').replace('\'', '').replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '')
    # Convertir a minúsculas
    text = text.lower()
    # Tokenización
    tokens = nltk.word_tokenize(text)
    # Eliminación de stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Lematización
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    # Reconstrucción del texto
    text = ' '.join(lemmas)
    return text



# function to print sentiments
# of the sentence.
def sentiment_scores(sentence):

	# Create a SentimentIntensityAnalyzer object.
	sid_obj = SentimentIntensityAnalyzer()

	# polarity_scores method of SentimentIntensityAnalyzer
	# object gives a sentiment dictionary.
	# which contains pos, neg, neu, and compound scores.
	sentiment_dict = sid_obj.polarity_scores(sentence)
	
	print("Overall sentiment dictionary is : ", sentiment_dict)
	print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
	print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral")
	print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")

	print("Sentence Overall Rated As", end = " ")

	# decide sentiment as positive, negative and neutral
	if sentiment_dict['compound'] >= 0.05 :
		print("Positive")

	elif sentiment_dict['compound'] <= - 0.05 :
		print("Negative")

	else :
		print("Neutral")

def text_transformers_eval(df, rel_path = 'REL_PATH', transformer = f"cardiffnlp/twitter-roberta-base-sentiment",folder = 'model', file_name = 'transformes_metrics.csv', dir_file = 'REL_PATH' + '/model/model_metrics'):
  '''
  Objective: 
  ---

    Evaluete the transformers models without training

  args.
  ---

  ret.
  ---
  '''
  
  MODEL = transformer
  tokenizer = AutoTokenizer.from_pretrained(MODEL)
  
  model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
  model.save_pretrained(rel_path + '/'+folder + '/' + MODEL)
  
  texts = df['text']
  y_pred = []
  for text in texts:
    encoded_input = tokenizer(text, return_tensors='tf')
    output = model(encoded_input)
    scores = output[0][0].numpy()
    scores = softmax(scores)
    res = np.argmax(scores)
    if res==2:
      res=1
    y_pred.append(res)
  
  y_test = list(df['label'])
  metrics = eval_metrics(y_pred,y_test)

  model_str = str(model)[0:str(model).find('(')]
  dict4save(metrics, file_name, dir_file, addcols=True, cols='model', vals=model_str,sep=';')
  return model, tokenizer, metrics, y_pred


def text_transformer_prediction(df, model, tokenizer, rel_path = 'REL_PATH', folder = "/data/processed/" ,file = "prediction_transf_1.json",colab=True):
  '''
  Objective:
  ---

    Make predictions with the transformer model 

  args.
  ---
  ret.
  ---
  '''
  y_pred = []
  texts = df['text']
  for text in texts:
    encoded_input = tokenizer(text, return_tensors='tf')
    output = model(encoded_input)
    scores = output[0][0].numpy()
    scores = softmax(scores)
    res = np.argmax(scores)
    if res==2:
      res=1
    y_pred.append(res)

  df['prediction'] = y_pred 
  id = df['test_idx']
  
  nuwe_prediction_file(id,y_pred,rel_path,folder,file, colab)
  return df   
