import numpy as np
seed = 123
np.random.seed(seed)
from tensorflow import set_random_seed
set_random_seed(seed)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Reshape, MaxPooling1D, Flatten, Dense, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from utils import load_data
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss

from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample
import sys
import csv 
from time import perf_counter
import time

import matplotlib.pyplot as plt


#Imports for Explainabaility Part

import sklearn
import lime
import lime.lime_tabular

from tensorflow.keras.utils import Sequence


from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance

import warnings
warnings.filterwarnings('ignore')


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import itertools




class DataGenerator(Sequence):
    def __init__(self, features, labels, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.labels = labels
        self.X_a = features[0]
        self.X_t = features[1]
        self.y_a = labels[0]
        self.y_t = labels[1]
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of steps per epoch'
        return int(np.floor(self.X_a.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.X_a.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_a = np.empty((self.batch_size, self.X_a.shape[1]))
        X_t = np.empty((self.batch_size, self.X_t.shape[1]))
        y_a = np.empty((self.batch_size, self.y_a.shape[1]), dtype=int)
        y_t = np.empty((self.batch_size))
           
                       
        # Generate data
        for i, ID in enumerate(indexes):
            # Store sample
            X_a[i] = self.X_a[ID]
            X_t[i] = self.X_t[ID]

            # Store class
            y_a[i] = self.y_a[ID]
            y_t[i] = self.y_t[ID]
                       

        return [X_a, X_t], {'output_a':y_a, 'output_t':y_t}
    

def get_model(input_length=10, n_filters=3, vocab_size=10, n_classes=9, embedding_size=5, n_modules=5, model_type='ACT', learning_rate=0.002):
    #inception model

    inputs = []
    for i in range(2):
        inputs.append(Input(shape=(input_length,)))

    inputs_ = []
    for i in range(2):
        if (i==0):
            a = Embedding(vocab_size, embedding_size, input_length=input_length)(inputs[0])
            inputs_.append(Embedding(vocab_size, embedding_size, input_length=input_length)(inputs[i]))
        else:
            inputs_.append(Reshape((input_length, 1))(inputs[i]))

    filters_inputs = Concatenate(axis=2)(inputs_)

    for m in range(n_modules):
        filters = []
        for i in range(n_filters):
            filters.append(Conv1D(filters=32, strides=1, kernel_size=1+i, activation='relu', padding='same')(filters_inputs))
        filters.append(MaxPooling1D(pool_size=3, strides=1, padding='same')(filters_inputs))
        filters_inputs = Concatenate(axis=2)(filters)
        #filters_inputs = Dropout(0.1)(filters_inputs)

    #pool = GlobalAveragePooling1D()(filters_inputs)
    pool = GlobalMaxPooling1D()(filters_inputs)
    #pool = Flatten()(filters_inputs)

    #pool = Dense(64, activation='relu')(pool)


    optimizer = Adam(lr=learning_rate)

    if (model_type == 'BOTH'):
        out_a = Dense(n_classes, activation='softmax', name='output_a')(pool)
        out_t = Dense(1, activation='linear', name='output_t')(pool)
        model = Model(inputs=inputs, outputs=[out_a, out_t])
        model.compile(optimizer=optimizer, loss={'output_a':'categorical_crossentropy', 'output_t':'mae'})
    else:
        if (model_type=='ACT'):
            out = Dense(n_classes, activation='softmax')(pool)
            model = Model(inputs=inputs, outputs=out)
            model.compile(optimizer=optimizer, loss='mse',metrics=['acc'])
        elif (model_type=='TIME'):
            out = Dense(1, activation='linear')(pool)
            model = Model(inputs=inputs, outputs=out)
            model.compile(optimizer=optimizer, loss='mae')

    model.summary()

    return model


def fit_and_score(params):
    print(params)
    start_time = perf_counter()

    model = get_model(input_length=params['input_length'], vocab_size=params['vocab_size'], n_classes=params['n_classes'], model_type=params['model_type'],
                      learning_rate=params['learning_rate'], embedding_size=params['embedding_size'], n_modules=params['n_modules'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    """
    num_train = int(X_a_train.shape[0]*0.8)

    X_a_train1 = X_a_train[:num_train]
    X_a_val1 = X_a_train[num_train:]
    X_t_train1 = X_t_train[:num_train]
    X_t_val1 = X_t_train[num_train:]

    y_a_train1 = y_a_train[:num_train]
    y_a_val1 = y_a_train[num_train:]
    y_t_train1 = y_t_train[:num_train]
    y_t_val1 = y_t_train[num_train:]

    # Generators
    train_generator = DataGenerator([X_a_train1,X_t_train1], [y_a_train1,y_t_train1], batch_size=2**params['batch_size'])
    val_generator = DataGenerator([X_a_val1,X_t_val1], [y_a_val1,y_t_val1], batch_size=2**params['batch_size'])
    """
   
    if (params['model_type'] == 'ACT'):
        h = model.fit([X_a_train, X_t_train],
                      y_a_train, epochs=200, verbose=0, 
                      validation_split=0.2, callbacks=[early_stopping], batch_size=2**params['batch_size'])
    elif (params['model_type'] == 'TIME'):
        h = model.fit([X_a_train, X_t],
                      y_t_train, epochs=200, 
                      validation_split=0.2, callbacks=[early_stopping], batch_size=2**params['batch_size'])
    else:
         h = model.fit([X_a_train, X_t_train],
                      {'output_a':y_a_train, 'output_t':y_t_train}, epochs=200, verbose=0,
                      validation_split=0.2, callbacks=[early_stopping], batch_size=2**params['batch_size'])
#        h = model.fit_generator(generator=train_generator, validation_data=val_generator, use_multiprocessing=True, workers=8, epochs=200, callbacks=[early_stopping], max_queue_size=10000, verbose=0)


    scores = [h.history['val_loss'][epoch] for epoch in range(len(h.history['loss']))]
    score = min(scores)
    print(score)

    global best_score, best_model, best_time, best_numparameters
    end_time = perf_counter()

    if best_score > score:
        best_score = score
        best_model = model
        best_numparameters = model.count_params()
        best_time = end_time - start_time

    return {'loss': score, 'status': STATUS_OK,  'n_epochs':  len(h.history['loss']), 'n_params':model.count_params(), 'time':end_time - start_time}


def classification_matrix(y_a_test, preds_a):
    cm = confusion_matrix(y_a_test, preds_a)
    classes = ['0', '1', '2', '3','4','5','6','7'] #specific to dataset used
    fig = plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title('Confusion matrix for Classes')
    plt.colorbar()
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    normalize = False
    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig('classification_matrix.png')
    plt.clf()
    plt.cla()


logfile = sys.argv[1]
model_type = sys.argv[2]
output_file = sys.argv[3]

current_time = time.strftime("%d.%m.%y-%H.%M", time.localtime())
outfile = open(output_file, 'w')

outfile.write("Starting time: %s\n" % current_time)

((X_a, X_t),
 (y_a, y_t),
 vocab_size,
 max_length,
 n_classes,
 divisor,
 prefix_sizes,
 vocabulary,
 padded_features) = load_data(logfile)

emb_size = (vocab_size + 1 ) // 2 # --> ceil(vocab_size/2)

# normalizing times
X_t = X_t / np.max(X_t)
# categorical output
y_a = to_categorical(y_a)


#n_iter = 20 #Commented for faster compilation at the expense of training weights
n_iter = 1

space = {'input_length':max_length, 'vocab_size':vocab_size, 'n_classes':n_classes, 'model_type':model_type, 'embedding_size':emb_size,
         'n_modules':hp.choice('n_modules', [1,2,3]),
         'batch_size': hp.choice('batch_size', [9,10]),
         'learning_rate': hp.loguniform("learning_rate", np.log(0.00001), np.log(0.01))}



final_brier_scores = []
final_accuracy_scores = []
final_mae_scores = []
final_mse_scores = []
#for f in range(3): #to run for once rather than 3 and averaging the output returns //k fold cross validation
for f in range(1):    
    print("Fold", f)
    outfile.write("\nFold: %d" % f)
    # split into train and test set
    p = np.random.RandomState(seed=seed+f).permutation(X_a.shape[0])
    elems_per_fold = int(round(X_a.shape[0]/3))

    X_a_train = X_a[p[:2*elems_per_fold]]
    X_t_train = X_t[p[:2*elems_per_fold]]
    X_a_test = X_a[p[2*elems_per_fold:]]
    X_t_test = X_t[p[2*elems_per_fold:]]
    y_a_train = y_a[p[:2*elems_per_fold]]
    y_a_test = y_a[p[2*elems_per_fold:]]
    y_t_train = y_t[p[:2*elems_per_fold]]
    y_t_test = y_t[p[2*elems_per_fold:]]

    # model selection
    print('Starting model selection...')
    best_score = np.inf
    best_model = None
    best_time = 0
    best_numparameters = 0

    trials = Trials()
    best = fmin(fit_and_score, space, algo=tpe.suggest, max_evals=n_iter, trials=trials, rstate= np.random.RandomState(seed+f))
    best_params = hyperopt.space_eval(space, best)

    outfile.write("\nHyperopt trials")
    outfile.write("\ntid,loss,learning_rate,n_modules,batch_size,time,n_epochs,n_params,perf_time")
    for trial in trials.trials:
        outfile.write("\n%d,%f,%f,%d,%d,%s,%d,%d,%f"%(trial['tid'],
                                                trial['result']['loss'],
                                                trial['misc']['vals']['learning_rate'][0],
                                                int(trial['misc']['vals']['n_modules'][0]+1),
                                                trial['misc']['vals']['batch_size'][0]+7,
                                                (trial['refresh_time']-trial['book_time']).total_seconds(),
                                                trial['result']['n_epochs'],
                                                trial['result']['n_params'],
                                                trial['result']['time']))

    outfile.write("\n\nBest parameters:")
    print(best_params, file=outfile)
    outfile.write("\nModel parameters: %d" % best_numparameters)
    outfile.write('\nBest Time taken: %f'%best_time)

    # evaluate
    print('Evaluating final model...')
    preds_a = best_model.predict([X_a_test,X_t_test])
    brier_score = np.mean(list(map(lambda x: brier_score_loss(y_a_test[x],preds_a[x]),[i[0] for i in enumerate(y_a_test)])))

    y_a_test = np.argmax(y_a_test, axis=1)
    preds_a = np.argmax(preds_a, axis=1)

    outfile.write("\nBrier score: %f" % brier_score)
    final_brier_scores.append(brier_score)

    accuracy = accuracy_score(y_a_test, preds_a)
    outfile.write("\nAccuracy: %f" % accuracy)
    final_accuracy_scores.append(accuracy)

    print(classification_report(y_a_test, preds_a))
    classification_matrix(y_a_test, preds_a)

    outfile.write(np.array2string(confusion_matrix(y_a_test, preds_a), separator=", "))
    
    outfile.flush()


print("\n\nFinal Brier score: ", final_brier_scores, file=outfile)
print("Final Accuracy score: ", final_accuracy_scores, file=outfile)

outfile.close()

#XAI Part

print("\n\nFinal Brier score: ", final_brier_scores)
print("Final Accuracy score: ", final_accuracy_scores)

# Lining-up the feature names

# feature_names_activities = "Activities"
# feature_names_time = "Time Span between Activities"

# features_name_helpd = ['Activity at Trace Position 13','Activity at Trace Position 12','Activity at Trace Position 11','Activity at Trace Position 10','Activity at Trace Position 9','Activity at Trace Position 8','Activity at Trace Position 7','Activity at Trace Position 6','Activity at Trace Position 5','Activity at Trace Position 4','Activity at Trace Position 3','Activity at Trace Position 2','Activity at Trace Position 1',
#                 'Time Corresponding to ATP 13','Time Corresponding to ATP 12', 'Time Corresponding to ATP 11', 'Time Corresponding to ATP 10', 'Time Corresponding to ATP 9', 'Time Corresponding to ATP 8', 'Time Corresponding to ATP 7', 'Time Corresponding to ATP 6', 'Time Corresponding to ATP 5', 'Time Corresponding to ATP 4', 'Time Corresponding to ATP 3', 'Time Corresponding to ATP 2', 'Time Corresponding to ATP 1']


features_name = ['Activity at Trace Position 14', 'Activity at Trace Position 13','Activity at Trace Position 12','Activity at Trace Position 11','Activity at Trace Position 10','Activity at Trace Position 9','Activity at Trace Position 8','Activity at Trace Position 7','Activity at Trace Position 6','Activity at Trace Position 5','Activity at Trace Position 4','Activity at Trace Position 3','Activity at Trace Position 2','Activity at Trace Position 1',
                'Time Corresponding to ATP 1', 'Time Corresponding to ATP 13','Time Corresponding to ATP 12', 'Time Corresponding to ATP 11', 'Time Corresponding to ATP 10', 'Time Corresponding to ATP 9', 'Time Corresponding to ATP 8', 'Time Corresponding to ATP 7', 'Time Corresponding to ATP 6', 'Time Corresponding to ATP 5', 'Time Corresponding to ATP 4', 'Time Corresponding to ATP 3', 'Time Corresponding to ATP 2', 'Time Corresponding to ATP 1']


# myList = list()
# myList.append(feature_names_activities)
# myList.append(feature_names_time)
# features_name = np.array(myList)
# print(features_name)

# Create the LIME Explainer


input1 = X_a_train
input2 = X_t_train

# print (X_a_train.shape)
# print (X_t_train.shape)

# merged_array = np.stack([input1,input2], axis=1)
# # input_lime = Concatenate()([input1, input2])

# # # print (merged_array.shape)
# merged_array= merged_array.transpose(0, 2, 1)
# # # merged_array.reshape (6604,13,13)
# print (merged_array.shape)

# print (X_a_test[0].shape)
# print (X_t_test[0].shape)


# # print (X_a_test[0])
# # print (X_t_test[0])

# # print (shap_input)

# def f_wrapper(X):
#     #input_list = X.tolist()
#     #print input_list
#     return best_model.predict(X).flatten()

    
# explainer = lime.lime_tabular.RecurrentTabularExplainer(training_data=merged_array, training_labels=y_a_train, feature_names = features_name)

# merged_array_test = np.stack([X_a_test[0:100], X_t_test[0:100]], axis=1)
# merged_array_test = merged_array_test.transpose(0, 2, 1)
# print(merged_array_test.shape)

# merged_array_test = np.hstack((X_a_test[0:100], X_t_test[0:100]))
# #merged_array_test= merged_array_test.transpose(1,0)
# print(merged_array_test.shape)

# print (X_a_test[0:1].shape)
# print (X_t_test[0:1].shape)

# A =  best_model.predict((X_a_test[0:1],X_t_test[0:1]))
# print (A)
# print (np.argmax(A))


# merged_array_test1= np.concatenate((X_a_test[0], X_t_test[0]))
# print(merged_array_test1.shape)

# X = np.array([X_a_test[0:1], X_t_test[0:1]])

# print (X.shape)
# input_list = X.tolist()
# print (input_list)


def classifier_pred(input_data):
    print (input_data.shape)
    act = input_data[:,0:13]
    tim = input_data[:,13:26]
    print (act.shape)
    print (tim.shape)
    preds = best_model.predict([act,tim])
    print (preds)
    return preds

# exp = explainer.explain_instance(merged_array_test, classifier_pred, num_features=2)
# exp.show_in_notebook(show_all=True)


# hiya = best_model.get_weights()
# print (hiya)

# import shap

# # print the JS visualization code to the notebook
# shap.initjs()

# # we use the first 100 training examples as our background dataset to integrate over
# explainer = shap.DeepExplainer(best_model, data=[X_a_train, X_t_train])

# print(X_a_test[0])
# print(X_t_test[0])
# temp = np.reshape(X_t_test[0],X_a_test[0].shape)
# print(temp)


# shap_values = explainer.shap_values([X_a_test[0:1], X_t_test[0:1]])

# # explain the first 10 predictions
# # explaining each prediction requires 2 * background dataset size runs
# #shap_values = explainer.shap_values([X_a_test, X_t_test])
# shap.summary_plot(shap_values, [X_a_test[0:1], X_t_test[0:1]] , feature_names=features_name, plot_type="bar")


# # # perm = PermutationImportance(best_model, scoring='accuracy', random_state=1).fit([X_a_train, X_t_train], y_a_train)
# # # eli5.show_weights(perm, feature_names = features_name)


# # # from alibi.explainers import KernelShap
# # # import skimage

# # # predict_fn = lambda x: clf.predict_proba(x)
# # # explainer = KernelShap(classifier_pred, link='logit', feature_names=features_name)

# # # explainer.fit(merged_array_test)
# # # explanation = explainer.explain(X)



import shap

# # Too many input data - use a random slice
# # rather than use the whole training set to estimate expected values, we summarize with
# # a set of weighted kmeans, each weighted by the number of points they represent.
# # X_train_summary = shap.kmeans([X_a_train,X_t_train], 20)

# # Compute Shap values
# explainer = shap.DeepExplainer(best_model,[X_a_train, X_t_train])

# # Make plot with combined shap values
# # The training set is too big so let's sample it. We get enough point to draw conclusions
# # X_train_sample = [X_a_train,X_t_train].sample(400)
# shap_values  = explainer.shap_values([X_a_test[0:10], X_t_test[0:10]])
# shap.summary_plot(shap_values, [X_a_test[0:10], X_t_test[0:10]] )

# from sklearn.inspection import plot_partial_dependence

# my_plots = plot_partial_dependence(best_model.predict([X_a_train, X_t_train]),merged_array, features=[0,1])




    

def lime_prob(input_data):
    # print(input_data.shape)
    new_p = input_data[:,0:14]
    new_p1 = input_data[:,14:28]
    # print (new_p.shape)
    # print (new_p1.shape)
    preds = best_model.predict([new_p,new_p1])
    preds = np.argmax(preds, axis=1)
    return preds


def shap_prob(input_data):
    # print(input_data.shape)
    new_p = input_data[:,0:14]
    new_p1 = input_data[:,14:28]
    # print (new_p.shape)
    # print (new_p1.shape)
    preds = best_model.predict([new_p,new_p1])
    return preds




from interpret.blackbox import PartialDependence
from interpret import show

from interpret import preserve
from interpret.provider import InlineProvider
from interpret import set_visualize_provider

set_visualize_provider(InlineProvider())

# print (y_a_train[1])
# print (y_a_train[2])

merged_array_train = np.hstack((X_a_train, X_t_train))
#merged_array = merged_array.transpose(1,0)
# print(merged_array.shape)

merged_array_test_shap = np.hstack((X_a_test[0:10], X_t_test[0:10]))

# multiple instances to test with lime
merged_array_test_lime = np.hstack((X_a_test[0:1], X_t_test[0:1]))
merged_array_test_lime1 = np.hstack((X_a_test[15:16], X_t_test[15:16]))
merged_array_test_lime2 = np.hstack((X_a_test[27:28], X_t_test[27:28]))
merged_array_test_lime3 = np.hstack((X_a_test[3:4], X_t_test[3:4]))



# Making categorical feature array


# See replacement for pad 0 as it contradicts with activity with 0 label


print (X_a[0:15])
print (padded_features[0:15])






#Next, merge with t array, also see dynamically creating features for variable length trace

#Testing purpose

# for i in range(15):
#     print ("Sample is this: ", X_a[i:i+1], X_t[i:i+1])
#     print ("Ground truth is this: ", np.argmax(y_a[i:i+1], axis=1))
#     pred = best_model.predict([X_a[i:i+1], X_t[i:i+1]])
#     pred = np.argmax(pred, axis=1)
#     print ("Predicted is this: ", pred)

    


# print (X_a_test[15:16], X_t_test[15:16])

# print (y_a_test[0:1])
# print (y_a_test[15:16])



#merged_array_test_small = merged_array_test_small.transpose(1,0)
# print (merged_array_test_small.shape)

from interpret.blackbox import LimeTabular

#Blackbox explainers need a predict function, and optionally a dataset
lime = LimeTabular(predict_fn=lime_prob, data=merged_array_train, feature_names=features_name)

#Pick the instances to explain, optionally pass in labels if you have them
lime_local = lime.explain_local(merged_array_test_lime, y_a_test[0:1], name='LIME')
lime_local1 = lime.explain_local(merged_array_test_lime1, y_a_test[15:16], name='LIME')
lime_local2 = lime.explain_local(merged_array_test_lime2, y_a_test[27:28], name='LIME')
lime_local3 = lime.explain_local(merged_array_test_lime3, y_a_test[3:4], name='LIME')

lime_local.visualize(0).write_html("lime.html")
lime_local1.visualize(0).write_html("lime1.html") 
lime_local2.visualize(0).write_html("lime2.html")  
lime_local3.visualize(0).write_html("lime3.html")  


# from interpret.blackbox import ShapKernel

#background_val1 = np.median(merged_array, axis=0).reshape(1, -1) 
background_val = shap.sample(merged_array_train,300)

print (vocabulary) #Getting the vocabulary hence





# # use Kernel SHAP to explain test set predictions
# explainer = shap.KernelExplainer(shap_prob, background_val)
# shap_values = explainer.shap_values(merged_array_test_shap)
# shap.summary_plot(shap_values, merged_array_test_shap, plot_type="bar", feature_names=features_name, show=False)
# plt.savefig('Shap_Bar.png',bbox_inches='tight')
# plt.clf() #Clears the Plot space for next plots 
# plt.cla()



# SHAP Dependance Plot Implementation
# explainer = shap.KernelExplainer(pdp_prob, background_val1)
# shap_values = explainer.shap_values(merged_array_test)
# shap.dependence_plot(0, shap_values[0], merged_array_test)



# PDP Implementation
# from interpret import set_show_addr, get_show_addr

# pdp = PartialDependence(predict_fn=pdp_prob, data=merged_array_test)
# pdp_global = pdp.explain_global(name='Partial Dependence')
# set_show_addr(('127.0.0.1', 7001))
# show(pdp_global)
# plt.show()

# preserve(pdp_global, 4, file_name='global-age-graph.html')

# pdp_global.visualize(0)  # visualizes the 20th featur
# pdp_global.visualize(0).write_html("graph0.html")  # can also pass in a full filepath here

# pdp_global.visualize(1).write_html("graph1.html")  # can also pass in a full filepath here
# pdp_global.visualize(2).write_html("graph2.html")  # can also pass in a full filepath here
# pdp_global.visualize(3).write_html("graph3.html")  # can also pass in a full filepath here
# pdp_global.visualize(13).write_html("graph13.html")  # can also pass in a full filepath here
# pdp_global.visualize(15).write_html("graph15.html")  # can also pass in a full filepath here