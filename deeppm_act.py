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
from tensorflow.keras.utils import Sequence

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

import json #For dictionary to String conversion

import matplotlib.pyplot as plt
import plotly.express as px


#Imports for Explainabaility Part

import sklearn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import itertools

import warnings
warnings.filterwarnings('ignore')

from interpret.blackbox import LimeTabular


input_length_lime = int

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

    print (params)
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


def classification_matrix(y_a_test, preds_a, n_classes):
    cm = confusion_matrix(y_a_test, preds_a)
    classes = [i for i in range(n_classes-1)]#specific to dataset used
    fig = plt.figure(figsize=(11,11))
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

    plt.savefig('confusion_matrix.png')
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
 vocabulary_class,
 padded_features,
 padded_features_time,
 categorical_features_name) = load_data(logfile)

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
    classification_matrix(y_a_test, preds_a, best_params['n_classes'])

    outfile.write(np.array2string(confusion_matrix(y_a_test, preds_a), separator=", "))
    
    outfile.flush()


print("\n\nFinal Brier score: ", final_brier_scores, file=outfile)
print("Final Accuracy score: ", final_accuracy_scores, file=outfile)

outfile.close()

################################################################************************#################################################################


#XAI Part Implementation

print("\n\nFinal Brier score: ", final_brier_scores)
print("Final Accuracy score: ", final_accuracy_scores)


# Dynamic feature extraction for instances

def make_features(i,j):
    temp = []
    temp = np.concatenate((padded_features[i:j], padded_features_time[i:j]), axis=None)
    temp1 = list (temp)
    return temp1

# Parsing the inputs the way LIME Explainer expects

class_size = best_params['input_length']
def lime_prob(input_data):
    new_p = input_data[:,0:class_size]
    new_p1 = input_data[:,class_size:]
    preds = best_model.predict([new_p,new_p1])
    preds = np.argmax(preds, axis=1)
    return preds

def shap_prob(input_data):
    new_p = input_data[:,0:class_size]
    new_p1 = input_data[:,class_size:]
    preds = best_model.predict([new_p,new_p1])
    return preds

#Converting dictionary to string
result = json.dumps(vocabulary_class) 
result = "Class Labels: " + result

def legend(i,j):
    temp = str(categorical_features_name[i:j])
    temp.replace('list', '')
    temp = temp[6 :-2 : ] #removing list keyword from string.
    temp = "Event Trace: " + temp
    return temp


merged_array_train = np.hstack((X_a, X_t))
y_a = np.argmax(y_a, axis=1)

# Generic Instance Explainability Recursive function for traces

def generate_interpretability(trace_start,trace_end):
    if trace_start == trace_end:
        return 0
    else: 
        plot_name  = str(trace_start)
        merged_array_test_gen = np.hstack((X_a[trace_start:trace_start+1], X_t[trace_start:trace_start+1]))
        lime_gen = LimeTabular(predict_fn=lime_prob, data=merged_array_train, feature_names=make_features(trace_start,trace_start+1))
        lime_local_gen = lime_gen.explain_local(merged_array_test_gen, y_a[trace_start:trace_start+1])
        plot = lime_local_gen.visualize(0).update_layout(xaxis_title=legend(trace_start, trace_start+1), font=dict(size=4))
        plot1 = lime_local_gen.visualize(0)
        plot.write_html(plot_name+".html")
        plot.write_image(plot_name+".pdf")
        plot.write_image(plot_name+".pdf")
        plot1.write_html(plot_name+"1"+".html")
        plot1.write_image(plot_name+"1"+".pdf")
        plot.write_image(plot_name+".pdf")
        trace_start = trace_start + 1
        return generate_interpretability (trace_start, trace_end)


# Test Case 1
generate_interpretability (7,10)

#Test Case 2
generate_interpretability (5474,5480)

#Test Case 3
generate_interpretability (10,13)


# Printing activties and their corresponding labels
print ("\n Different activities with their corresponding labels can be interpreted from this legend: \n")
for key, value in enumerate(vocabulary_class):
    print (key,"---------", value)

#Testing purpose

# for i in range(15):
#     print ("Sample is this: ", X_a[i:i+1], X_t[i:i+1])
#     print ("Ground truth is this: ", np.argmax(y_a[i:i+1], axis=1))
#     pred = best_model.predict([X_a[i:i+1], X_t[i:i+1]])
#     pred = np.argmax(pred, axis=1)
#     print ("Predicted is this: ", pred)


# Test Case Formulation
# Test Case 1

# # multiple instances to test with lime
# merged_array_test_lime = np.hstack((X_a[7:8], X_t[7:8]))
# merged_array_test_lime1 = np.hstack((X_a[8:9], X_t[8:9]))
# merged_array_test_lime2 = np.hstack((X_a[9:10], X_t[9:10]))

# #Using Lime Tabular and applying local explanation with dynamic features
# lime = LimeTabular(predict_fn=lime_prob, data=merged_array_train, feature_names=make_features(7,8))
# lime_local = lime.explain_local(merged_array_test_lime, y_a[7:8], name='LIME')
# lime_local.visualize(0).write_html("lime.html")

# lime1 = LimeTabular(predict_fn=lime_prob, data=merged_array_train, feature_names=make_features(8,9))
# lime_local1 = lime1.explain_local(merged_array_test_lime1, y_a[8:9], name='LIME1')
# lime_local1.visualize(0).write_html("lime1.html") 

# lime2 = LimeTabular(predict_fn=lime_prob, data=merged_array_train, feature_names=make_features(9,10))
# lime_local2 = lime2.explain_local(merged_array_test_lime2, y_a[9:10], name='LIME2')
# lime_local2.visualize(0).write_html("lime2.html") 


# # Test Case 2

# # multiple instances to test with lime
# merged_array_test_lime9 = np.hstack((X_a[5474:5475], X_t[5474:5475]))
# merged_array_test_lime10 = np.hstack((X_a[5475:5476], X_t[5475:5476]))
# merged_array_test_lime11 = np.hstack((X_a[5476:5477], X_t[5476:5477]))
# merged_array_test_lime12 = np.hstack((X_a[5477:5478], X_t[5477:5478]))
# merged_array_test_lime13 = np.hstack((X_a[5478:5479], X_t[5478:5479]))
# merged_array_test_lime14 = np.hstack((X_a[5479:5480], X_t[5479:5480]))


# #Using Lime Tabular and applying local explanation with dynamic features
# lime9 = LimeTabular(predict_fn=lime_prob, data=merged_array_train, feature_names=make_features(5474,5475))
# lime_local9 = lime9.explain_local(merged_array_test_lime9, y_a[5474:5475], name='LIME9')
# lime_local9.visualize(0).write_html("lime9.html") 

# lime10 = LimeTabular(predict_fn=lime_prob, data=merged_array_train, feature_names=make_features(5475,5476))
# lime_local = lime10.explain_local(merged_array_test_lime10, y_a[5475:5476], name='LIME10')
# lime_local.visualize(0).write_html("lime10.html")

# lime11 = LimeTabular(predict_fn=lime_prob, data=merged_array_train, feature_names=make_features(5476,5477))
# lime_local11 = lime11.explain_local(merged_array_test_lime11, y_a[5476:5477], name='LIME11')
# lime_local11.visualize(0).write_html("lime11.html") 

# lime12 = LimeTabular(predict_fn=lime_prob, data=merged_array_train, feature_names=make_features(5477,5478))
# lime_local12 = lime12.explain_local(merged_array_test_lime12, y_a[5477:5478], name='LIME12')
# lime_local12.visualize(0).write_html("lime12.html") 

# lime13 = LimeTabular(predict_fn=lime_prob, data=merged_array_train, feature_names=make_features(5478,5479))
# lime_local13 = lime13.explain_local(merged_array_test_lime13, y_a[5478:5479], name='LIME13')
# lime_local13.visualize(0).write_html("lime13.html") 

# lime14 = LimeTabular(predict_fn=lime_prob, data=merged_array_train, feature_names=make_features(5479,5480))
# lime_local14 = lime14.explain_local(merged_array_test_lime14, y_a[5479:5480], name='LIME14')
# lime_local14.visualize(0).write_html("lime14.html") 



# SHAP Part

# from interpret.blackbox import ShapKernel

# merged_array_test_shap = np.hstack((X_a_test[0:10], X_t_test[0:10]))
# background_val1 = np.median(merged_array, axis=0).reshape(1, -1) 
# background_val = shap.sample(merged_array_train,300)

# # use Kernel SHAP to explain test set predictions
# explainer = shap.KernelExplainer(shap_prob, background_val)
# shap_values = explainer.shap_values(merged_array_test_shap)
# shap.summary_plot(shap_values, merged_array_test_shap, plot_type="bar", feature_names=features_name, show=false) # see features name
# plt.savefig('Shap_Bar.png',bbox_inches='tight')
# plt.clf() #Clears the Plot space for next plots 
# plt.cla()



# For Backup
# def generate_interpretability(trace_start,trace_end):
#     if trace_start == trace_end:
#         return 0
#     else: 
#         plot_name  = str(trace_start)
#         merged_array_test_gen = np.hstack((X_a[trace_start:trace_start+1], X_t[trace_start:trace_start+1]))
#         lime_gen = LimeTabular(predict_fn=lime_prob, data=merged_array_train, feature_names=make_features(trace_start,trace_start+1))
#         lime_local_gen = lime_gen.explain_local(merged_array_test_gen, y_a[trace_start:trace_start+1])
#         plot = lime_local_gen.visualize(0).update_layout(xaxis_title=result, font=dict(size=4), yaxis_title=legend(trace_start, trace_start+1))
#         plot1 = lime_local_gen.visualize(0)
#         plot.write_html(plot_name+".html")
#         plot.write_image(plot_name+".pdf")
#         plot.write_image(plot_name+".pdf")
#         plot1.write_html(plot_name+"1"+".html")
#         plot1.write_image(plot_name+"1"+".pdf")
#         plot.write_image(plot_name+".pdf")
#         trace_start = trace_start + 1
#         return generate_interpretability (trace_start, trace_end)