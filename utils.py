def load_data(logfile=None):

    import datetime
    import time
    import numpy as np
    import csv
    from datetime import datetime
    from keras.preprocessing.sequence import pad_sequences

    vocabulary = set()

    csvfile = open(logfile, 'r')
    logreader = csv.reader(csvfile, delimiter=';')
    # logreader = csv.reader(csvfile, delimiter=',') # For Helpdesk, BPI12
    next(logreader, None)  # skip the headers

    lastcase = '' 
    casestarttime = None
    lasteventtime = None
    firstLine = True

    lines = [] #these are all the activity seq
    timeseqs = [] #time sequences (differences between two events)

    numcases = 0
    max_length = 0

    for row in logreader:
        #t = datetime.strptime(row[2], "%Y/%m/%d %H:%M:%S.%f") # For BPI12 Dataset
        # t = datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S") # For Helpdesk Dataset
        t = datetime.strptime(row[2], "%d.%m.%Y-%H:%M:%S")
        if row[0]!=lastcase:  #'lastcase' is to save the last executed case for the loop
            casestarttime = t
            lasteventtime = t
            lastcase = row[0]
            if not firstLine:
                lines.append(line)
                timeseqs.append(times)
                if len(line) > max_length:
                    max_length = len(line)
            line = []
            times = []
            numcases += 1

        vocabulary.add(row[1])
        line.append(row[1])
        timesincelastevent = t - lasteventtime
        timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds + timesincelastevent.microseconds/1000000
        # +1 avoid zero
        times.append(timediff+1)
        lasteventtime = t
        firstLine = False

    lines.append(line)
    timeseqs.append(times)

    vocabulary = {key: idx for idx, key in enumerate(vocabulary)}
    divisor = np.mean([item for sublist in timeseqs for item in sublist]) #average time between events
    numcases += 1
    print("Num cases: ", numcases)
    elems_per_fold = int(round(numcases/3))

    if len(line) > max_length:
        max_length = len(line)

    X = []
    X1 = []
    y = []
    y_t = []

    categorical_features_name = []
    categorical_features_time = []

    max_length = 0
    prefix_sizes = []
    seqs = 0
    vocab = set()

    count = 0

    for seq, time in zip(lines, timeseqs):
        code = []
        code.append(vocabulary[seq[0]])
        code1 = []
        code1.append(np.log(time[0]+1))

        vocab.add(seq[0])

        for i in range(1,len(seq)):
            prefix_sizes.append(len(code))

            if len(code)>max_length:
                max_length = len(code)

            # Building Activity Names and Time from Index for Explainability part
            sub_feature_name = []
            sub_feature_time = []
            vocabulary_clone = vocabulary.copy()
            for j in code[:]:
                
                for name, index in vocabulary_clone.items():
                    if index == j:
                        sub_feature_name.append(name)
                        sub_feature_time.append("Time corresponding to "+name)

            categorical_features_name.append(sub_feature_name)
            categorical_features_time.append(sub_feature_time)

            X.append(code[:])
            X1.append(code1[:])
            y.append(vocabulary[seq[i]])
            y_t.append(time[i]/divisor)

            code.append(vocabulary[seq[i]])
            code1.append(np.log(time[i]+1))
            seqs += 1

            vocab.add(seq[i])
            
    prefix_sizes = np.array(prefix_sizes)

    print (prefix_sizes)

    print("Num sequences:", seqs)
    print("Activities: ",vocab )
    vocab_size = len(vocab)

    X = np.array(X)
    X1 = np.array(X1)
    y = np.array(y)
    y_t = np.array(y_t)

    categorical_features_name = np.array(categorical_features_name)
    categorical_features_time = np.array(categorical_features_time)

    y_unique = np.unique(y)
    dict_y = {}
    i = 0
    for el in y_unique:
        dict_y[el] = i
        i += 1
    for i in range(len(y)):
        y[i] = dict_y[y[i]]
    y_unique = np.unique(y, return_counts=True)
    print("Classes: ", y_unique)
    n_classes = y_unique[0].shape[0]

    # Establishing vocabulary for classes by removing non-predicatable class from vocabulary (For Helpdesk Dataset)
    rebel = int
    vocabulary_class = {}

    # Finding where the class occurs which is not to be predicted
    for key,value in enumerate(dict_y):
        if (key!=value):
            rebel = key
            break

    # deleting that class from dictionary
    for name in vocabulary_clone.copy():
        if (vocabulary_clone[name] == rebel):
            vocabulary_clone.pop(name)

    vocabulary_class = vocabulary_clone.copy()
    for index,name in enumerate(vocabulary_class.copy()):
            vocabulary_class[name] = index

    # padding
    padded_X = pad_sequences(X, maxlen=max_length, padding='pre', dtype='float64')
    padded_X1 = pad_sequences(X1, maxlen=max_length, padding='pre', dtype='float64')
    padded_features = pad_sequences(categorical_features_name, maxlen=max_length, padding='pre', dtype=object, value="Zero Padded Feature") #Padding feature name for Padded feature
    padded_features_time = pad_sequences(categorical_features_time, maxlen=max_length, padding='pre', dtype=object, value="Zero Padded Feature") #Padding feature time for Padded feature

    return ( (padded_X, padded_X1), (y, y_t), vocab_size, max_length, n_classes, divisor, prefix_sizes, vocabulary, vocabulary_class, padded_features, padded_features_time)
