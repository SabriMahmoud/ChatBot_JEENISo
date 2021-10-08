import json
import nltk
import numpy
import random
import tensorflow
import tflearn
import pickle
nltk.download('punkt')






def clean_data() :
    try:
        with open("data.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)
        return(words,labels,training,output)
    except:
        words = []
        labels = []
        docs_x = []
        docs_y = []

        for it in data["intents"]:
            for ptrn in it["patterns"]:
                wrds = nltk.word_tokenize(ptrn)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(it["tag"])

            if it["tag"] not in labels:
                labels.append(it["tag"])

        words = [w.lower() for w in words if w != "?"]
        words = sorted(list(set(words)))

        labels = sorted(labels)

        training = []
        output = []

        out_empty = [0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_x):
            cell = []

            wrds = [w.lower()for w in doc]

            for w in words:
                if w in wrds:
                    cell.append(1)
                else:
                    cell.append(0)

            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1

            training.append(cell)
            output.append(output_row)


        training = numpy.array(training)
        output = numpy.array(output)

        with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)
        return(words,labels,training,output)


def create_model(number_of_neurons = 8):
    tensorflow.compat.v1.reset_default_graph()

    _,_,training,output = clean_data()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, number_of_neurons)
    net = tflearn.fully_connected(net, number_of_neurons)
    net = tflearn.fully_connected(net, number_of_neurons)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

    try:
        model.load("model.tflearn")
    except:
        model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        model.save("model.tflearn")
    return model

def cell_of_words(s, words):
    cell = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [word.lower() for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                cell[i] = 1

    return numpy.array(cell)


def JuniorChatBot(model):
    
    while True:

        
        words,labels,_,_ = clean_data()
        
        inp = input("parlez :")
        if inp.lower() == "quit":
            break

        results = model.predict([cell_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if (results[results_index] > 0.7):
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))
        else:
            print("Pouvez-vous reformuler la question ?")

if __name__ == "__main__" :
    with open('intents.json') as file:
        data = json.load(file)
    model = create_model(number_of_neurons=16)
    print("Salutation  comment je peux vous aider ? ")
    JuniorChatBot(model)
