"""
    Bag of Words object

    example:

        import load
        X_train, y_train, X_test, y_test = load(True, False, False, False)
        bow = BOW()
        bow.fit(X_train)
        X_bow = bow.transform(X_train)

    X_bow is a numpy array with shape = (len(X_train),len(bow.words)). 
    bow.words is accumulated by ./data/words-dict.pickle, which is generated anew in bow.fit(X_train) if it does not already exist

"""
import numpy as np
import pickle

class BOW:
    def __init__(self):
        pass

    def fit(self,X,y=None,new_pickle=False):
        # collect word dictionary (string keys with index values) from ./data/word-dict.pickle, or create it. If new_pickle is True, ./data/word-dict.pickle will be created or overwritten 
        try:
            if not new_pickle:
                self.words = pickle.load( open( "./data/word-dict.pickle", "rb" ))
        except:
            new_pickle = True

        if new_pickle:
            words = []
            for item in X:
                header,body = item[0],item[1]
                words += header + body
            words = list(set(words))
            words_dict = {}
            for i,w in enumerate(words):
                words_dict[w] = i
            pickle.dump(words_dict,open("./data/word-dict.pickle",'wb'))
            self.words = words_dict

        return self

    def transform(self,X):
        '''
        Transform list of tuples X = (headline, article) into Bag of Words matrix 

        inputs:
            - X = [(headline1,article1), (headline2,article2), ...] 

        outputs:
            - bowX = ndarray of shape (len(X),len(self.words)), each index of a row corresponds to the specific word in self.words and contains the number of instances of that word within the headline-article pair.

        '''
        bowX = np.zeros((len(X),len(self.words)))

        for i, x in enumerate(X):
            for word in x[0]:
                bowX[i,:] += self.bowVector(word)
            for word in x[1]:
                bowX[i,:] += self.bowVector(word)

        return bowX

    def bowVector(self,word):
        # defines a vector of length len(self.words) that is zero-valued everywhere but where the input word string is found
        bow = np.zeros(len(self.words))
        try:
            bow[self.words[word]] = 1
        except:
            print('Word not found.')
        return bow


if __name__=='__main__':
    import load
    X_train, y_train, X_test, y_test = load.load(True,False,False,False)
    bow = BOW()
    bow.fit(X_train,new_pickle=False)
    print(len(linear.words), 'unique words')
    bow.transform(X_train[0:10])
