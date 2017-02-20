import csv
from pickle import load, dump

#dump(x_train, open("trainset.pckl", "wb" ))

def refresh():
    with open('x_train.csv', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        x_train = []
        for row in spamreader:
            try:
                x_train.append([float(item) for item in row])
            except:
                pass

    with open('y_train.csv', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        y_train = []
        for row in spamreader:
            try:
                y_train.append([float(row[0])])
            except:
                pass

    with open('x_test.csv', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        x_test = []
        for row in spamreader:
            try:
                x_test.append([float(item) for item in row])
            except:
                pass

    dump((x_train, y_train), open("trainset.pckl", "wb"))
    dump(x_test, open("testset.pckl", "wb"))

def get_train():
    return load(open("trainset.pckl", "rb"))

def get_test():
    return load(open("testset.pckl", "rb"))

def push_test(y_test, filename):
    with open(filename, 'w', newline='\n') as file:
        for item in y_test:
            file.write(str(int(item)) + "\n")

if __name__ == "__main__":
    refresh()
    a, b = get_train()
    import numpy as np
    a = np.array(a)
    np.set_printoptions(suppress=True, precision=3)
    print(np.amax(a, axis=0))
