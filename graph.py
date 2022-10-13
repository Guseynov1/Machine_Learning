import matplotlib.pyplot as plt

def graph1(param, accuracy, id):
    plt.figure(figsize=[10,10])
    plt.plot(param, accuracy)
    plt.xlabel(id)
    plt.ylabel('accuracy')
    plt.grid('on')
    plt.show()

def graph2(param, accuracy_test, accuracy_train, id_x, id_y):
    plt.figure(figsize=[10,10])
    plt.plot(param, accuracy_test)
    plt.plot(param, accuracy_train)
    plt.legend(["accuracy_train", "accuracy_test"])
    plt.xlabel(id_x)
    plt.ylabel(id_y)
    plt.grid('on')
    plt.show()



