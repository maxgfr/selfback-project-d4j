# D4JSelfback
Activity recognition through Neural Network

==The data provides from 50 person from wrist.==

## In resssource directory :

Data folder contains 6 files in whuch each file contain 1 collumn per axis (x,y,z) and all of the data from the activity
Label folder contains 6 files in which we can found the label for each file in the data folder

The **batchsize** is 500

## FOR CNN : 

NetworkD4J_CNN500 is the model which corresponds to CNN in the model folder. It uses data and label folder 

### Configuration : 

* 500 epochs,
* 0.01 of learning rate,
* no normalization of the data input
* DataInput.java : **height of 1, width of 500 and depth of 3**

### Scores at 500 epochs
```
Accuracy:        0.9456
Precision:       0.9326
Recall:          0.9216
F1 Score:        0.9271
```
About labelisation of the examples :

```
Examples labeled as 0 classified by model as 0: 618 times
Examples labeled as 0 classified by model as 1: 2 times
Examples labeled as 0 classified by model as 3: 1 times
Examples labeled as 0 classified by model as 4: 50 times
Examples labeled as 0 classified by model as 5: 21 times
Examples labeled as 1 classified by model as 0: 2 times
Examples labeled as 1 classified by model as 1: 1929 times
Examples labeled as 1 classified by model as 2: 2 times
Examples labeled as 1 classified by model as 3: 7 times
Examples labeled as 1 classified by model as 4: 5 times
Examples labeled as 1 classified by model as 5: 8 times
Examples labeled as 2 classified by model as 2: 1509 times
Examples labeled as 2 classified by model as 3: 16 times
Examples labeled as 2 classified by model as 4: 1 times
Examples labeled as 3 classified by model as 0: 2 times
Examples labeled as 3 classified by model as 2: 28 times
Examples labeled as 3 classified by model as 3: 1504 times
Examples labeled as 3 classified by model as 4: 1 times
Examples labeled as 3 classified by model as 5: 4 times
Examples labeled as 4 classified by model as 0: 17 times
Examples labeled as 4 classified by model as 2: 2 times
Examples labeled as 4 classified by model as 3: 4 times
Examples labeled as 4 classified by model as 4: 563 times
Examples labeled as 4 classified by model as 5: 175 times
Examples labeled as 5 classified by model as 0: 10 times
Examples labeled as 5 classified by model as 2: 1 times
Examples labeled as 5 classified by model as 3: 2 times
Examples labeled as 5 classified by model as 4: 74 times
Examples labeled as 5 classified by model as 5: 1442 times
```

![alt text](https://github.com/maxgfr/D4JSelfback/blob/master/screen/CNN_500/Capture.PNG)
