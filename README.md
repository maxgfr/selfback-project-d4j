# D4JSelfback
Activity recognition through Neural Network

==The data provides from 50 person from wrist.==

## In resssource directory :

Data folder contains 6 files in whuch each file contain 1 collumn per axis (x,y,z) and all of the data from the activity
Label folder contains 6 files in which we can found the label for each file in the data folder

The **batchsize** is 500

## FOR CNN : 

NetworkD4J_1 is the model which corresponds to CNN in the model folder

CNN uses data and label folder 

DataInput.java : **height of 1, width of 500 and depth of 3**

Configuration : **350 epochs, 0.01 of learning rate**

**Scores**
Accuracy:        0.8982

Precision:       0.8775

Recall:          0.8477

F1 Score:        0.8623

Examples labeled as 0 classified by model as 0: 590 times

Examples labeled as 0 classified by model as 1: 4 times

Examples labeled as 0 classified by model as 2: 1 times

Examples labeled as 0 classified by model as 3: 3 times

Examples labeled as 0 classified by model as 4: 42 times

Examples labeled as 0 classified by model as 5: 56 times

Examples labeled as 1 classified by model as 0: 4 times

Examples labeled as 1 classified by model as 1: 1923 times

Examples labeled as 1 classified by model as 2: 5 times

Examples labeled as 1 classified by model as 3: 6 times

Examples labeled as 1 classified by model as 4: 3 times

Examples labeled as 1 classified by model as 5: 12 times

Examples labeled as 2 classified by model as 0: 1 times

Examples labeled as 2 classified by model as 1: 1 times

Examples labeled as 2 classified by model as 2: 1497 times

Examples labeled as 2 classified by model as 3: 24 times

Examples labeled as 2 classified by model as 5: 5 times

Examples labeled as 3 classified by model as 0: 2 times

Examples labeled as 3 classified by model as 1: 1 times

Examples labeled as 3 classified by model as 2: 49 times

Examples labeled as 3 classified by model as 3: 1455 times

Examples labeled as 3 classified by model as 4: 3 times

Examples labeled as 3 classified by model as 5: 23 times

Examples labeled as 4 classified by model as 0: 52 times

Examples labeled as 4 classified by model as 2: 1 times

Examples labeled as 4 classified by model as 3: 11 times

Examples labeled as 4 classified by model as 4: 303 times

Examples labeled as 4 classified by model as 5: 389 times

Examples labeled as 5 classified by model as 0: 38 times

Examples labeled as 5 classified by model as 2: 7 times

Examples labeled as 5 classified by model as 3: 14 times

Examples labeled as 5 classified by model as 4: 57 times

Examples labeled as 5 classified by model as 5: 1418 times

![alt text](https://github.com/maxgfr/D4JSelfback/blob/master/screen/Capture.PNG)
