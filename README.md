# selfback-project-d4j

Activity recognition through Neural Network

==The data provides from 50 person from wrist.==

## Resssource directory :

Data folder contains 6 files in whuch each file contain 1 collumn per axis (x,y,z) and all of the data from the activity
Label folder contains 6 files in which we can found the label for each file in the data folder

The **batchsize** is 500

## Evaluation :

- I make one evaluation with the same data to train and test my model and an other evaluation with 85% of my all data to train and to test my model

- Each model in directory ``model`` was trained with 100% of the data available.

## FOR CNN : 

NetworkD4J_CNN500 is the model which corresponds to CNN in the model folder. It uses data and label folder 

### Configuration : 

* 500 epochs,
* 0.01 of learning rate,
* No normalization of the data input
* DataInput.java : **height of 1, width of 500 and depth of 3**

```java
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iteration)
                .activation(Activation.RELU)
                .learningRate(learningRate)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.RMSPROP).momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(1,10) //depends height
                        .nIn(3)//depth
                        .nOut(150)
                        .stride(1,1)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) //max pooling
                        .kernelSize(1,2)
                        .stride(1,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(1,10)
                        .nIn(150)
                        .nOut(100)
                        .stride(1,1)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(1,2)
                        .stride(1,2)
                        .build())
                .layer(4, new ConvolutionLayer.Builder(1,10)
                        .nIn(100)
                        .nOut(80)
                        .stride(1,1)
                        .build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(1,2)
                        .stride(1,2)
                        .build())
                .layer(6, new ConvolutionLayer.Builder(1,10)
                        .nIn(80)
                        .nOut(60)
                        .stride(1,1)
                        .build())
                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(1,2)
                        .stride(1,2)
                        .build())
                .layer(8, new ConvolutionLayer.Builder(1,10)
                        .stride(1,1)
                        .nIn(60)
                        .nOut(40)
                        .build())
                .layer(9, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(1,2)
                        .stride(1,2)
                        .build())
                .layer(10, new DenseLayer.Builder() //fullyConnected
                        .nOut(900)
                        .activation(Activation.TANH)
                        .build())
                .layer(11, new DenseLayer.Builder()
                        .nOut(300)
                        .activation(Activation.TANH)
                        .dropOut(0.5)
                        .build())
                .layer(12, new OutputLayer.Builder()
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numOutputs)
                        .build())
                .setInputType(InputType.convolutional(1, 500, 3))
                .backprop(true)
                .pretrain(false)
                .build();
```

### Scores at the latest epoch with same data with 500 epochs:
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

### Scores at the latest epoch with 85% to train and 15% to test the model with 500 epochs :

```
Accuracy:        0.874
Precision:       0.8237
Recall:          0.8136
F1 Score:        0.8187
```
About labelisation of the examples :

```
Examples labeled as 0 classified by model as 0: 70 times
Examples labeled as 0 classified by model as 1: 1 times
Examples labeled as 0 classified by model as 3: 1 times
Examples labeled as 0 classified by model as 4: 15 times
Examples labeled as 0 classified by model as 5: 5 times
Examples labeled as 1 classified by model as 0: 3 times
Examples labeled as 1 classified by model as 1: 258 times
Examples labeled as 1 classified by model as 5: 1 times
Examples labeled as 2 classified by model as 0: 1 times
Examples labeled as 2 classified by model as 1: 2 times
Examples labeled as 2 classified by model as 2: 172 times
Examples labeled as 2 classified by model as 3: 1 times
Examples labeled as 2 classified by model as 5: 1 times
Examples labeled as 3 classified by model as 1: 1 times
Examples labeled as 3 classified by model as 2: 7 times
Examples labeled as 3 classified by model as 3: 180 times
Examples labeled as 3 classified by model as 5: 6 times
Examples labeled as 4 classified by model as 0: 4 times
Examples labeled as 4 classified by model as 3: 1 times
Examples labeled as 4 classified by model as 4: 33 times
Examples labeled as 4 classified by model as 5: 44 times
Examples labeled as 5 classified by model as 0: 8 times
Examples labeled as 5 classified by model as 2: 1 times
Examples labeled as 5 classified by model as 3: 1 times
Examples labeled as 5 classified by model as 4: 22 times
Examples labeled as 5 classified by model as 5: 161 times
```

![alt text](https://github.com/maxgfr/D4JSelfback/blob/master/screen/CNN_500/Capture.PNG)
