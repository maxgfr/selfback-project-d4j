# D4JSelfback
Activity recognition through Neural Network

==The data provides from 50 person from wrist.==

## In resssource directory :

Data folder contains 6 files in whuch each file contain 1 collumn per axis (x,y,z) and all of the data from the activity
Label folder contains 6 files in which we can found the label for each file in the data folder

## FOR CNN : 

NetworkD4J_1 is the model which corresponds to CNN with configuration : Classifier CNN = new Classifier(0.01,1,500,15, 6);

The configuration of this CNN is  : 
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
						.seed(seed)
						.iterations(iteration)
						.regularization(false).l2(0.005)
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
                .layer(10, new OutputLayer.Builder()
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numOutputs)
                        .build())
                .setInputType(InputType.convolutional(1, 500, 3))
                .backprop(true)
                .pretrain(false)
                .build();

He uses the folder : data and label

![alt text](https://raw.githubusercontent.com/maxgfr/D4JSelfback/blob/master/screen/Capture.PNG)
![alt text](https://github.com/maxgfr/D4JSelfback/blob/master/screen/Capture.PNG)

After : 350 epoch(s) completed

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


Scores =>
 Accuracy:        0.8982
 Precision:       0.8775
 Recall:          0.8477
 F1 Score:        0.8623