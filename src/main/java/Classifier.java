import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * Created by maxime on 01-Jun-17.
 */
public class Classifier {

    private int seed;//123
    private double learningRate;//0.01
    private int iteration;
    /** Defines number of samples that going to be propagated through the network.*/
    private int batchSize;//500
    private int nbEpochs;
    private int numInputs;//3
    private int numOutputs;//6
    private int numHiddenNodes;//30
    private MultiLayerNetwork model;

    public Classifier (int seed, double learningRate, int iteration, int batchSize, int nEpochs, int numInputs, int numOutputs, int numHiddenNodes) {
        this.seed = seed;
        this.learningRate = learningRate;
        this.iteration = iteration;
        this.batchSize= batchSize;
        this.nbEpochs = nEpochs;
        this.numInputs= numInputs;
        this.numOutputs= numOutputs;
        this.numHiddenNodes= numHiddenNodes;
    }

    public Classifier (double learningRate, int iteration, int batchSize, int nEpochs, int numInputs, int numOutputs) {
        this.learningRate = learningRate;
        this.iteration = iteration;
        this.batchSize = batchSize;
        this.nbEpochs = nEpochs;
        this.numInputs= numInputs;
        this.numOutputs= numOutputs;
    }

    public MultiLayerNetwork getModel () {return model;}

    public void setModel (MultiLayerNetwork mln) {model = mln;}

    public void createLSTM () {

        System.out.println("We're starting to create the LSTM network");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)    //Random number generator seed for improved repeatability. Optional.
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(iteration)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .learningRate(learningRate)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
                .gradientNormalizationThreshold(0.5)
                .regularization(true).l2(0.0001)
                .list()
                .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(numInputs).nOut(numHiddenNodes).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false)
                .backprop(true)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        model = new MultiLayerNetwork(conf);

        model.init();

        model.setListeners(new ScoreIterationListener(1));

        System.out.println("We finished to create the LSTM network");

    }

    public void createFeedForward () {

        System.out.println("We're starting to create the FeedForward network");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iteration)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        model = new MultiLayerNetwork(conf);

        model.init();

        model.setListeners(new ScoreIterationListener(100));

        System.out.println("We're starting to create the FeedForward network");
    }

    public void createCNN () {

        System.out.println("We're starting to create the CNN network");

        Map<Integer, Double> lrSchedule = new HashMap<Integer, Double>();
        lrSchedule.put(0, learningRate);
        lrSchedule.put(1000, 0.005);
        lrSchedule.put(3000, 0.001);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(iteration)
                .regularization(true).l2(0.0005)
                .learningRate(learningRate)
                .learningRateDecayPolicy(LearningRatePolicy.Schedule)
                .learningRateSchedule(lrSchedule)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder()
                        .nIn(numInputs)
                        .kernelSize(1,10)
                        .stride(1,1)
                        .nOut(150)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(1,2)
                        .stride(1,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder()
                        .nIn(numInputs)
                        .kernelSize(1,10)
                        .stride(1,1)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(1,2)
                        .stride(1,2)
                        .build())
                .layer(4, new ConvolutionLayer.Builder()
                        .kernelSize(1,10)
                        .stride(1,1)
                        .nIn(numInputs)
                        .nOut(80)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(1,2)
                        .stride(1,2)
                        .build())
                .layer(6, new ConvolutionLayer.Builder()
                        .kernelSize(1,10)
                        .stride(1,1)
                        .nIn(numInputs)
                        .nOut(60)
                        .activation(Activation.RELU)
                        .build())
                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(1,2)
                        .stride(1,2)
                        .build())
                .layer(8, new ConvolutionLayer.Builder()
                        .nIn(numInputs)
                        .kernelSize(1,10)
                        .stride(1,1)
                        .nOut(40)
                        .activation(Activation.RELU)
                        .build())
                .layer(9, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(1,2)
                        .stride(1,2)
                        .build())
                .layer(10, new OutputLayer.Builder()
                        .nIn(numInputs)
                        .nOut(numOutputs)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .backprop(true)
                .pretrain(false)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        model = new MultiLayerNetwork(conf);

        model.init();

        model.setListeners(new ScoreIterationListener(100));

        System.out.println("We're starting to create the CNN network");
    }

    public void trainLSTM (DataSetIterator iteratorTrain, DataSetIterator testData) {

        System.out.println("We're starting to train the LSTM network");

        dispModel();

        for (int i=1; i<nbEpochs+1; i++) {
            model.fit(iteratorTrain);
            makeEvaluation(model,testData);
            System.out.println(i+" epoch(s) completed");
        }

        System.out.println("We finished to train the LSTM network");
    }

    public void trainFeedForward (List<DataSet> train, List<DataSet>  test) {

        System.out.println("We're starting to train the FeedForward network");
        dispModel();
        for (DataSet ds : train) {
            model.fit(ds);
        }
        System.out.println("We finished to train the FeedForward network");
    }

    public void trainCNN (){

    }

    public void makePrediction(DataSetIterator it) {
        //evaluate the model on the test set
        System.out.println("Prediction is starting");
        List<Integer> list = new LinkedList<Integer>();
        while (it.hasNext()) {
            DataSet ds = it.next();
            int[] ls = model.predict(ds.getFeatureMatrix());
            for(int s : ls) {
                list.add(s);
            }
        }
        dispOccurence(list);
    }

    private void makeEvaluation (MultiLayerNetwork network, DataSetIterator testData) {
        Evaluation evaluation = network.evaluate(testData);
        System.out.println("Evaluation of model with Accuracy = "+evaluation.accuracy()+" and F1 = "+evaluation.f1());
        testData.reset();
    }

    //http://localhost:9000/train
    private void dispModel () {
        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        //Then add the StatsListener to collect this information from the network, as it trains
        model.setListeners(new StatsListener(statsStorage));
    }

    private void dispOccurence (List<Integer> myList) {
        int downstairs,jogging,sitting,standing,upstairs,walking;
        downstairs=jogging=sitting=standing=upstairs=walking = 0;
        for (Integer x : myList) {
            switch (x) {
                case 0:  downstairs++;
                    break;
                case 1:  jogging++;
                    break;
                case 2:  sitting++;
                    break;
                case 3:  standing++;
                    break;
                case 4:  upstairs++;
                    break;
                case 5: walking++;
                    break;
            }
            System.out.println("Number of occurrence :" +
                    "\nfor downstairs is "+downstairs+
                    "\nfor jogging is "+jogging+
                    "\nfor sitting is "+sitting+
                    "\nfor standing is "+ standing+
                    "\nfor upstairs is " +upstairs+
                    "\nfor walking is " +walking);
        }
    }

}

