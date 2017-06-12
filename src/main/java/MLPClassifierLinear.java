import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
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
public class MLPClassifierLinear {

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

    public MLPClassifierLinear (int seed, double learningRate, int iteration, int batchSize, int nEpochs, int numInputs, int numOutputs, int numHiddenNodes) {
        this.seed = seed;
        this.learningRate = learningRate;
        this.iteration = iteration;
        this.batchSize= batchSize;
        this.nbEpochs = nEpochs;
        this.numInputs= numInputs;
        this.numOutputs= numOutputs;
        this.numHiddenNodes= numHiddenNodes;
    }

    public MultiLayerNetwork getModel () {return model;}

    public void setModel (MultiLayerNetwork mln) {model = mln;}

    public void train (DataSetIterator iteratorTrain, DataSetIterator testData) {

        System.out.println("We're starting to train the network");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)    //Random number generator seed for improved repeatability. Optional.
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(iteration)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .learningRate(learningRate)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
                .gradientNormalizationThreshold(0.5)
                .list()
                .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(numInputs).nOut(numHiddenNodes).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false)
                .backprop(true)
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(100)
                .tBPTTBackwardLength(100)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        model = new MultiLayerNetwork(conf);

        model.init();

        model.setListeners(new ScoreIterationListener(1));

        dispModel();

        for (int i=1; i<nbEpochs+1; i++) {
            model.fit(iteratorTrain);
            makeEvaluation(model,testData);
            System.out.println(i+" epoch(s) completed");
        }

        System.out.println("We finished to train the network");
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

