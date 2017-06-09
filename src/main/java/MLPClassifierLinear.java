import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Created by maxime on 01-Jun-17.
 */
public class MLPClassifierLinear {

    private int seed;//123
    private double learningRate;//0.01
    private int iteration;//50
    /** Defines number of samples that going to be propagated through the network.*/
    private int batchSize;//50
    private int nbEpochs;//20
    private int numInputs;//500
    private int numOutputs;//6
    private int numHiddenNodes;//20
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

    public void train (DataSetIterator iteratorTrain) {

        System.out.println("We're starting to train the network");

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
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false)
                .backprop(true)
                .build();


        model = new MultiLayerNetwork(conf);

        model.init();

        model.setListeners(new ScoreIterationListener(10));  //Print score every 10 parameter updates

        dispModel();

        for (int i=0; i<nbEpochs; i++) {
            model.fit(iteratorTrain);
            System.out.println(i+" epoch completed");
        }

        System.out.println("We finished to train the network");
    }

    public void makeEvaluation (DataSetIterator testData) {
        //evaluate the model on the test set
        System.out.println("Evaluation is starting");
        Evaluation eval = new Evaluation(6);
        while (testData.hasNext()) {
            DataSet ds = testData.next();
            INDArray output = model.output(ds.getFeatureMatrix(),false);
            eval.eval(ds.getLabels(), output);
        }
        System.out.println("The evaluation of the model is : "+ eval.stats());
    }

    public void makePredictionForADataSet(DataSet ds) {
        //evaluate the model on the test set
        System.out.println("Prediction is starting");
        int[] tab = model.predict(ds.getFeatureMatrix());
        for (int t : tab) {
            System.out.println(t);
        }
    }

    public void makePredictionForADataSetIterator(DataSetIterator it) {
        //evaluate the model on the test set
        System.out.println("Prediction is starting");
        int res = 0;
        int i = 0;
        while (it.hasNext()) {
            DataSet ds = it.next();
            int[] tab = model.predict(ds.getFeatureMatrix());
            for (int t : tab) {
                res+=t;
                res = res/tab.length;
            }
            System.out.println(i+" iteration(s), res="+res);
            i++;
        }
        int avg = res/i;

        System.out.println("The average prediction is: "+avg);
    }

    public void saveModel () throws IOException {
        File locationToSave = new File("NetworkFromD4J.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(model, locationToSave, saveUpdater);
        System.out.println("Model save in a zip");
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


}
