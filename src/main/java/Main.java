import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Created by maxime on 01-Jun-17.
 */
public class Main {

    public static void main(String[] args) throws Exception {


        final File data = new ClassPathResource("data").getFile();
        final File data_test = new ClassPathResource("data_test/standing.csv").getFile();

        KerasManager kerasManager = KerasManager.getInstance();
        DataSetManager dataSetManager = new DataSetManager(500,6);
        DataSetManager myOwnDataSetManager = new DataSetManager(1,500,3);
        DataSetIterator trainData = myOwnDataSetManager.createMyOwnDataSetIterator(data,true);
        DataSetIterator testData = myOwnDataSetManager.createMyOwnDataSetIterator(data,false);
        //DataSetIterator trainData = myOwnDataSetManager.createMyOwnDataSetIteratorSameData(data);
        //DataSetIterator testData = myOwnDataSetManager.createMyOwnDataSetIteratorSameData(data);

        Classifier LSTM = new Classifier(123,0.01,1,200,3,6,300,30);
        Classifier CNN = new Classifier(123, 0.01,1,500, 3,6,20);
        Classifier myOwnNetwork = new Classifier(123,0.01,1,500,6);

        /**To train LSTM model */
        /*LSTM.createLSTM();
        LSTM.trainLSTM(trainData,testData);
        kerasManager.saveModelD4J(LSTM.getComputationGraph());*/


        /**To train  CNN model */
        /*CNN.createCNN();
        CNN.trainCNN(trainData,testData);
        kerasManager.saveModelD4J(CNN.getModel());*/

        /**To train my own model */
        /*myOwnNetwork.createMyOwnNetwork();
        myOwnNetwork.trainMyOwnNetwork(trainData,testData);
        kerasManager.saveModelD4J(myOwnNetwork.getModel());*/

        /**To test D4J model from zip */
        DataSetIterator testModelData = myOwnDataSetManager.createDataSetIteratorTest(data_test);
        DataSetIterator sameData = myOwnDataSetManager.createDataSetIteratorTest(data_test);

        File modelCNN = new File ("model/NetworkD4J_CNN500.zip");
        MultiLayerNetwork networkRestored1 = kerasManager.restoreModelFromD4J(modelCNN);
        CNN.setModel(networkRestored1);
        CNN.makePrediction(testModelData);
        //CNN.dispTabProbabilities(testModelData);

        File modelMINE = new File ("model/NetworkD4J_CNN+RNN500.zip");
        MultiLayerNetwork networkRestored2 = kerasManager.restoreModelFromD4J(modelMINE);
        myOwnNetwork.setModel(networkRestored2);
        myOwnNetwork.makePrediction(sameData);
        //myOwnNetwork.dispTabProbabilities(sameData);

        /**To test from Keras model the model */
        /*File model = new File ("model/cnn_wrist_33.h5");
        MultiLayerNetwork networkRestored = kerasManager.restoreModelFromKeras(model);
        myOwnNetwork.setModel(networkRestored);
        DataSetIterator dataToTest = myOwnDataSetManager.createDataSetIteratorTest(data_test);
        myOwnNetwork.makePrediction(dataToTest);*/
    }
}
