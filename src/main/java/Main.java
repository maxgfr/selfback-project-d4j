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

        KerasManager kerasManager = KerasManager.getInstance();
        //DataSetManager dataSetManager = DataSetManager.getInstance(500,6);
        DataSetManager myOwnDataSetManager = DataSetManager.getInstance(1,500,3);
        Classifier LSTM = new Classifier(123,0.01,1,200,3,6,300,30);
        Classifier CNN = new Classifier(123, 0.01,1,1500, 3,6,20);
        Classifier myOwnNetwork = new Classifier(123,0.01,1,1500,6);

        /**To train LSTM model */
        /*final File data = new ClassPathResource("data").getFile();
        LSTM.createLSTM();
        DataSetIterator testData = myOwnDataSetManager.createMyOwnDataSetIterator(data);
        DataSetIterator trainData = myOwnDataSetManager.createMyOwnDataSetIterator(data);
        LSTM.trainLSTM(trainData,testData);
        kerasManager.saveModelD4J(LSTM.getComputationGraph());*/


        /**To train  CNN model */
        /*final File data = new ClassPathResource("data").getFile();
        final File data_test = new ClassPathResource("data").getFile();
        CNN.createCNN();
        DataSetIterator trainData = myOwnDataSetManager.createMyOwnDataSetIterator(data);
        DataSetIterator testData = myOwnDataSetManager.createMyOwnDataSetIterator(data_test);
        CNN.trainCNN(trainData,testData);
        kerasManager.saveModelD4J(CNN.getModel());*/

        /**To train my own model */
        /*final File data = new ClassPathResource("data").getFile();
        myOwnNetwork.createMyOwnNetwork();
        DataSetIterator testData = myOwnDataSetManager.createMyOwnDataSetIterator(data);
        DataSetIterator trainData = myOwnDataSetManager.createMyOwnDataSetIterator(data);
        myOwnNetwork.trainMyOwnNetwork(trainData,testData);
        kerasManager.saveModelD4J(myOwnNetwork.getModel());*/

        /**To test D4J model from zip */
        File model = new File ("model/NetworkD4J_1.zip");
        final File data_test = new ClassPathResource("data_test/downstairs.csv").getFile();
        DataSetIterator testModelData = myOwnDataSetManager.createDataSetIteratorTest(data_test);
        DataSetIterator sameData = myOwnDataSetManager.createDataSetIteratorTest(data_test);
        MultiLayerNetwork networkRestored = kerasManager.restoreModelFromD4J(model);
        myOwnNetwork.setModel(networkRestored);
        myOwnNetwork.makePrediction(testModelData);
        myOwnNetwork.dispTabProbabilities(sameData);

        /**To test from Keras model the model */
        /*File model = new File ("model/cnn_wrist_33.h5");
        MultiLayerNetwork networkRestored = kerasManager.restoreModelFromKeras(model);
        FeedForward.setModel(networkRestored);
        try {
            File file = new File ("data_test/jogging.csv");
            DataSetIterator testModelData = myOwnDataSetManager.createDataSetIteratorTest(file,false);
            FeedForward.makePrediction(testModelData);
        } catch (IOException io){
            io.getMessage();
        } catch (InterruptedException inter){
            inter.getMessage();
        }*/

    }
}
