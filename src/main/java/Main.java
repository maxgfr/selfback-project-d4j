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
        DataSetManager dataSetManager = DataSetManager.getInstance(500,6,3);
        Classifier LSTM = new Classifier(123,0.01,1,500,40,3,6,20);
        Classifier CNN = new Classifier(0.01,1,500,500, 6);

        /**To train LSTM model */
        final File data = new ClassPathResource("data").getFile();
        LSTM.createLSTM();
        DataSetIterator testData = dataSetManager.createMyOwnDataSetIterator(data);
        DataSetIterator trainData = dataSetManager.createMyOwnDataSetIterator(data);
        LSTM.trainLSTM(trainData,testData);
        kerasManager.saveModelD4J(LSTM.getComputationGraph());


        /**To train  CNN model */
        /*final File data = new ClassPathResource("data").getFile();
        final File datatest = new ClassPathResource("data").getFile();
        CNN.createCNN();
        DataSetIterator trainData = dataSetManager.createMyOwnDataSetIterator(data);
        DataSetIterator testData = dataSetManager.createMyOwnDataSetIterator(datatest);
        CNN.trainCNN(trainData,testData);
        kerasManager.saveModelD4J(CNN.getModel());*/

        /**To test D4J model from zip */
        /*File model = new File ("NetworkD4J.zip");
        MultiLayerNetwork networkRestored = kerasManager.restoreModelFromD4J(model);
        FeedForward.setModel(networkRestored);
        try {
            File file = new File ("data_test/jogging.csv");
            DataSetIterator testModelData = dataSetManager.createDataSetIteratorForFeedForward(file,false);
            FeedForward.makePrediction(testModelData);
        } catch (IOException io){
            io.getMessage();
        } catch (InterruptedException inter){
            inter.getMessage();
        }*/

        /**To test CNN D4J model from zip */
        /*File model = new File ("NetworkD4J.zip");
        MultiLayerNetwork networkRestored = kerasManager.restoreModelFromD4J(model);
        CNN.setModel(networkRestored);
        try {
            File file = new File ("data_test/sitting.csv");
            DataSetIterator testModelData = dataSetManager.createDataSetTestCNN(file);
            CNN.makePrediction(testModelData);
        } catch (IOException io){
            io.getMessage();
        } catch (InterruptedException inter){
            inter.getMessage();
        }*/


        /**To test from Keras model the model */
        /*File model = new File ("model/cnn_wrist_33.h5");
        MultiLayerNetwork networkRestored = kerasManager.restoreModelFromKeras(model);
        FeedForward.setModel(networkRestored);
        try {
            File file = new File ("data_test/jogging.csv");
            DataSetIterator testModelData = dataSetManager.createDataSetIteratorForFeedForward(file,false);
            FeedForward.makePrediction(testModelData);
        } catch (IOException io){
            io.getMessage();
        } catch (InterruptedException inter){
            inter.getMessage();
        }*/

    }
}
