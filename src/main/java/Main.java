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

        /**
         * LSTM
         * For allDataUltraLight (5000*5) I use 10 epochs and 5 iterations and 100 hiddenNodes ==> 0.85
         * For allDataVeryLight (25000*5) I use 60 epochs and 2 iterations and 10 hiddenNodes and regularization L2 0.0001(10^-4)==> 0.85
         * For allDataLight (50000*5) I use 10 epochs and 1 iterations and 10 hiddenNodes and regularization L2 0.0001(10^-4)==> 0.85
         */

        /**
         * FeedForward
         * numberOfHiddenNodes = 2 * input + output
         * For allData I use 100 epochs,100 nodes, 1 iteration
         */

        KerasManager kerasManager = KerasManager.getInstance();
        DataSetManager dataSetManager = DataSetManager.getInstance(500,6,3);
        Classifier LSTM = new Classifier(123,0.01,1,500,40,3,6,10);
        Classifier FeedForward = new Classifier(123,0.01,1,500,2,3,6,12);
        Classifier CNN = new Classifier(0.01,1,500,15, 6);

        /**To train LSTM model */
        /*final File data = new ClassPathResource("data_light").getFile();
        final File index = new ClassPathResource("index").getFile();
        LSTM.createLSTM();
        DataSetIterator testData = dataSetManager.createDataSetIteratorForLSTM(data,index);
        DataSetIterator trainData = dataSetManager.createDataSetIteratorForLSTM(data,index);
        LSTM.trainRNN(trainData,testData);
        kerasManager.saveModelD4J(LSTM.getModel());*/

        /**To train  FeedForward model */
        /*final File allData = new ClassPathResource("allData.csv").getFile();
        DataSetIterator testData = dataSetManager.createDataSetIteratorForFeedForward(allData,true);
        DataSetIterator trainData = dataSetManager.createDataSetIteratorForFeedForward(allData,true);
        FeedForward.createFeedForward();
        FeedForward.trainFeedForward(trainData,testData);
        kerasManager.saveModelD4J(FeedForward.getModel());*/

        /**To train  CNN model */
        final File data = new ClassPathResource("data").getFile();
        final File datatest = new ClassPathResource("data_thigh34").getFile();
        CNN.createCNN();
        DataSetIterator trainData = dataSetManager.createDataSetIteratorForCNN(data);
        DataSetIterator testData = dataSetManager.createDataSetIteratorForCNN(datatest);
        CNN.trainCNN(trainData,testData);
        kerasManager.saveModelD4J(CNN.getModel());

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
        /*File model = new File ("cnn_wrist_33.h5");
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
