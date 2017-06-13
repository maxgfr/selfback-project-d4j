import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;

/**
 * Created by maxime on 01-Jun-17.
 */
public class Main {

    public static void main(String[] args) throws Exception {

        /**
         * LSTM
         * For allDataUltraLight (5000*5) I use 10 epochs and 5 iterations and 100 hiddenNodes ==> 0.85
         * For allDataVeryLight (25000*5) I use 60 epochs and 2 iterations and 30 hiddenNodes and regularization L2 0.0001(10^-4)==> 0.85
         * For allDataLight (50000*5) I use 25 epochs and 1 iterations and 30 hiddenNodes and regularization L2 0.0001(10^-4)==> 0.85
         */

        /**
         *CNN
         */

        KerasManager kerasManager = KerasManager.getInstance();
        DataSetManager dataSetManager = DataSetManager.getInstance(500,6,3);
        Classifier LSTM = new Classifier(123,0.01,1,50,40,3,6,10);
        Classifier CNN = new Classifier(0.01,1,50,40,3,6,10);

        /**To train  LSTM the model */
        /*final File allDataLight = new ClassPathResource("allDataLight.csv").getFile();
        LSTM.createLSTM();
        DataSetIterator trainData = dataSetManager.createDataSetIteratorForLSTM(allDataLight,true);
        DataSetIterator testData = dataSetManager.createDataSetIteratorForLSTM(allDataLight,true);
        LSTM.trainLSTM(trainData,testData);
        kerasManager.saveModelD4J(LSTM.getModel());*/

        /**To train  CNN the model */
        final File allDataLight = new ClassPathResource("allDataLight.csv").getFile();
        CNN.createCNN();
        dataSetManager.createDataSetIteratorForCNN(allDataLight);
        CNN.trainCNN(dataSetManager.listDataSetTrain,dataSetManager.listDataSetTest);
        kerasManager.saveModelD4J(CNN.getModel());

        /**To test the model */
        File model = new File ("NetworkD4J.zip");
        MultiLayerNetwork networkRestored = kerasManager.restoreModelFromD4J(model);
        LSTM.setModel(networkRestored);
        try {
            File file = new File ("raw/jogging.csv");
            DataSetIterator testModelData = dataSetManager.createDataSetIteratorForLSTM(file,false);
            LSTM.makePrediction(testModelData);
        } catch (IOException io){
            io.getMessage();
        } catch (InterruptedException inter){
            inter.getMessage();
        }

    }
}
