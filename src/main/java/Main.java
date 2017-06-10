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

        /**For allDataLight I use 20 epochs and 600 iterations*/

        KerasManager kerasManager = KerasManager.getInstance();
        DataSetManager dataSetManager = DataSetManager.getInstance(500,6,3);
        MLPClassifierLinear network = new MLPClassifierLinear(123,0.01,600,500,20,3,6,30);

        /**To train the model */
        /*final File allDataLight = new ClassPathResource("allData.csv").getFile();
        DataSetIterator trainData = dataSetManager.createTrainDataSetIterator(allDataLight);
        DataSetIterator testData = dataSetManager.createTrainDataSetIterator(allDataLight);
        network.train(trainData,testData);
        kerasManager.saveModelD4J(network.getModel());*/

        /**To test the model */
        File model = new File ("NetworkD4J.zip");
        MultiLayerNetwork networkRestored = kerasManager.restoreModelFromD4J(model);
        network.setModel(networkRestored);
        try {
            File file = new File ("raw/walking.csv");
            DataSetIterator testData = dataSetManager.createDataSetIterator(file);
            network.makePrediction(testData);
        } catch (IOException io){
            io.getMessage();
        } catch (InterruptedException inter){
            inter.getMessage();
        }

    }
}
