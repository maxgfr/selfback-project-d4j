import org.datavec.api.util.ClassPathResource;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

/**
 * Created by maxime on 01-Jun-17.
 */
public class Main {

    public static void main(String[] args) throws Exception {

        KerasManager kerasManager = KerasManager.getInstance();
        DataSetManager dataSetManager = DataSetManager.getInstance(500,6,3);
        MLPClassifierLinear network = new MLPClassifierLinear(123,0.01,1,500,20,3,6,300);

        /**To train the model */
        final File allDataLight = new ClassPathResource("allDataLight.csv").getFile();
        DataSetIterator trainData = dataSetManager.createTrainDataSetIterator(allDataLight);
        DataSetIterator testData = dataSetManager.createDataSetIteror(allDataLight);
        network.train(trainData,testData);
        kerasManager.saveModelD4J(network.getModel());

        /**To test the model */
        /*File model = new File ("NetworkFromD4J.zip");
        MultiLayerNetwork networkRestored = kerasManager.restoreModelFromD4J(model);
        network.setModel(networkRestored);
        try {
            File file = new File ("raw/sitting.csv");
            DataSetIterator testData = dataSetManager.createDataSetIteror(file);
            network.makePrediction(testData);
        } catch (IOException io){
            io.getMessage();
        } catch (InterruptedException inter){
            inter.getMessage();
        }*/

    }
}
