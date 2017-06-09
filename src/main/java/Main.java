import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;

/**
 * Created by maxime on 01-Jun-17.
 */
public class Main {

    public static void main(String[] args) throws Exception {

        final File allData = new ClassPathResource("allData.csv").getFile();

        DataSetManager dataSetManager = DataSetManager.getInstance(500,6,3);

        try {
            DataSetIterator trainData = dataSetManager.createDataSetIterator(allData);

            MLPClassifierLinear network = new MLPClassifierLinear(123,0.01,1,500,1,3,6,300);

            network.train(trainData);

            File file = new File ("raw/sitting.csv");

            DataSetIterator testData = dataSetManager.createDataSetIterator(file);

            network.makeEvaluation(testData);

            network.saveModel();

        } catch (IOException io){
            io.getMessage();
        } catch (InterruptedException inter){
            inter.getMessage();
        }


        /** Restore and save model from Keras*/
        /*KerasManager kerasManager = KerasManager.getInstance();

        try {
            MultiLayerNetwork mln = kerasManager.loadFile("cnn_wrist_33.h5");
            kerasManager.saveModel(mln);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (UnsupportedKerasConfigurationException e) {
            e.printStackTrace();
        } catch (InvalidKerasConfigurationException e) {
            e.printStackTrace();
        }*/

    }
}
