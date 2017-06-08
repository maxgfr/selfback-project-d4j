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
            dataSetManager.createDataSetIterator(allData);
        } catch (IOException io){
            io.getMessage();
        } catch (InterruptedException inter){
            inter.getMessage();
        }

        MLPClassifierLinear network = new MLPClassifierLinear(123,0.01,2,500,100,3,6,300);

        network.train(dataSetManager.getDataSetIterator());


        File file = new File ("raw/sitting.csv");

        try {
            DataSetIterator it = dataSetManager.launchDataTest(file);
            network.makeEvaluation(it);
        } catch (IOException io){
            io.getMessage();
        } catch (InterruptedException inter){
            inter.getMessage();
        }

        network.saveModel();

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
