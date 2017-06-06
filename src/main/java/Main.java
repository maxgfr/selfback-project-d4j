import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

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
            dataSetManager.createDataSetFromOneFile(allData);
        } catch (IOException io){
            io.getMessage();
        } catch (InterruptedException inter){
            inter.getMessage();
        }

        MLPClassifierLinear network = new MLPClassifierLinear(123,0.01,100,500,20,3,6,300);

        network.train(dataSetManager.getTrainingData());

        network.makeEvaluation(dataSetManager.getTestData());

        network.saveModel();

        /** Restore and save model from Keras*/

        KerasManager kerasManager = KerasManager.getInstance();

        try {
            MultiLayerNetwork mln = kerasManager.loadFile("cnn_wrist_33.h5");
            kerasManager.saveModel(mln);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (UnsupportedKerasConfigurationException e) {
            e.printStackTrace();
        } catch (InvalidKerasConfigurationException e) {
            e.printStackTrace();
        }

    }
}
