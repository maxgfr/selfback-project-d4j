import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by maxime on 01-Jun-17.
 */
public class KerasManager {

    private static KerasManager INSTANCE = null;

    private KerasManager() {
    }

    public static synchronized KerasManager getInstance() {
        if (INSTANCE == null)
        { 	INSTANCE = new KerasManager();
        }
        return INSTANCE;
    }

    public MultiLayerNetwork restoreModelFromKeras(File nameFile) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {
        MultiLayerNetwork restored = KerasModelImport.importKerasSequentialModelAndWeights(nameFile.getPath());
        return restored;
    }

    public void saveKerasModel(MultiLayerNetwork model) throws IOException {
        File locationToSave = new File("NetworkKeras.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(model, locationToSave, saveUpdater);
    }

    public MultiLayerNetwork restoreModelFromD4J (File file) throws IOException {
        MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(file);
        System.out.println("Model restore from D4J zip");
        return restored;
    }

    public void saveModelD4J (MultiLayerNetwork model) throws IOException {
        File locationToSave = new File("NetworkD4J.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(model, locationToSave, saveUpdater);
        System.out.println("Model save in a zip");
    }

    public void saveModelD4J (ComputationGraph model) throws IOException {
        File locationToSave = new File("NetworkD4J.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(model, locationToSave, saveUpdater);
        System.out.println("Model save in a zip");
    }
}
