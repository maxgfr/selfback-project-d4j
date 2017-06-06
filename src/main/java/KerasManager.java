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

    public void displayFilesForFolder(final File folder) {
        for (final File fileEntry : folder.listFiles()) {
            if (fileEntry.isDirectory()) {
                displayFilesForFolder(fileEntry);
            } else {
                System.out.println(fileEntry.getName());
            }
        }
    }

    public List<File> listFilesForFolder (final File folder) {
        List<File> list = new LinkedList<File>();
        for (final File fileEntry : folder.listFiles()) {
            if (fileEntry.isDirectory()) {
                listFilesForFolder(fileEntry);
            } else {
                list.add(fileEntry);
            }
        }
        return list;
    }

    public MultiLayerNetwork loadFile(String nameFile) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {
        MultiLayerNetwork restored = KerasModelImport.importKerasSequentialModelAndWeights(nameFile);
        return restored;
    }

    public void saveModel(MultiLayerNetwork model) throws IOException {
        File locationToSave = new File("NetworkKeras.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(model, locationToSave, saveUpdater);
    }
}
