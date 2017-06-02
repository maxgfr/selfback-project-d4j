import org.datavec.api.util.ClassPathResource;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by maxime on 01-Jun-17.
 */
public class Main {

    public static void main(String[] args) throws Exception {

        ReadFile readFile = ReadFile.getInstance();

        final File folderDownstairs = new ClassPathResource("downstairs").getFile();
        final File folderJogging = new ClassPathResource("jogging").getFile();
        final File folderSitting = new ClassPathResource("sitting").getFile();
        final File folderStanding = new ClassPathResource("standing").getFile();
        final File folderUpstairs = new ClassPathResource("upstairs").getFile();
        final File folderWalking = new ClassPathResource("walking").getFile();

        List<String> listDownstairs = readFile.listFilesPathForFolder(folderDownstairs);
        List<String> listJogging= readFile.listFilesPathForFolder(folderJogging);
        List<String> listSitting = readFile.listFilesPathForFolder(folderSitting);
        List<String> listStanding = readFile.listFilesPathForFolder(folderStanding);
        List<String> listUpstairs = readFile.listFilesPathForFolder(folderUpstairs);
        List<String> listWalking = readFile.listFilesPathForFolder(folderWalking);


        DataSetManager dataSetManager = DataSetManager.getInstance(50);

        List<DataSetIterator> listSetIteratorDownstairs = dataSetManager.createDataSetIterator(listDownstairs);
        List<DataSetIterator> listSetIteratorJogging = dataSetManager.createDataSetIterator(listJogging);
        List<DataSetIterator> listSetIteratorSitting = dataSetManager.createDataSetIterator(listSitting);
        List<DataSetIterator> listSetIteratorStanding = dataSetManager.createDataSetIterator(listStanding);
        List<DataSetIterator> listSetIteratorUpstairs = dataSetManager.createDataSetIterator(listUpstairs);
        List<DataSetIterator> listSetIteratorWalking = dataSetManager.createDataSetIterator(listWalking);

        MLPClassifierLinear network = new MLPClassifierLinear(123,0.01,50,20,500,6,20);

        network.trainMLNetwork(listSetIteratorDownstairs);
        network.trainMLNetwork(listSetIteratorJogging);
        network.trainMLNetwork(listSetIteratorSitting);
        network.trainMLNetwork(listSetIteratorStanding);
        network.trainMLNetwork(listSetIteratorUpstairs);
        network.trainMLNetwork(listSetIteratorWalking);

        network.saveModel();
    }
}
