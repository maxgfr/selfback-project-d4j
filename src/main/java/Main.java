import org.datavec.api.util.ClassPathResource;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by maxime on 01-Jun-17.
 */
public class Main {

    public static void main(String[] args) throws Exception {

        ReadFile readFile = ReadFile.getInstance();

        final File folderDownstairs = new ClassPathResource("data").getFile();

        List<File> listOfCSV = readFile.listFilesForFolder(folderDownstairs);

        DataSetManager dataSetManager = DataSetManager.getInstance(50);

        try {
            //List<DataSetIterator> listSetIterator = dataSetManager.createDataSetIterator(listOfCSV);

            //DataSetIterator dataSetIterator = dataSetManager.createOneDataSet(listOfCSV.get(0));

            DataSet dataSet = dataSetManager.createOneDataSetAll(listOfCSV.get(0));

            MLPClassifierLinear network = new MLPClassifierLinear(123,0.01,50,20,500,6,20);

            //network.trainMLNetwork(listSetIterator);

            network.trainMLNetworkOne(dataSet);

            network.saveModel();

            network.dispModel();
        } catch (IOException io){
            io.getMessage();
        } catch (InterruptedException inter){
            inter.getMessage();
        }
    }
}
