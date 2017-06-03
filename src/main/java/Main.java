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

        final File allData = new ClassPathResource("allData.csv").getFile();

        DataSetManager dataSetManager = DataSetManager.getInstance(50,6,3);

        try {
            dataSetManager.createDataSetFromOneFile(allData);
        } catch (IOException io){
            io.getMessage();
        } catch (InterruptedException inter){
            inter.getMessage();
        }

        MLPClassifierLinear network = new MLPClassifierLinear(123,0.01,1,50,20,3,6,3);

        //network.trainMLNetwork(listSetIterator);

        network.train(dataSetManager.getDataTrain());

        //network.trainMLNetworkOne(dataSet);

        //network.saveModel();

        //network.dispModel();
    }
}
