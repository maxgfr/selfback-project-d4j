import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by maxime on 01-Jun-17.
 */
public class Main {

    public static void main(String[] args) throws Exception {

        List<String> listDownstairs = new LinkedList<String>();
        List<String> listJogging = new LinkedList<String>();
        List<String> listSitting = new LinkedList<String>();
        List<String> listStanding = new LinkedList<String>();
        List<String> listUpstairs = new LinkedList<String>();
        List<String> listWalking = new LinkedList<String>();

        ReadFile readFile = ReadFile.getInstance();

        final File folderDownstairs = new ClassPathResource("downstairs").getFile();
        final File folderJogging = new ClassPathResource("jogging").getFile();
        final File folderSitting = new ClassPathResource("sitting").getFile();
        final File folderStanding = new ClassPathResource("standing").getFile();
        final File folderUpstairs = new ClassPathResource("upstairs").getFile();
        final File folderWalking = new ClassPathResource("walking").getFile();

        listDownstairs = readFile.listFilesPathForFolder(folderDownstairs);
        listJogging = readFile.listFilesPathForFolder(folderJogging);
        listSitting = readFile.listFilesPathForFolder(folderSitting);
        listStanding = readFile.listFilesPathForFolder(folderStanding);
        listUpstairs = readFile.listFilesPathForFolder(folderUpstairs);
        listWalking = readFile.listFilesPathForFolder(folderWalking);


        int batchSize = 50;

        final String filenameTrain  = new ClassPathResource("linear_data_train.csv").getFile().getPath();
        final String filenameTest  = new ClassPathResource("linear_data_eval.csv").getFile().getPath();

        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        //rr.initialize(new FileSplit(new File("src/main/resources/classification/linear_data_train.csv")));
        rr.initialize(new FileSplit(new File(filenameTrain)));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,0,2);

        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(filenameTest)));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,0,2);
    }
}
