import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by maxime on 02-Jun-17.
 */
public class DataSetManager {

    /** Instance unique non préinitialisée */
    private static DataSetManager INSTANCE = null;
    /** Defines number of samples that going to be propagated through the network.*/
    private int batchSize;//50
    /** Classes : downstairs,jogging,etc.*/
    private int numClasses;//6
    /**3 if the label index is on the 4th column*/
    private int labelIndex;//3

    /** Constructor private */
    private DataSetManager(int batchSize, int numClasses, int labelIndex) {
        this.batchSize = batchSize;
        this.numClasses = numClasses;
        this.labelIndex = labelIndex;
    }

    /** Singleton */
    public static synchronized DataSetManager getInstance(int batchSize, int numClasses, int labelIndex) {
        if (INSTANCE == null)
        { 	INSTANCE = new DataSetManager(batchSize,numClasses,labelIndex);
        }
        return INSTANCE;
    }

    public List<DataSetIterator> createDataSetIterator (List<File> list) throws IOException, InterruptedException {

        List<DataSetIterator> listDataSetIterator  = new LinkedList<DataSetIterator>();

        for (File file : list) {
            //Load the training data:
            RecordReader rr = new CSVRecordReader(1,",");
            rr.initialize(new FileSplit(file));
            listDataSetIterator.add (new RecordReaderDataSetIterator(rr,batchSize,labelIndex,numClasses));
        }

        return listDataSetIterator;
    }

    public DataSet createOneDataSetAll(File file) throws IOException, InterruptedException {

        RecordReader rr = new CSVRecordReader(1,",");
        rr.initialize(new FileSplit(file));

        RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(rr,batchSize,labelIndex,numClasses);
        iterator.setCollectMetaData(true);

        DataSet allData = iterator.next();
        allData.shuffle(123);

        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);  //Use 65% of data for training

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        //Let's view the example metadata in the training and test sets:
        List<RecordMetaData> trainMetaData = trainingData.getExampleMetaData(RecordMetaData.class);
        List<RecordMetaData> testMetaData = testData.getExampleMetaData(RecordMetaData.class);

        //Let's show specifically which examples are in the training and test sets, using the collected metadata
        System.out.println("  +++++ Training Set Examples MetaData +++++");
        String format = "%-20s\t%s";
        for(RecordMetaData recordMetaData : trainMetaData){
            System.out.println(String.format(format, recordMetaData.getLocation(), recordMetaData.getURI()));
            //Also available: recordMetaData.getReaderClass()
        }
        System.out.println("\n\n  +++++ Test Set Examples MetaData +++++");
        for(RecordMetaData recordMetaData : testMetaData){
            System.out.println(recordMetaData.getLocation());
        }


        //Normalize data as per basic CSV example
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData);     //Apply normalization to the training data
        normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set

        return  trainingData;

    }
}
