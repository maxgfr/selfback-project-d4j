import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.CachingDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by maxime on 02-Jun-17.
 */
public class DataSetManager {

    /** Instance unique non préinitialisée */
    private static DataSetManager INSTANCE = null;
    /** Defines number of samples that going to be propagated through the network.*/
    private int batchSize;//500
    /** Classes : downstairs,jogging,etc.*/
    private int numClasses;//6
    /**3 if the label index is on the 4th column*/
    private int labelIndex;//3
    /**4 For feedforward network*/
    private List<DataSet> listDataSetTrain;
    private List<DataSet> listDataSetTest;

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

    public List<DataSet> getListTrainData () {
        return listDataSetTrain;
    }

    public List<DataSet> getListTestData () {
        return listDataSetTest;
    }

    public DataSetIterator createDataSetIteratorForRNN (File file, boolean train) throws IOException, InterruptedException {

        DataSetIterator iterator = null;

        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(1,",");

        trainFeatures.initialize(new FileSplit(file));

        if (train){
            iterator = new SequenceRecordReaderDataSetIterator(trainFeatures,batchSize,numClasses,labelIndex);
        } else {
            iterator = new RecordReaderDataSetIterator(trainFeatures,batchSize);
        }
        System.out.println("Normalizer");

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(iterator);
        iterator.setPreProcessor(normalizer);

        System.out.println("End fit normalizer");

        return iterator;

    }

    public void createDataSetIteratorForFeedForward (File file) throws IOException, InterruptedException {

        RecordReader trainFeatures = new CSVRecordReader(1,",");
        trainFeatures.initialize(new FileSplit(file));
        DataSetIterator iterator = new RecordReaderDataSetIterator(trainFeatures,batchSize,labelIndex,numClasses);

        System.out.println("Normalizer");

        listDataSetTrain = new LinkedList<DataSet>();

        listDataSetTest = new LinkedList<DataSet>();

        while (iterator.hasNext()) {
            DataSet ds = iterator.next();
            SplitTestAndTrain testAndTrain = ds.splitTestAndTrain(0.65);  //Use 65% of data for training

            DataSet trainingData = testAndTrain.getTrain();
            DataSet testData = testAndTrain.getTest();

            //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(trainingData);
            normalizer.transform(trainingData);
            normalizer.transform(testData);

            listDataSetTrain.add(trainingData);
            listDataSetTest.add(trainingData);
        }

        //Collections.shuffle(listDataSetTrain);
        //Collections.shuffle(listDataSetTest);

        System.out.println("End Normalizer");

    }

}
