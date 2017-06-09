import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.File;
import java.io.IOException;

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


    public DataSetIterator createTrainDataSetIterator (File file) throws IOException, InterruptedException {

        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(1,",");

        trainFeatures.initialize(new FileSplit(file));

        DataSetIterator iterator = new SequenceRecordReaderDataSetIterator(trainFeatures,batchSize,labelIndex,numClasses);

        System.out.println("Normalizer");

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(iterator);
        iterator.reset();
        iterator.setPreProcessor(normalizer);

        System.out.println("End fit normalizer");

        return iterator;

    }

    public DataSetIterator createDataSetIteror (File file) throws IOException, InterruptedException {

        SequenceRecordReader features = new CSVSequenceRecordReader(1,",");
        features.initialize(new FileSplit(file));

        DataSetIterator iterator = new RecordReaderDataSetIterator(features,batchSize);

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(iterator);
        iterator.setPreProcessor(normalizer);

        return iterator;
    }

}
