import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.File;
import java.io.IOException;

/**
 * Created by maxime on 02-Jun-17.
 */
public class DataSetManager {

    /** Instance of our class */
    private static DataSetManager INSTANCE = null;
    /** Defines number of samples that going to be propagated through the network.*/
    private int batchSize;//500
    /** Classes : downstairs,jogging,etc.*/
    private int numClasses;//6
    /**3-4-5 Parameters for my own dataset iterator*/
    private int height;//1
    private int width;//500
    private int depth;//3

    /** Constructor private */
    private DataSetManager(int height, int width, int depth) {
        this.height = height;
        this.width  = width;
        this.depth = depth;
    }

    /** Constructor private */
    private DataSetManager(int batchSize, int numClasses) {
        this.batchSize = batchSize;
        this.numClasses = numClasses;
    }

    /** Singleton */
    public static synchronized DataSetManager getInstance(int batchSize, int numClasses) {
        if (INSTANCE == null) {
            INSTANCE = new DataSetManager(batchSize,numClasses);
        }
        return INSTANCE;
    }

    /** Singleton */
    public static synchronized DataSetManager getInstance(int height, int width, int depth) {
        if (INSTANCE == null) {
            INSTANCE = new DataSetManager(height,width,depth);
        }
        return INSTANCE;
    }

    public DataSetIterator createDataSetIteratorForLSTM (File fileData, File fileLabel) throws IOException, InterruptedException {

        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(1,",");
        trainFeatures.initialize(new NumberedFileInputSplit(fileData.getAbsolutePath() + "/%d.csv", 1, 6));

        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(fileLabel.getAbsolutePath() + "/%d.csv", 1, 6));

        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, batchSize, numClasses,
                false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        System.out.println("Normalizer");

        //Normalize the training data
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainData);
        trainData.reset();
        trainData.setPreProcessor(normalizer);

        System.out.println("End fit normalizer");

        return trainData;

    }

    public DataSetIterator createMyOwnDataSetIterator (File fileData, boolean toTrain) throws IOException, InterruptedException {

        DataInput di = new DataInput(height,width,depth);

        INDArrayDataSetIterator iterator = null;

        if (toTrain) {
            iterator = di.getDataSetIterator(fileData);
        } else {
            iterator = di.getDataSetIteratorTest(fileData);
        }

        return iterator;

    }

}
