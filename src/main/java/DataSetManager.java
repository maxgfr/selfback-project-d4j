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

    private int batchSize;//500
    private int numClasses;//6
    /**Parameters for my own dataset iterator*/
    private int height;//1
    private int width;//500
    private int depth;//3

    public DataSetManager(int height, int width, int depth) {
        this.height = height;
        this.width  = width;
        this.depth = depth;
    }

    public DataSetManager(int batchSize, int numClasses) {
        this.batchSize = batchSize;
        this.numClasses = numClasses;
    }

    public DataSetIterator createDataSetIteratorForLSTM (File fileData, File fileLabel) throws IOException, InterruptedException {

        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(1,",");
        trainFeatures.initialize(new NumberedFileInputSplit(fileData.getAbsolutePath() + "/%d.csv", 1, 6));

        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(fileLabel.getAbsolutePath() + "/%d.csv", 1, 6));

        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, batchSize, numClasses,
                false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_START);

        /*System.out.println("Normalizer");
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainData);
        trainData.reset();
        trainData.setPreProcessor(normalizer);
        System.out.println("End fit normalizer");*/

        return trainData;

    }

    public DataSetIterator createMyOwnDataSetIterator (File fileData, boolean toTrain) throws IOException, InterruptedException {

        DataInput di = DataInput.getInstance(height,width,depth,fileData);

        INDArrayDataSetIterator iterator = null;

        if (toTrain) {
            iterator = di.getDataSetIteratorToTrain();
        } else {
            iterator = di.getDataSetIteratorToTest();
        }

        return iterator;

    }

    public DataSetIterator createMyOwnDataSetIteratorSameData (File fileData, boolean toTrain) throws IOException, InterruptedException {

        DataInput di = DataInput.getInstance(height,width,depth,fileData);

        INDArrayDataSetIterator iterator = null;

        if (toTrain) {
            iterator = di.getDataSetIteratorSameDataTrain();
        } else {
            iterator = di.getDataSetIteratorSameDataTest();
        }

        return iterator;
    }

    public DataSetIterator createDataSetIteratorTest (File fileData) throws IOException, InterruptedException {

        DataInput di = DataInput.getInstance(height,width,depth,null);

        return di.getDataSetIteratorTest(fileData);

    }

}
