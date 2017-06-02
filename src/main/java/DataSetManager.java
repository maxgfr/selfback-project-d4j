import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;

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

    /** Constructeur privé */
    private DataSetManager(int batchSize) {
        this.batchSize = batchSize;
    }

    /** Point d'accès pour l'instance unique du singleton */
    public static synchronized DataSetManager getInstance(int batchSize) {
        if (INSTANCE == null)
        { 	INSTANCE = new DataSetManager(batchSize);
        }
        return INSTANCE;
    }

    public List<DataSetIterator> createDataSetIterator (List<String> list) throws IOException, InterruptedException {

        List<DataSetIterator> listDataSetIterator  = new LinkedList<DataSetIterator>();

        for (String path : list) {
            //Load the training data:
            RecordReader rr = new CSVRecordReader(1,",");
            normalizeCSV(path);
            rr.initialize(new FileSplit(new File(path)));
            listDataSetIterator.add (new RecordReaderDataSetIterator(rr,batchSize));
        }

        return listDataSetIterator;
    }

    private void normalizeCSV (String path) {
        Schema schema = new Schema.Builder()
                .addColumnString("time")
                .addColumnString("x")
                .addColumnString("y")
                .addColumnInteger("z")
                .build();

        TransformProcess tp = new TransformProcess.Builder(schema)
                //Let's remove some column we don't need
                .removeColumns("time")
                .build();

        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("Storm Reports Record Reader Transform");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);


        File f = new File(path);
        String newPath = f.getAbsolutePath();

        // read the data file
        JavaRDD<String> lines = sc.textFile(newPath);
        // convert to Writable
        JavaRDD<List<Writable>> stormReports = lines.map(new StringToWritablesFunction(new CSVRecordReader()));
        // run our transform process
        JavaRDD<List<Writable>> processed = SparkTransformExecutor.execute(stormReports,tp);
        // convert Writable back to string for export
        JavaRDD<String> toSave= processed.map(new WritablesToStringFunction(","));

        toSave.saveAsTextFile(newPath+"uo");

    }
}
