import au.com.bytecode.opencsv.CSVReader;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by maxime on 15-Jun-17.
 */
public class DataInput {

    private int x;
    private int y;
    private int z;

    public DataInput(int x , int y, int z){
       this.x = x;
       this.y = y;
       this.z = z;
    }

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    public int getZ() {
        return z;
    }

    public void parsingArray (String fileName) throws IOException {
        double[] tab = new double[x * y * z];

        CSVReader reader = new CSVReader(new FileReader(fileName), '\n');

        INDArray array = Nd4j.create(x,y,z);
        array.add();


        //Read CSV line by line and use the string array as you want
        String[] nextLine;
        while ((nextLine = reader.readNext()) != null) {
            if (nextLine != null) {
                String res = Arrays.toString(nextLine);
                res.split(",",1);
                res.split(",",1);
                res.split(",",1);
                res.split(",",1);
                //Verifying the read data here
                System.out.println();
            }
        }
    }

    /*public INDArray getData(){

        try{
            double[] tab = new double[x * y * z];

            int w = 0;

            for(int k = 0; k < z; k++){
                for(int j = 0; j < y; j++){
                    for(int i = 0; i < x; i++){
                        tab[w] = parsingArray();
                        w++;
                    }
                }
            }

            System.out.println("Size " + tab.length);
            INDArray array = Nd4j.create(tab, new int[]{1, 1, 2880});
            return array;

        } catch(IOException e){
            e.printStackTrace();
        }

        return null;
    }*/

    public INDArrayDataSetIterator getDataSetIterator(){
        INDArray negLabel = Nd4j.rand(1, 1);
        negLabel.add(0);
        INDArray posLabel = Nd4j.rand(1, 2);

        List<INDArray> labels = new ArrayList<INDArray>();
        List<INDArray> datas = new ArrayList<INDArray>();

        /*for(int i = 1; i <= 15; i++){
            INDArray arr = Nd4j.create(tab, new int[]{1, 1, 2880});
            if(arr != null){
                datas.add(arr);
                if(arr == 0){
                    labels.add(negLabel);
                }else{
                    labels.add(posLabel);
                }
            }
        }*/

        /**il faut que je crée un arraylist qui contient la data du tout fichier
        /et un avec les labels*/

        ArrayList<Pair> featureAndLabel = new ArrayList<Pair>();
        for(int i = 0; i < datas.size(); i++){
            featureAndLabel.add(new Pair(datas.get(i), labels.get(i)));
            System.out.println("label du array " + i + ": " + labels.get(i));
        }
        System.out.println("Size dataset: " + featureAndLabel.size());
        Iterable featLab = featureAndLabel;
        INDArrayDataSetIterator ds = new INDArrayDataSetIterator(featLab, 1);
        return ds;
    }

    public void getLabel () {

    }
}