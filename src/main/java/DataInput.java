import au.com.bytecode.opencsv.CSVReader;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by maxime on 15-Jun-17.
 */
public class DataInput {

    private int nHeight;//1
    private int nWidth;//500
    private int nDepth;//3

    private List<INDArray> allListData;
    private List<INDArray> allListLabel;

    private static final int DOWNSTAIRS = 0;
    private static final int JOGGING = 1;
    private static final int SITTING = 2;
    private static final int STANDING = 3;
    private static final int UPSTAIRS = 4;
    private static final int WALKING = 5;

    public DataInput(int x , int y, int z){
        nHeight = x;
        nWidth = y;
        nDepth = z;
        allListData = new LinkedList<INDArray>();
        allListLabel = new LinkedList<INDArray>();
    }

    public INDArrayDataSetIterator getDataSetIterator(File folder, int value){

        fusionList(folder);

        ArrayList<Pair> featureAndLabel = new ArrayList<Pair>();
        for(int i = 0; i < allListData.size(); i++){
            featureAndLabel.add(new Pair(allListData.get(i), allListLabel.get(i)));
        }
        System.out.println("Size dataset: " + featureAndLabel.size());
        Iterable featLab = featureAndLabel;
        INDArrayDataSetIterator ds = new INDArrayDataSetIterator(featLab, 500);
        return ds;
    }

    private List<INDArray> getListLabel (int value, int size) throws Exception {
        INDArray downstairs = Nd4j.zeros(1, 1);
        INDArray jogging = Nd4j.zeros(1, 1);
        INDArray sitting = Nd4j.zeros(1, 1);
        INDArray standing = Nd4j.zeros(1, 1);
        INDArray upstairs = Nd4j.zeros(1, 1);
        INDArray walking = Nd4j.zeros(1, 1);
        downstairs.put(0,0,0);
        jogging.put(0,0,1);
        sitting.put(0,0,2);
        standing.put(0,0,3);
        upstairs.put(0,0,4);
        walking.put(0,0,5);

        List<INDArray> listLabel = new ArrayList<INDArray>();

        switch (value) {
            case DOWNSTAIRS :
                for (int i = 0; i<size ; i++) {
                    listLabel.add(downstairs);
                }
                break;
            case JOGGING :
                for (int i = 0; i<size ; i++) {
                    listLabel.add(jogging);
                }
                break;
            case SITTING :
                for (int i = 0; i<size ; i++) {
                    listLabel.add(sitting);
                }
                break;
            case STANDING :
                for (int i = 0; i<size ; i++) {
                    listLabel.add(standing);
                }
                break;
            case UPSTAIRS :
                for (int i = 0; i<size ; i++) {
                    listLabel.add(upstairs);
                }
                break;
            case WALKING :
                for (int i = 0; i<size ; i++) {
                    listLabel.add(walking);
                }
                break;
            default : throw new Exception("Label unavailable");

        }
        return listLabel;
    }

    private List<INDArray> getData (String fileName) throws IOException {

        CSVReader reader = new CSVReader(new FileReader(fileName), '\n');

        List<INDArray> datas = new LinkedList<INDArray>();
        String[] nextLine;
        int k = 0;
        int j = 0;
        INDArray myArray = Nd4j.zeros(1,nDepth, nHeight,nWidth);
        nextLine = reader.readNext(); //skip the first line

        while ((nextLine = reader.readNext()) != null) {
            if (nextLine != null) {
                String res = Arrays.toString(nextLine);
                String base = res.substring(1,res.length()-1); // remove [ and ]
                for (int i = 0; i < 3; i++) {
                    if (j==nWidth) {//500
                        j=0;
                        k++;
                        if (k==nDepth) { //3
                            k=0;
                            datas.add(myArray);
                            //System.out.println(myArray);
                            myArray = Nd4j.zeros(1,nDepth, nHeight,nWidth);
                        }
                    }
                    double value = Double.parseDouble(base.split(",")[i]);
                    myArray.putScalar(0,k,0,j,value);
                    j++;
                }
            }
        }
        return datas;
    }

    private void fusionList (File folder) {

        List<File> listFile = listFilesForFolder(folder);

        int nbElements = listFile.size();

        for (int i = 0; i<nbElements ; i++){
            try {
                List<INDArray> datas = getData(listFile.get(i).getPath());
                int size = datas.size();
                List<INDArray> labels = getListLabel(i,size);
                for (INDArray array : datas) {
                    allListData.add(array);
                }
                for (INDArray label : labels) {
                    allListLabel.add(label);
                }
            } catch (IOException e) {
                e.printStackTrace();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private List<File> listFilesForFolder (File folder) {
        List<File> list = new LinkedList<File>();
        for (final File fileEntry : folder.listFiles()) {
            if (fileEntry.isDirectory()) {
                listFilesForFolder(fileEntry);
            } else {
                list.add(fileEntry);
            }
        }
        return list;
    }
}