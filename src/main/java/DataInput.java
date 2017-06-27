import au.com.bytecode.opencsv.CSVReader;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.lang.reflect.Array;
import java.util.*;
import java.util.ArrayList;

/**
 * Created by maxime on 15-Jun-17.
 */
public class DataInput {

    private static final int DOWNSTAIRS = 0;
    private static final int JOGGING = 1;
    private static final int SITTING = 2;
    private static final int STANDING = 3;
    private static final int UPSTAIRS = 4;
    private static final int WALKING = 5;

    private int nHeight;//1
    private int nWidth;//500
    private int nDepth;//3

    private List<INDArray> allListData;
    private List<INDArray> allListLabel;

    public DataInput(int x , int y, int z){
        nHeight = x;
        nWidth = y;
        nDepth = z;
        allListData = new ArrayList<INDArray>();
        allListLabel = new ArrayList<INDArray>();
    }

    public INDArrayDataSetIterator getDataSetIterator(File folder){

        fusionList(folder);

        ArrayList<Pair> featureAndLabel = mergeFeaturesWithLabels(allListData,allListLabel);

        Collections.shuffle(featureAndLabel);

        Iterable featLab = featureAndLabel;

        INDArrayDataSetIterator ds = new INDArrayDataSetIterator(featLab, 500);
        return ds;
    }

    public INDArrayDataSetIterator getDataSetIteratorTest(File nameFile){

        List<INDArray> datas = null;

        try {
            int numOfLine = countLine(nameFile.getPath());
            System.out.println("number of line in this CSV : "+numOfLine);
            datas = getData(nameFile.getPath());
        } catch (IOException e) {
            e.printStackTrace();
        }

        List<INDArray> fakeLabel = createFalseRandLabels(6,datas.size());

        ArrayList<Pair> featuresAndLabels = mergeFeaturesWithLabels(datas,fakeLabel);

        Iterable featLab = featuresAndLabels;

        INDArrayDataSetIterator ds = new INDArrayDataSetIterator(featLab, 1);

        return ds;
    }

    private List<INDArray> createFalseRandLabels(int numOutcomes,int numSamples){

        List<INDArray> falseTarget= new ArrayList<INDArray>();

        for(int i=0;i<numSamples;i++){
            INDArray y = Nd4j.zeros(1,numOutcomes);
            if (Math.random()<0.15){
                y.putScalar(0,0,1);
            } else if (Math.random()>0.15 && Math.random()<0.3){
                y.putScalar(0,1,1);
            } else if (Math.random()>0.3 && Math.random()<0.45){
                y.putScalar(0,2,1);
            } else if (Math.random()>0.45 && Math.random()<0.65){
                y.putScalar(0,3,1);
            }else if (Math.random()>0.65 && Math.random()<0.80){
                y.putScalar(0,4,1);
            } else{
                y.putScalar(0,5,1);
            }
            falseTarget.add(y);
        }

        return falseTarget;
    }

    private ArrayList<Pair> mergeFeaturesWithLabels(List<INDArray> features,List<INDArray> labels){

        ArrayList<Pair> featuresAndLabels = new ArrayList<Pair>();

        for(int i=0;i<features.size();i++){
            featuresAndLabels.add(new Pair(features.get(i),labels.get(i)));
        }

        System.out.println("Size dataset: " + featuresAndLabels.size());

        return featuresAndLabels;
    }

    private List<INDArray> getListLabel (int value, int size) throws Exception {
        int numOut = 6;
        INDArray downstairs = Nd4j.zeros(1, numOut);
        INDArray jogging = Nd4j.zeros(1, numOut);
        INDArray sitting = Nd4j.zeros(1, numOut);
        INDArray standing = Nd4j.zeros(1, numOut);
        INDArray upstairs = Nd4j.zeros(1, numOut);
        INDArray walking = Nd4j.zeros(1, numOut);
        downstairs.putScalar(0,0,1);
        jogging.putScalar(0,1,1);
        sitting.putScalar(0,2,1);
        standing.putScalar(0,3,1);
        upstairs.putScalar(0,4,1);
        walking.putScalar(0,5,1);

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

        List<INDArray> datas = new ArrayList<INDArray>();
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
        List<File> list = new ArrayList<File>();
        for (final File fileEntry : folder.listFiles()) {
            if (fileEntry.isDirectory()) {
                listFilesForFolder(fileEntry);
            } else {
                list.add(fileEntry);
            }
        }
        return list;
    }

    private int countLine(String filename) throws IOException {
        InputStream is = new BufferedInputStream(new FileInputStream(filename));
        try {
            byte[] c = new byte[1024];
            int count = 0;
            int readChars = 0;
            boolean empty = true;
            while ((readChars = is.read(c)) != -1) {
                empty = false;
                for (int i = 0; i < readChars; ++i) {
                    if (c[i] == '\n') {
                        ++count;
                    }
                }
            }
            return (count == 0 && !empty) ? 1 : count;
        } finally {
            is.close();
        }
    }
}