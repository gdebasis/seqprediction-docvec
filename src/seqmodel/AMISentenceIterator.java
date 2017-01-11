/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package seqmodel;

import java.io.*;
import java.util.*;
import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 *
 * @author dganguly
 */

class FeatureVec {
    INDArray vec;
    int label;

    public FeatureVec(INDArray vec, int label) {
        this.vec = vec;
        this.label = label;
    }    
}

public class AMISentenceIterator implements DataSetIterator {
    int batchSize;
    int vectorSize;
    int truncateLength;
    int cursor;
    File dataSetDir;
    File[] files;
    int labelOffset; // -1 or -2
    
    public AMISentenceIterator(String dataSetDir, int batchSize, int truncateLength, int numInputDimensions, String decPref) {
        this.batchSize = batchSize;
        this.truncateLength = truncateLength;
        this.dataSetDir = new File(dataSetDir);
        files = this.dataSetDir.listFiles();
        labelOffset = decPref.equals("decs")? -2 : -1;
        vectorSize = numInputDimensions;
    }
    
    @Override
    public DataSet next(int batchSize) {
        if (cursor >= files.length) throw new NoSuchElementException();
        try {
            return nextDataSet(batchSize);    
        }
        catch (Exception ex) { ex.printStackTrace(); return null; }        
    }
    
    // Get the INDArray representation from a tab separated line of component values.
    // The first <numdimensions> components are the vector values and the final
    // two are the labels...
    FeatureVec getVec(String str) {
        String[] tokens = str.split("\t");
        int vectorSize = tokens.length - 2;  // trailing two are labels...
        assert(vectorSize == this.vectorSize);
        
        double[] vec = new double[vectorSize];        
        for (int i=0; i < vectorSize; i++) {
            vec[i] = Double.parseDouble(tokens[i]);
        }
        
        FeatureVec fvec = new FeatureVec(Nd4j.create(vec), Integer.parseInt(tokens[tokens.length + labelOffset]));
        return fvec;
    }
    
    private DataSet nextDataSet(int batchSize) throws Exception {
        File[] filesInThisBatch = new File[batchSize];
        int i=0, j=0;
        for (i=0; i<batchSize && cursor<totalExamples(); i++) {
            File datafile = files[cursor];
            filesInThisBatch[i] = datafile;
            cursor++;
        }
        
        int numFilesInThisBatch = Math.min(i, filesInThisBatch.length);
        System.out.println("#files read from batch: " + numFilesInThisBatch);
        int maxLength = 0;

        // Get the maximum length of sequence (each element being a sentence vector)
        for (i = 0; i < numFilesInThisBatch; i++) {
            List<String> docvecs = FileUtils.readLines(filesInThisBatch[i]);
            if (docvecs.size() > maxLength) {
                maxLength = docvecs.size();
            }
        }
        
        if (maxLength > truncateLength) maxLength = truncateLength;
        
        // we have filesInThisBatch.size() examples of varying lengths
        INDArray features = Nd4j.create(numFilesInThisBatch, vectorSize, maxLength);
        INDArray labels = Nd4j.create(numFilesInThisBatch, 2, maxLength);    //Two labels: positive or negative
        
        // Because we are dealing with reviews of different lengths and only one output at the final time step: use padding arrays
        // Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
        //INDArray featuresMask = Nd4j.zeros(numFilesInThisBatch, maxLength);
        //INDArray labelsMask = Nd4j.zeros(numFilesInThisBatch, maxLength);        
        
        for (i = 0; i < numFilesInThisBatch; i++) {

            List<String> docvecs = FileUtils.readLines(filesInThisBatch[i]);
            
            j = 0;
            for (String docvec : docvecs) {
                if (docvec.trim().equals("")) continue;
                FeatureVec fvec = getVec(docvec);
                
                //label = Math.max(label, fvec.label);
                //features.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)}, fvec.vec);
                //labels.putScalar(new int[]{i, fvec.label, j}, 1.0);   //Set label: [0,1] for negative, [1,0] for positive
                
                features.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)}, fvec.vec);
                labels.putScalar(new int[]{i, fvec.label, j}, 1.0);   //Set label: [0,1] for negative, [1,0] for positive
                
                //featuresMask.putScalar(new int[]{i, j}, 1.0); //Word is present (not padding) for this example + time step -> 1.0 in features mask
                //labelsMask.putScalar(new int[]{i, j}, 1.0); //Specify that an output exists at the final time step for this example
                
                j++;
                if (j >= maxLength)
                    break;                
            }            
        }

        return new DataSet(features, labels /*,featuresMask, labelsMask*/);            
    }
    

    @Override
    public int totalExamples() {
        return files.length;
    }

    @Override
    public int inputColumns() {
        return this.vectorSize;
    }

    @Override
    public int totalOutcomes() {
        return 2; // binary classification...
    }

    @Override
    public void reset() {
        cursor = 0;
    }

    public boolean resetSupported() {
        return true;
    }
    
    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        return cursor;
    }

    @Override
    public int numExamples() {
        return totalExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dspp) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public List<String> getLabels() {
        return Arrays.asList("1", "0");
    }

    @Override
    public boolean hasNext() {
        return cursor < numExamples();
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }
}
