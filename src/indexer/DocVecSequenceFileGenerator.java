/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package indexer;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Fields;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.MultiFields;
import org.apache.lucene.store.FSDirectory;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;

/**
 * This file generates the dataset in time-series format to be loaded for
 * the RNN training...
 * 
 * @author dganguly
 */
public class DocVecSequenceFileGenerator {
    Properties prop;
    IndexReader reader;
    WordVectors wvecs;
    
    public DocVecSequenceFileGenerator(String propFile) throws Exception {
        prop = new Properties();
        prop.load(new FileReader(propFile));
        File indexDir = new File(prop.getProperty("index"));
        reader = DirectoryReader.open(FSDirectory.open(indexDir.toPath()));
        
        String docVecFile = prop.getProperty("dvec.out.file");
        wvecs = WordVectorSerializer.loadTxtVectors(new File(docVecFile));
        
        prepareOutputFolders();
    }
    
    final void prepareOutputFolders() throws Exception {
        String[] rcdTypes = { "train", "test" }; 
        
        for (String rcdType : rcdTypes) {
            File outRcdTypeDir = new File(prop.getProperty("docvec.dir") + "/" + rcdType);
            if (!outRcdTypeDir.exists())
                outRcdTypeDir.mkdir();
            else {
                outRcdTypeDir.delete();
                outRcdTypeDir.mkdir();                
            }
        }
    }

    void saveSegment(List<String> sentences, int sequenceId, boolean train) throws Exception {
        
        boolean subsampling = Boolean.parseBoolean(prop.getProperty("subsampling", "false"));
        if (subsampling)
            sentences = subsampleSegments(sentences);
        
        String rcdType = train? "train" : "test";
        File outDir = new File(prop.getProperty("docvec.dir") + "/" + rcdType);
            
        System.out.println("Writing sequence " + sequenceId + " in: " + outDir);
        
        FileWriter writer = new FileWriter(outDir.getPath() + "/vecs." + sequenceId + ".txt");        
        BufferedWriter bw = new BufferedWriter(writer);
        
        for (String sentence : sentences) {
            bw.write(sentence + "\n");
        }

        bw.close();
        
        writer.close();
    }
    
    void clearFiles() {
        for (String rcdType : "train test".split("\\s+")) {
            File outDir = new File(prop.getProperty("docvec.dir") + "/" + rcdType);
            if (!outDir.exists()) {
                outDir.mkdir();
            }
            else {
                File[] files = outDir.listFiles();
                for (File f : files) {
                    f.delete();
                }     
            }
        }        
    }
    
    void removeEmptyFiles() {
        for (String rcdType : "train test".split("\\s+")) {
            File outDir = new File(prop.getProperty("docvec.dir") + "/" + rcdType);
            File[] files = outDir.listFiles();
            for (File f : files) {
                if (f.length() == 0)
                    f.delete();
            }            
        }
    }
    
    // Get the number of records for train/test split
    long getNumDocs() throws Exception {
        Fields fields = MultiFields.getFields(reader);
        return fields.terms(AMI_FIELDS.FIELD_DOC_NAME).size();
    }
    
    String vecToStr(double[] vec) {
        StringBuffer buff = new StringBuffer();
        for (double c : vec) {
            buff.append(c).append("\t");
        }
        buff.deleteCharAt(buff.length()-1);
        return buff.toString();
    }
    
    ArrayList<String> subsampleSegments(List<String> segments) {
        ArrayList<String> subsampled = new ArrayList<>();
        String classifyType = prop.getProperty("classify.type", "decs");
        int offset = classifyType.equals("decs")? -2 : -1;
        int numSegments = segments.size();
        int subSampleSize = Integer.parseInt(prop.getProperty("subsample.size", "2"));
        for (int i=0; i < numSegments; i++) {
            String segment = segments.get(i);
            String[] tokens = segment.split("\\s+");

            int label = Integer.parseInt(tokens[tokens.length + offset]);
            
            if (label == 1) {
                if (i>=subSampleSize) {
                    for (int k=1; k <= subSampleSize; k++) {
                        subsampled.add(segments.get(i-k));                        
                    }
                }
                subsampled.add(segment);
                if (i < numSegments-subSampleSize) {
                    for (int k=1; k <= subSampleSize; k++) {
                        subsampled.add(segments.get(i+k));   
                    }
                }
                i = i + subSampleSize;
            }
        }
        return subsampled;
    }
    
    public void writeDocs() throws Exception {        
        
        clearFiles();
        
        int numDocs = reader.numDocs();
        int prevSegment = -1, segment = 0;
        List<String> sentencesInSegment = new ArrayList<>();
        String prevDocName = null, docName = null;
        int sequenceId = 0;
        
        int numRcds = (int)getNumDocs();
        float train_test_ratio = Float.parseFloat(prop.getProperty("train-test-ratio", "0.8"));
        int numTrainingRcds = (int)(train_test_ratio*numRcds);
        int rcdCount = 0;
        boolean train = true;
        
        for (int i=0; i < numDocs; i++) {
            Document d = reader.document(i);
            double[] dvec = wvecs.getWordVector("DOCNO_" + i);
            
            if (dvec == null) {
                System.err.println("No vec found for doc# " + i);
                continue;
            }
            
            docName = d.get(AMI_FIELDS.FIELD_DOC_NAME);
            segment = Integer.parseInt(d.get(AMI_FIELDS.FIELD_SENTENCE_ID));
            
            int decScore = Integer.parseInt(d.get(AMI_FIELDS.FIELD_DECISION_SCORE)) > 0? 1 : 0;
            int prefScore = Integer.parseInt(d.get(AMI_FIELDS.FIELD_PREF_SCORE)) > 0? 1 : 0;
            
            StringBuilder builder = new StringBuilder();
            builder
                    .append(vecToStr(dvec))
                    .append("\t")
                    .append(decScore)
                    .append("\t")
                    .append(prefScore);
            
            
            // Detect changes in segments            
            if (prevSegment >-1 && segment!=prevSegment) {
                saveSegment(sentencesInSegment, /*decBuffer, prefBuffer,*/ sequenceId++, train);
                sentencesInSegment = new ArrayList<>();
            }
            
            // Detect changes in documents
            if (prevDocName!=null && !prevDocName.equals(docName)) {
                rcdCount++;
                if (rcdCount >= numTrainingRcds) {
                    train = false;
                    sequenceId = 0;
                }
            }
            
            sentencesInSegment.add(builder.toString());
            prevSegment = segment;
            prevDocName = docName;                     
        }
        
        saveSegment(sentencesInSegment, sequenceId++, train); // last batch saved
        reader.close();        
    }
    
    public static void main(String[] args) {
        if (args.length == 0) {
            args = new String[1];
            System.out.println("Usage: java DocVecSequenceFileGenerator <prop-file>");
            args[0] = "init.properties";
        }
        
        try {
            DocVecSequenceFileGenerator gen = new DocVecSequenceFileGenerator(args[0]);
            gen.writeDocs();
            gen.removeEmptyFiles();
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }    
    }    
}
