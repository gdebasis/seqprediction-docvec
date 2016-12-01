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
public class SequenceFileGenerator {
    Properties prop;
    IndexReader reader;
    WordVectors wvecs;
    
    public SequenceFileGenerator(String propFile) throws Exception {
        prop = new Properties();
        prop.load(new FileReader(propFile));
        File indexDir = new File(prop.getProperty("index"));
        reader = DirectoryReader.open(FSDirectory.open(indexDir.toPath()));
        
        String docVecFile = prop.getProperty("dvec.out.file");
        wvecs = WordVectorSerializer.loadTxtVectors(new File(docVecFile));
    }

    void saveSegment(List<String> sentences, int sequenceId, boolean train) throws Exception {        
        String rcdType = train? "train" : "test";
        File outRcdTypeDir = new File(prop.getProperty("docvec.dir") + "/" + rcdType);
        if (!outRcdTypeDir.exists())
            outRcdTypeDir.mkdir();
            
        File outDir = new File(outRcdTypeDir.getPath());
        if (!outDir.exists())
            outDir.mkdir();

        System.out.println("Writing sequence " + sequenceId + " in: " + outDir);
        
        FileWriter writer = new FileWriter(outDir.getPath() + "/" + sequenceId + ".txt");
        BufferedWriter bw = new BufferedWriter(writer);
        for (String sentence : sentences) {
            bw.write(sentence);
        }
        
        bw.close();
        writer.close();
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
        return buff.toString();
    }
    
    public void writeDocs() throws Exception {        
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
            
            if (dvec == null)
                continue;
            
            docName = d.get(AMI_FIELDS.FIELD_DOC_NAME);
            segment = Integer.parseInt(d.get(AMI_FIELDS.FIELD_SENTENCE_ID));
            
            String decScore = d.get(AMI_FIELDS.FIELD_DECISION_SCORE);
            String prefScore = d.get(AMI_FIELDS.FIELD_PREF_SCORE);
            
            StringBuilder builder = new StringBuilder();
            builder
                    .append(i)
                    .append("\t")
                    .append(vecToStr(dvec))
                    .append(decScore)
                    .append("\t")
                    .append(prefScore)
                    .append("\n");
            
            // Detect changes in segments            
            if (prevSegment >-1 && segment!=prevSegment) {
                saveSegment(sentencesInSegment, sequenceId++, train);
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
            System.out.println("Usage: java Doc2VecTrainFileGenerator <prop-file>");
            args[0] = "init.properties";
        }
        
        try {
            SequenceFileGenerator gen = new SequenceFileGenerator(args[0]);
            gen.writeDocs();
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }    
    }    
}
