/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package indexer;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Document;

/**
 *
 * @author dganguly
 */
public class WordVecSequenceFileGenerator extends DocVecSequenceFileGenerator {
    Analyzer analyzer;

    public WordVecSequenceFileGenerator(String propFile) throws Exception {
        super(propFile);
        analyzer = new WhitespaceAnalyzer(); //AMIIndexer.constructAnalyzer(prop.getProperty("stopfile"));        
    }

    
    String embedWords(Document d) throws Exception {
        String content = d.get(AMI_FIELDS.FIELD_CONTENT);
        int decScore = Integer.parseInt(d.get(AMI_FIELDS.FIELD_DECISION_SCORE)) > 0? 1 : 0;
        int prefScore = Integer.parseInt(d.get(AMI_FIELDS.FIELD_PREF_SCORE)) > 0? 1 : 0;
        
        List<String> tokens = new ArrayList<>();
        TokenStream stream = analyzer.tokenStream("dummy", new StringReader(content));
        CharTermAttribute termAtt = stream.addAttribute(CharTermAttribute.class);
        stream.reset();
        
        StringBuffer buff = new StringBuffer();
        boolean labelsStoredWithWords = Boolean.parseBoolean(prop.getProperty("word.labels", "false"));
        
        while (stream.incrementToken()) {
            String term = termAtt.toString().toLowerCase();
            String[] wordAndLabel = null;
            
            if (labelsStoredWithWords) {
                wordAndLabel = term.split("\\" + AMIIndexer.WORD_LABEL_DELIM);
                term = wordAndLabel[0]; // the first part is the word
                decScore = Integer.parseInt(wordAndLabel[1]);
                prefScore = Integer.parseInt(wordAndLabel[2]);
            }
            
            double[] x = wvecs.getWordVector(term);
            if (x == null) {
                System.err.println("No vec found for word " + term);
                continue;
            }
            
            String wvec = vecToStr(x);
            if (decScore > 1)
                decScore = 1;
            if (prefScore > 1)
                prefScore = 1;
            buff.append(wvec).append("\t").append(decScore).append("\t").append(prefScore).append("\n");
        }
        stream.close();
        
        return buff.toString();
    }
    
    @Override
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
            
            docName = d.get(AMI_FIELDS.FIELD_DOC_NAME);
            segment = Integer.parseInt(d.get(AMI_FIELDS.FIELD_SENTENCE_ID));
            
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

            String line = embedWords(d);
            if (!line.trim().equals(""))
                sentencesInSegment.add(line);
            
            prevSegment = segment;
            prevDocName = docName;                     
        }
        
        saveSegment(sentencesInSegment, sequenceId++, train); // last batch saved
        reader.close();        
    }
    
    public static void main(String[] args) {
        if (args.length == 0) {
            args = new String[1];
            System.out.println("Usage: java WordVecSequenceFileGenerator <prop-file>");
            args[0] = "init.properties";
        }
        
        try {
            WordVecSequenceFileGenerator gen = new WordVecSequenceFileGenerator(args[0]);
            gen.writeDocs();
            gen.removeEmptyFiles();
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }    
    }    
}
