/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package indexer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.StopFilter;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.FSDirectory;



/**
 * Write the csv formatted annotations in a Lucene index.
 * Each sentence is to be stored as a separate document in the index.
 * Each sentence contains a field which has the document name.
 * @author Debasis
 */

interface AMI_FIELDS {
    String FIELD_SENTENCE_ID = "id"; // globally unique id (doc_name.segmentid.sentenceid) 
    String FIELD_DOC_NAME = "docname";
    String FIELD_CONTENT = "content";  // analyzed content
    String FIELD_DECISION_SCORE = "decision_score";
    String FIELD_PREF_SCORE = "pref_score";
    String FIELD_SPEAKER_ID = "speakerid";
}

public class AMIIndexer {
    
    Properties prop;
    File indexDir;    
    Analyzer analyzer;
    IndexWriter writer;
    List<String> stopwords;
    
    static final String SENTENCE_DELIMS = ".?!";
    static final String DELIMS = ",;.!'\".?$&*(){}[]<>/\\|";

    public AMIIndexer(String propFile) throws Exception {
        prop = new Properties();
        prop.load(new FileReader(propFile));                
        analyzer = constructAnalyzer(prop.getProperty("stopfile"));            
        String indexPath = prop.getProperty("index");        
        indexDir = new File(indexPath);
    }
    
    static public List<String> buildStopwordList(String stopwordFileName) {
        List<String> stopwords = new ArrayList<>();
        String line;

        try (FileReader fr = new FileReader(stopwordFileName);
            BufferedReader br = new BufferedReader(fr)) {
            while ( (line = br.readLine()) != null ) {
                stopwords.add(line.trim());
            }
            br.close();
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
        return stopwords;
    }
    
    static public Analyzer constructAnalyzer(String stopwordFileName) {
        Analyzer eanalyzer = new EnglishAnalyzer(
                StopFilter.makeStopSet(buildStopwordList(stopwordFileName))); // default analyzer
        return eanalyzer;        
    }
    
    public void process() throws Exception {
       System.out.println("Indexing AMI annotations...");
        
        IndexWriterConfig iwcfg = new IndexWriterConfig(analyzer);
        iwcfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);

        writer = new IndexWriter(FSDirectory.open(indexDir.toPath()), iwcfg);        
        indexAnnotation();
        writer.close();
    }
    
    boolean isEOS(String word) {
        for (char eos : AMIIndexer.SENTENCE_DELIMS.toCharArray()) {
            if (word.charAt(0) == eos)
                return true;
        }
        return false;
    }
    
    boolean isPunct(String word) {
        char ch = word.charAt(0);
        return DELIMS.indexOf(ch) >= 0;
    }
    
    int getLabel(String label) {
        if (label.equals("nan"))
            return 0;
        return (int)(Float.parseFloat(label));        
    }
    
    boolean isConsecutiveSegment(String prevSegmentId, String thisSegmentId, String prevFileName, String thisFileName) {
        if (!prevFileName.equals(thisFileName))
            return false;
        int prevSegment = Integer.parseInt(prevSegmentId);
        int thisSegment = Integer.parseInt(thisSegmentId);
        return thisSegment == prevSegment+1;
    }
    
    void indexAnnotation() throws Exception {
        boolean indexSentence = prop.getProperty("doc.unit").equals("sentence"); // sentence/segment
        String annotationFile = prop.getProperty("annotation.csvfile");
        FileReader fr = new FileReader(annotationFile);
        BufferedReader br = new BufferedReader(fr);
        
        String line, prevFileName = null, prevSegmentId = null;
        int sentenceIdInDoc = 0;
        StringBuffer buff = new StringBuffer();
        int dec = 0, pref = 0;
        
        while ((line = br.readLine()) != null) {
            String[] tokens = line.split("\t");
            String word = tokens[0];
            String fileName = tokens[2];
            String segmentId = tokens[3];
                        
            // Propagate the maximum for this sentence/segment
            int label = getLabel(tokens[16]);
            if (label > dec)
                dec = label;
            label = getLabel(tokens[20]);
            if (label > pref)
                pref = label;
            
            if (indexSentence) {
                if (isEOS(word)) {
                    buff.append(word);
                    Document d = constructDoc(buff.toString(), fileName, segmentId, tokens[9], dec, pref);
                    writer.addDocument(d);
                    buff = new StringBuffer();
                    dec = 0;
                    pref = 0;
                    continue;
                }
            }
            else if (prevFileName!= null && prevSegmentId!= null) { // if there is a change in segment id (for same file) or a change in the file name itself
                if (isConsecutiveSegment(prevSegmentId, segmentId, prevFileName, fileName)) {
                    Document d = constructDoc(buff.toString(), fileName, fileName + "_" + segmentId, tokens[9], dec, pref);
                    writer.addDocument(d);
                    buff = new StringBuffer();
                    dec = 0;
                    pref = 0;
                    prevSegmentId = segmentId;
                    prevFileName = fileName;
                    buff.append(word);
                    continue;
                }
            }
            
            prevFileName = fileName;
            prevSegmentId = segmentId;
            
            if (!isPunct(word)) {
                buff.append(" ");
            }
            else if (indexSentence && buff.length() > 0 && isEOS(word)) {
                buff.deleteCharAt(buff.length()-1);
            }
            buff.append(word);
        }
    }
    
    // wordOffset comes from the previous line...
    public Document constructDoc(String sentence, String fileName, String sentenceId, String speaker, int dec, int pref) throws Exception {
        
        System.out.println("Storing sentence/segment " + sentenceId + " of doc: " + fileName);
        
        Document doc = new Document();
        // Meta
        doc.add(new Field(AMI_FIELDS.FIELD_DOC_NAME, fileName, Field.Store.YES, Field.Index.NOT_ANALYZED));
        // sentence id within segment or 
        doc.add(new Field(AMI_FIELDS.FIELD_SENTENCE_ID, sentenceId, Field.Store.YES, Field.Index.NOT_ANALYZED));
        
        // content
        doc.add(new Field(AMI_FIELDS.FIELD_CONTENT, sentence,
                Field.Store.YES, Field.Index.ANALYZED, Field.TermVector.YES));
        
        // class labels
        doc.add(new Field(AMI_FIELDS.FIELD_SPEAKER_ID, speaker, Field.Store.YES, Field.Index.NOT_ANALYZED));
        doc.add(new Field(AMI_FIELDS.FIELD_DECISION_SCORE, String.valueOf(dec), Field.Store.YES, Field.Index.NOT_ANALYZED));
        doc.add(new Field(AMI_FIELDS.FIELD_PREF_SCORE, String.valueOf(pref), Field.Store.YES, Field.Index.NOT_ANALYZED));
                
        return doc;
    }    

    public static void main(String[] args) {
        if (args.length == 0) {
            args = new String[1];
            System.out.println("Usage: java AMIIndexer <prop-file>");
            args[0] = "init.properties";
        }
        
        try {
            AMIIndexer indexer = new AMIIndexer(args[0]);
            indexer.process();
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
        
    }
}
