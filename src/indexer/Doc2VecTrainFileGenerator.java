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
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.store.FSDirectory;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;

/**
 * Dump the contents of the index to train doc2vec.
 * @author dganguly
 */
public class Doc2VecTrainFileGenerator {
    Properties prop;
    IndexReader reader;
    WordVectors wvecs;
    
    public Doc2VecTrainFileGenerator(String propFile) throws Exception {
        prop = new Properties();
        prop.load(new FileReader(propFile));
        File indexDir = new File(prop.getProperty("index"));
        reader = DirectoryReader.open(FSDirectory.open(indexDir.toPath()));
        
        String docVecFile = prop.getProperty("dvec.file");
        wvecs = WordVectorSerializer.loadTxtVectors(new File(docVecFile));
    }

    void saveSegment(List<String> sentences, String docName, int sequenceId) throws Exception {
        String outFilePath = prop.getProperty("docvec.train.dir");
        File outDir = new File(outFilePath + "/docvec." + docName);
        outDir.mkdir();
        
        FileWriter writer = new FileWriter(outDir.getPath() + "/" + sequenceId);
        BufferedWriter bw = new BufferedWriter(writer);
        for (String sentence : sentences) {
            bw.write(sentence);
        }
        
        bw.close();
        writer.close();
    }
    
    public void writeDocs() throws Exception {        
        int numDocs = reader.numDocs();
        int prevSegment = -1, segment = 0;
        List<String> sentencesInSegment = new ArrayList<>();
        int sequenceId = 0;
        String prevDocName = null, docName = null;
        
        for (int i=0; i < numDocs; i++) {
            Document d = reader.document(i);
            double[] dvec = wvecs.getWordVector("DOCNO_" + i);
            
            if (dvec == null)
                continue;
            
            docName = d.get(AMI_FIELDS.FIELD_DOC_NAME);
            String content = d.get(AMI_FIELDS.FIELD_CONTENT);
            segment = Integer.parseInt(d.get(AMI_FIELDS.FIELD_SENTENCE_ID));
            
            content = preProcess(
                    AMIIndexer.constructAnalyzer(prop.getProperty("stopfile")),
                    content);
            if (content.trim().length() == 0)
                continue;
            
            String decScore = d.get(AMI_FIELDS.FIELD_DECISION_SCORE);
            String prefScore = d.get(AMI_FIELDS.FIELD_PREF_SCORE);
            
            StringBuilder builder = new StringBuilder();
            builder
                    .append(i)
                    .append("\t")
                    .append(content)
                    .append("\t")
                    .append(decScore)
                    .append("\t")
                    .append(prefScore)
                    .append("\n");
            
            // Detect changes in segments            
            if (prevSegment >-1 && segment!=prevSegment) {
                saveSegment(sentencesInSegment, docName, sequenceId++);
                sentencesInSegment = new ArrayList<>();
            }
            
            // Detect changes in documents
            if (prevDocName!=null && !prevDocName.equals(docName)) {
                sequenceId = 0; // reset sequence id if there's a change in docname
            }
            
            sentencesInSegment.add(builder.toString());
            prevSegment = segment;
            prevDocName = docName;                     
        }
        
        saveSegment(sentencesInSegment, docName, sequenceId++); // last batch saved
        reader.close();
        
    }
    
    String preProcess(Analyzer analyzer, String text) throws Exception {

        StringBuffer tokenizedContentBuff = new StringBuffer();
        TokenStream stream = analyzer.tokenStream("dummy", new StringReader(text));
        CharTermAttribute termAtt = stream.addAttribute(CharTermAttribute.class);
        stream.reset();

        while (stream.incrementToken()) {
            String term = termAtt.toString();
            term = term.toLowerCase();
            tokenizedContentBuff.append(term).append(" ");
        }
        
        stream.end();
        stream.close();
        return tokenizedContentBuff.toString();
    }
    
    public static void main(String[] args) {
        if (args.length == 0) {
            args = new String[1];
            System.out.println("Usage: java Doc2VecTrainFileGenerator <prop-file>");
            args[0] = "init.properties";
        }
        
        try {
            Doc2VecTrainFileGenerator gen = new Doc2VecTrainFileGenerator(args[0]);
            gen.writeDocs();
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }    
    }    
}
