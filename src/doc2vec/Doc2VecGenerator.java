/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package doc2vec;

import indexer.AMIIndexer;
import indexer.AMI_FIELDS;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.StringReader;
import java.util.Properties;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.store.FSDirectory;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.LuceneSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;

/**
 *
 * @author Debasis
 */

class LuceneDocIterator implements SentenceIterator {
    IndexReader reader;
    int docId;
    Analyzer analyzer;
    int numDocs;
    
    public LuceneDocIterator(File indexDir, String stopFile) throws Exception {
        reader = DirectoryReader.open(FSDirectory.open(indexDir.toPath()));        
        docId = 0;
        analyzer = AMIIndexer.constructAnalyzer(stopFile);
        numDocs = reader.numDocs();
    }

    @Override
    public String nextSentence() {
        String content = null;
        try {
            Document doc = reader.document(docId);
            content = preProcess(analyzer, doc.get(AMI_FIELDS.FIELD_CONTENT));
            docId++;
        }
        catch (Exception ex) { ex.printStackTrace(); }
        return content;
    }

    @Override
    public boolean hasNext() {
        return docId < numDocs;
    }

    @Override
    public void reset() {
        docId = 0;
    }

    @Override
    public void finish() {
        try {
            reader.close();
        }
        catch (Exception ex) { ex.printStackTrace(); }
    }

    @Override
    public SentencePreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public void setPreProcessor(SentencePreProcessor spp) {
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
}

public class Doc2VecGenerator {

    Properties prop;
    
    // The docfile is a single tab separated file... each line in the file
    // representing a new document...
    // <DOCID> \t <TEXT>    
    ParagraphVectors vec;
    int minwordfreq;
    String stopFile;
    int numDimensions;
    
    public Doc2VecGenerator(String propFile) throws Exception {
        prop = new Properties();
        prop.load(new FileReader(propFile));
        minwordfreq = Integer.parseInt(prop.getProperty("minwordfreq", "2"));
        stopFile = prop.getProperty("stopfile");
        numDimensions = Integer.parseInt(prop.getProperty("vec.numdimensions", "200"));
    }

    // Read sentences from Lucene index
    void learnDocEmbeddings(File indexDir) throws Exception {
        SentenceIterator iter = new LuceneDocIterator(indexDir, stopFile);
        InMemoryLookupCache cache = new InMemoryLookupCache();

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());
        
        LabelsSource source = new LabelsSource("DOCNO_");
        
        vec = new ParagraphVectors.Builder()
                .minWordFrequency(minwordfreq)
                .iterations(3)
                .epochs(5)
                .layerSize(numDimensions)
                .learningRate(0.025)
                .labelsSource(source)
                .windowSize(5)
                .iterate(iter)
                .vocabCache(cache)
                .tokenizerFactory(t)
                .sampling(0.1f)
                .workers(4)
                .trainWordVectors(true)
                .build();
        vec.fit();        
    }
    
    // Read sentences from new-line separated file
    void learnDocEmbeddings(String docFile) throws Exception {
        
        SentenceIterator iter = new BasicLineIterator(docFile);
        InMemoryLookupCache cache = new InMemoryLookupCache();

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());
        
        LabelsSource source = new LabelsSource("DOCNO_");
        
        vec = new ParagraphVectors.Builder()
                .minWordFrequency(minwordfreq)
                .iterations(3)
                .epochs(5)
                .layerSize(numDimensions)
                .learningRate(0.025)
                .labelsSource(source)
                .windowSize(5)
                .iterate(iter)
                .vocabCache(cache)
                .tokenizerFactory(t)
                .sampling(0.1f)
                .workers(4)
                .trainWordVectors(true)
                .build();
        vec.fit();
    }
    
    public void processAll() throws Exception {
        System.out.println("Learning doc embeddings");
        
        /* Call this to train docvec on a file (each line a sentence)
        String docFileName = prop.getProperty("docvec.in.file");
        learnDocEmbeddings(docFileName);
        */
        
        /* Call this to train doc2vec on the Lucene index..        
        */
        String indexPath = prop.getProperty("index");
        learnDocEmbeddings(new File(indexPath));
        
        String outDocVecFile = prop.getProperty("dvec.out.file");
        BufferedWriter bw = new BufferedWriter(new FileWriter(outDocVecFile));
        
        System.out.println("Writing out the doc vectors for indexing...");
        
        WordVectorSerializer.writeWordVectors(vec, bw);
        
        bw.close();
    }
    
    public static void main(String[] args) {
        if (args.length == 0) {
            args = new String[1];
            System.out.println("Usage: java Doc2VecGenerator <prop-file>");
            args[0] = "init.properties";
        }

        try {
            Doc2VecGenerator doc2vecGen = new Doc2VecGenerator(args[0]);
            doc2vecGen.processAll();
        }
        catch (Exception ex) { ex.printStackTrace(); }
    }
}
