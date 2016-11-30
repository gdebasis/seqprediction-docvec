/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package doc2vec;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.documentiterator.LabelsSource;

/**
 *
 * @author Debasis
 */
public class Doc2VecGenerator {

    Properties prop;
    
    // The docfile is a single tab separated file... each line in the file
    // representing a new document...
    // <DOCID> \t <TEXT>    
    String docFileName;
    File docFile;
    ParagraphVectors vec;
    int minwordfreq;
    
    public Doc2VecGenerator(String propFile) throws Exception {
        prop = new Properties();
        prop.load(new FileReader(propFile));
        
        docFileName = prop.getProperty("docvec.train.file");
        docFile = new File(docFileName);
        
        minwordfreq = Integer.parseInt(prop.getProperty("minwordfreq", "2"));
    }
    
    void learnDocEmbeddings() throws Exception {
        int numDimensions = Integer.parseInt(prop.getProperty("vec.numdimensions", "200"));
        
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
        
        learnDocEmbeddings();
        String outDocVecFile = prop.getProperty("dvec.file");
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
