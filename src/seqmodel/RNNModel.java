/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package seqmodel;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Properties;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.records.reader.impl.CSVSequenceRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.NumberedFileInputSplit;
import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.canova.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.BaseRecurrentLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author dganguly
 */
public class RNNModel {
    Properties prop;
    DataSetIterator train;
    DataSetIterator test;
    int numInputDimensions;
    int labelIndex;
    
    public RNNModel(String propFile) throws Exception {
        prop = new Properties();
        prop.load(new FileReader(propFile));        
        numInputDimensions = Integer.parseInt(prop.getProperty("vec.numdimensions", "200"));
        labelIndex = prop.getProperty("classify.type", "dec").equals("dec")? numInputDimensions + 1 : numInputDimensions + 2;
    }
    
    /*
    DataSetIterator getAMISentenceIterator(String fileName) throws Exception {
        String trainDvecFile = prop.getProperty(fileName);
        RecordReader recordReader = new CSVRecordReader(0, ",");
        recordReader.initialize(new FileSplit(new ClassPathResource(trainDvecFile).getFile()));
        
        final int labelIndex = numInputDimensions + 1; // the index of the class label
        final int numPossibleLabels = 2; // binary judgements
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, labelIndex, numPossibleLabels);        
        return iterator;
    }
    */
    
    DataSetIterator getAMISentenceIterator(String fileName) throws Exception {
        final int miniBatchSize = 10;
        final int numPossibleLabels = 2;
        String trainDvecFile = prop.getProperty(fileName);
        SequenceRecordReader reader = new CSVSequenceRecordReader(0, "\\s+");
        reader.initialize(new NumberedFileInputSplit(trainDvecFile + ".%d", 0, 9));
        DataSetIterator iterator = new
            SequenceRecordReaderDataSetIterator(reader, miniBatchSize, numPossibleLabels, labelIndex, false);
        return iterator;
    }
    
    MultiLayerNetwork buildRNN(DataSetIterator iter) {
        final int tbpttLength = 10;
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                        .updater(Updater.RMSPROP)
                        .regularization(true).l2(1e-5)
                        .weightInit(WeightInit.XAVIER)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
                        .learningRate(0.0018)
                        .list(2)
                        .layer(0,
                                new GravesLSTM.Builder()
                                .nIn(numInputDimensions)
                                .nOut(200)
                                .activation("softsign").build())
                        .layer(1, new RnnOutputLayer.Builder().activation("softmax")
                                .lossFunction(LossFunctions.LossFunction.MCXENT)
                                .nIn(numInputDimensions)
                                .nOut(2) // 2 for binary classification
                                .build())
                        .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
                        .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));
        return net;
    }
    
    public void evaluate() throws Exception {
        train = getAMISentenceIterator("train.dvec.file");
        test = getAMISentenceIterator("test.dvec.file");
        MultiLayerNetwork rnn = buildRNN(train);
        
        final int nEpochs = 5; //Number of epochs (full passes of training data) to train on
        
        for (int i=0; i<nEpochs; i++) {
            rnn.fit(train);        
        
            Evaluation evaluation = new Evaluation();
            while(test.hasNext()){
                DataSet t = test.next();
                INDArray features = t.getFeatureMatrix();
                INDArray lables = t.getLabels();
                INDArray inMask = t.getFeaturesMaskArray();
                INDArray outMask = t.getLabelsMaskArray();
                INDArray predicted = rnn.output(features, false, inMask, outMask);

                evaluation.evalTimeSeries(lables, predicted, outMask);
            }
            
            train.reset();
            test.reset();
            
            System.out.println(evaluation.stats());
        }
    }
    
    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            args = new String[1];
            System.out.println("Usage: java RNNModel <prop-file>");
            args[0] = "init.properties";
        }

        try {
            RNNModel rnnModel = new RNNModel(args[0]);
            rnnModel.evaluate();
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }        
    }
}
