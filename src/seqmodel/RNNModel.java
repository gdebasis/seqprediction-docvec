/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package seqmodel;

import java.io.FileReader;
import java.util.Properties;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
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
    AMISentenceIterator train;
    AMISentenceIterator test;
    int numInputDimensions;
    String classificationType;
    
    static final int TRUNCATED_BPP_LEN = 1;
    static final int SEED = 1000;
    static final int NUM_EPOCHS = 1; //Number of epochs (full passes of training data) to train on
    static final int BATCH_SIZE = 10;
    static final int TRUNCATE_LEN = 25; // max 25 sentences in a segment...
    static final int NUM_DIMENSIONS_LSTM = 5; // max 25 sentences in a segment...
            
    public RNNModel(String propFile) throws Exception {
        prop = new Properties();
        prop.load(new FileReader(propFile));        
        numInputDimensions = Integer.parseInt(prop.getProperty("vec.numdimensions", "200"));
        classificationType = prop.getProperty("classify.type", "decs");
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
    
    /*
    DataSetIterator getAMISentenceIterator(String dirName) throws Exception {
        final int miniBatchSize = 10;
        final int numPossibleLabels = 2;
        
        SequenceRecordReader featureReader = new CSVSequenceRecordReader(0, "\t");
        SequenceRecordReader labelReader = new CSVSequenceRecordReader(0, "\t");

        File dataSetDir = new File(dirName);
        File[] seqFiles = dataSetDir.listFiles(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return name.startsWith("vecs");
            }
        });
        
        featureReader.initialize(new NumberedFileInputSplit(dirName + "vecs.%d.txt", 0, seqFiles.length));
        labelReader.initialize(new NumberedFileInputSplit(dirName + "/" + classificationType + ".%d.txt", 0, seqFiles.length));
        
        DataSetIterator iterator = new
            SequenceRecordReaderDataSetIterator(featureReader, labelReader, miniBatchSize, numPossibleLabels, false);
        return iterator;
    }
    */
    
    MultiLayerNetwork buildRNN(AMISentenceIterator iter) {
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(SEED)
                        .iterations(10)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        //.optimizationAlgo(OptimizationAlgorithm.LBFGS)
                        //.optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                        .updater(Updater.RMSPROP)
                        //.updater(Updater.ADAGRAD)
                        //.updater(Updater.SGD)
                        //.regularization(true).l2(0.0001)
                        //.regularization(true).l1(0.001)
                        .weightInit(WeightInit.RELU)
                        //.weightInit(WeightInit.UNIFORM)
                        //.weightInit(WeightInit.XAVIER)
                        //.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
                        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).gradientNormalizationThreshold(1.0)
                        .learningRate(0.001)
                        .list(2)
                        .layer(0,
                                new GravesLSTM.Builder()
                                .nIn(numInputDimensions)
                                .nOut(NUM_DIMENSIONS_LSTM)
                                //.activation("softsign").build())
                                .activation("softmax")
                                //.activation("tanh")
                                .build())
                        /*        
                        .layer(1,
                                new GravesLSTM.Builder()
                                .nIn(25)
                                .nOut(NUM_DIMENSIONS_LSTM)
                                //.activation("softsign").build())
                                .activation("softmax")
                                //.activation("tanh")
                                .build())
                        */        
                        .layer(1, new RnnOutputLayer.Builder()
                                .activation("softmax")
                                //.activation("tanh")
                                //.activation("sigmoid")
                                //.lossFunction(LossFunctions.LossFunction.MCXENT)
                                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                                .nIn(NUM_DIMENSIONS_LSTM)
                                .nOut(2) // 2 for binary classification
                                .build())
                        .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(TRUNCATED_BPP_LEN).tBPTTBackwardLength(TRUNCATED_BPP_LEN)
                        //.backpropType(BackpropType.Standard)
                        .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));
        return net;
    }
    
    AMISentenceIterator getAMISentenceIterator(String dataDir) {
        AMISentenceIterator iter = new AMISentenceIterator(dataDir, BATCH_SIZE, TRUNCATE_LEN, numInputDimensions, classificationType);
        return iter;
    }
    
    public void evaluate() throws Exception {
        String dataSetBaseDir = prop.getProperty("docvec.dir");
        train = getAMISentenceIterator(dataSetBaseDir + "/train/");
        test = getAMISentenceIterator(dataSetBaseDir + "/test/");
        
        System.out.println("Traning num_instances: " + train.numExamples());
        System.out.println("Test num_instances: " + test.numExamples());
        
        //+++ DEBUG:
        //System.out.println("train:");
        //train.reset();
        //while (train.hasNext()) {
        //    System.out.println(train.next());
        //}
        
        //System.out.println("test:");
        //test.reset();
        //while (test.hasNext()) {
        //    System.out.println(test.next());
        //}
        //--- DEBUG
        
        
        MultiLayerNetwork rnn = buildRNN(train);
        
        for (int i=0; i<NUM_EPOCHS; i++) {
            System.out.println("Epoch: " + i);
            rnn.fit(train);        
            
            Evaluation evaluation = new Evaluation();
            while(test.hasNext()){
                DataSet t = test.next();
                INDArray features = t.getFeatureMatrix();
                INDArray lables = t.getLabels();
                //INDArray inMask = t.getFeaturesMaskArray();
                //INDArray outMask = t.getLabelsMaskArray();
                INDArray predicted = null;
                predicted = rnn.output(features, false/*, inMask, outMask*/);

                evaluation.evalTimeSeries(lables, predicted/*, outMask*/);
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
