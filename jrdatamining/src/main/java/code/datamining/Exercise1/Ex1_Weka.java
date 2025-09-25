package code.datamining.Exercise1;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.rules.OneR;
import weka.classifiers.bayes.NaiveBayes;

import java.util.Random;

public class Ex1_Weka {
    private Instances dataset;

    public void loadDataset(String filePath) throws Exception {
        DataSource source = new DataSource(filePath);
        dataset = source.getDataSet();
        if (dataset.classIndex() == -1)
            dataset.setClassIndex(dataset.numAttributes() - 1);
    }

    public Instances getDataset() {
        return dataset;
    }

    public Evaluation useTrainingSet(Classifier cls) throws Exception {
        Evaluation eval = new Evaluation(dataset);
        cls.buildClassifier(dataset);
        eval.evaluateModel(cls, dataset);
        return eval;
    }

    public Evaluation crossValidation(Classifier cls, int folds) throws Exception {
        Evaluation eval = new Evaluation(dataset);
        cls.buildClassifier(dataset);
        eval.crossValidateModel(cls, dataset, folds, new Random(1));
        return eval;
    }

    public Evaluation percentageSplit(Classifier cls, double percent) throws Exception {
        int trainSize = (int) Math.round(dataset.numInstances() * percent / 100);
        int testSize = dataset.numInstances() - trainSize;
        Instances train = new Instances(dataset, 0, trainSize);
        Instances test = new Instances(dataset, trainSize, testSize);

        Evaluation eval = new Evaluation(train);
        cls.buildClassifier(train);
        eval.evaluateModel(cls, test);
        return eval;
    }

    public void printEvaluation(Evaluation eval) throws Exception {
        System.out.println("Correctly Classified Instances: " + eval.correct());
        System.out.println("Incorrectly Classified Instances: " + eval.incorrect());
        System.out.println("Kappa statistic: " + eval.kappa());
        System.out.println("Mean absolute error (MAE): " + eval.meanAbsoluteError());
        System.out.println("Root mean squared error (RMSE): " + eval.rootMeanSquaredError());
        System.out.println("Relative absolute error: " + eval.relativeAbsoluteError());
        System.out.println("Root relative squared error: " + eval.rootRelativeSquaredError());
        System.out.println(eval.toMatrixString("=== Confusion Matrix ==="));
        System.out.println("======================================");
    }

    public void saveModel(String path, Classifier cls) throws Exception {
        weka.core.SerializationHelper.write(path, cls);
    }

    public Classifier loadModel(String path) throws Exception {
        return (Classifier) weka.core.SerializationHelper.read(path);
    }

    public J48 getJ48() {
        J48 j48 = new J48();
        try {
            j48.setOptions(new String[]{"-C", "0.25", "-M", "2"});
        } catch (Exception e) {
            e.printStackTrace();
        }
        return j48;
    }

    public OneR getOneR() {
        return new OneR();
    }

    public NaiveBayes getNaiveBayes() {
        return new NaiveBayes();
    }
}
