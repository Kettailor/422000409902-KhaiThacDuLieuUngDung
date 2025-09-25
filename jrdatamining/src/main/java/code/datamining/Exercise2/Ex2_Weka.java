package code.datamining.Exercise2;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;

import java.util.Random;

public class Ex2_Weka {
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

    public Evaluation suppliedTestSet(Classifier cls, String testFile) throws Exception {
        DataSource source = new DataSource(testFile);
        Instances testSet = source.getDataSet();
        if (testSet.classIndex() == -1)
            testSet.setClassIndex(testSet.numAttributes() - 1);

        Evaluation eval = new Evaluation(dataset);
        cls.buildClassifier(dataset);
        eval.evaluateModel(cls, testSet);
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

    public void printPredictions(Classifier cls, String testFile, int startUserId) throws Exception {
    DataSource source = new DataSource(testFile);
    Instances testSet = source.getDataSet();
    if (testSet.classIndex() == -1)
        testSet.setClassIndex(testSet.numAttributes() - 1);

    for (int i = 0; i < testSet.numInstances(); i++) {
        Instance inst = testSet.instance(i);
        double[] dist = cls.distributionForInstance(inst);
        int predIndex = (int) cls.classifyInstance(inst);
        String predicted = testSet.classAttribute().value(predIndex);
        String actual = testSet.classAttribute().value((int) inst.classValue());
        boolean correct = actual.equals(predicted);

        System.out.printf(
            "User #%d | Actual=%s | Predicted=%s | Prob=%.3f | Result=%s%n",
            startUserId + i,
            actual,
            predicted,
            dist[predIndex],
            (correct ? "PASS" : "FAIL")
        );
    }
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
}

