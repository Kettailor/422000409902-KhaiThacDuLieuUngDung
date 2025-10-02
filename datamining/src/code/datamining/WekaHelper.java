package code.datamining;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

/**
 * Helper class để tải dữ liệu, huấn luyện và đánh giá mô hình bằng Weka.
 * Gom các hàm trùng lặp trong Ex1/Ex2 vào đây.
 */
public class WekaHelper {

    /** Load dataset từ file .arff */
    public static Instances loadData(String path) throws Exception {
        DataSource source = new DataSource(path);
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        return data;
    }

    /** Train model và evaluate trên cùng tập huấn luyện */
    public static void evaluateTrainingSet(Instances data, Classifier model) throws Exception {
        model.buildClassifier(data);
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(model, data);

        System.out.println("=== Training Set Evaluation ===");
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }

    /** Cross-validation k-fold */
    public static void evaluateCrossValidation(Instances data, Classifier model, int folds) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(model, data, folds, new Random(1));

        System.out.println("=== " + folds + "-Fold Cross Validation ===");
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }

    /** Percentage split (train/test) */
    public static void evaluatePercentageSplit(Instances data, Classifier model, double percent) throws Exception {
        int trainSize = (int) Math.round(data.numInstances() * percent / 100);
        int testSize = data.numInstances() - trainSize;

        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);

        model.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(model, test);

        System.out.println("=== Percentage Split (" + percent + "% train, " + (100 - percent) + "% test) ===");
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }

    /** Đánh giá bằng một test set riêng (file .arff khác) */
    public static void evaluateWithTestSet(Instances train, Instances test, Classifier model) throws Exception {
        model.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(model, test);

        System.out.println("=== Supplied Test Set Evaluation ===");
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }
}
