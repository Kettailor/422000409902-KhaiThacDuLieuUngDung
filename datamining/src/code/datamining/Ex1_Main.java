package code.datamining;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.J48;

/**
 * Bài tập 1: thử nghiệm với J48, OneR, NaiveBayes trên tập dữ liệu khác nhau.
 */
public class Ex1_Main {
    public static void main(String[] args) {
        try {
            // Dataset mẫu (sửa path nếu cần)
            String dataset = "data/weather.nominal.arff";
            String testset = "data/weather.test.arff";

            // Load data
            var data = WekaHelper.loadData(dataset);
            var test = WekaHelper.loadData(testset);

            // Chọn classifiers
            Classifier[] models = {
                new J48(),        // Decision Tree
                new OneR(),       // One Rule
                new NaiveBayes()  // Naive Bayes
            };

            for (Classifier model : models) {
                System.out.println("\n>>> Running " + model.getClass().getSimpleName());

                // Training set
                WekaHelper.evaluateTrainingSet(data, model);

                // Cross-validation
                WekaHelper.evaluateCrossValidation(data, model, 10);

                // Percentage split 66/34
                WekaHelper.evaluatePercentageSplit(data, model, 66);

                // Supplied test set
                WekaHelper.evaluateWithTestSet(data, test, model);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
