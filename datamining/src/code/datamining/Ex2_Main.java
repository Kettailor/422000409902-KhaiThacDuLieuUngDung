package code.datamining;


import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.J48;

/**
 * Bài tập 2: tương tự Ex1 nhưng có thể thay dataset khác (VD: iris.arff).
 */
public class Ex2_Main {
    public static void main(String[] args) {
        try {
            // Dataset mẫu
            String dataset = "data/iris.arff";
            String testset = "data/iris.test.arff";

            var data = WekaHelper.loadData(dataset);
            var test = WekaHelper.loadData(testset);

            Classifier[] models = {
                new J48(),
                new OneR(),
                new NaiveBayes()
            };

            for (Classifier model : models) {
                System.out.println("\n>>> Running " + model.getClass().getSimpleName());

                WekaHelper.evaluateTrainingSet(data, model);
                WekaHelper.evaluateCrossValidation(data, model, 10);
                WekaHelper.evaluatePercentageSplit(data, model, 70);
                WekaHelper.evaluateWithTestSet(data, test, model);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
