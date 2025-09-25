package code.datamining.Exercise1;
import weka.classifiers.Classifier;

public class Ex1_App {
    public static void main(String[] args) {
        try {
            Ex1_Weka helper = new Ex1_Weka();

            helper.loadDataset("data/weather.nominal.arff");

            Classifier j48 = helper.getJ48();
            System.out.println("=== J48: Use training set ===");
            helper.printEvaluation(helper.useTrainingSet(j48));

            System.out.println("=== J48: 10-fold Cross-validation ===");
            helper.printEvaluation(helper.crossValidation(j48, 10));

            System.out.println("=== J48: Percentage split 66% ===");
            helper.printEvaluation(helper.percentageSplit(j48, 66));

            Classifier oneR = helper.getOneR();
            System.out.println("=== OneR: 10-fold Cross-validation ===");
            helper.printEvaluation(helper.crossValidation(oneR, 10));

            Classifier nb = helper.getNaiveBayes();
            System.out.println("=== Naive Bayes: 10-fold Cross-validation ===");
            helper.printEvaluation(helper.crossValidation(nb, 10));

            helper.saveModel("j48.model", j48);
            Classifier loaded = helper.loadModel("j48.model");
            System.out.println("Loaded model: " + loaded.getClass().getSimpleName());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
