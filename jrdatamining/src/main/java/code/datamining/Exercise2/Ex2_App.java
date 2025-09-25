package code.datamining.Exercise2;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

public class Ex2_App {
    public static void main(String[] args) {
        try {
            Ex2_Weka helper = new Ex2_Weka();

            helper.loadDataset("data/buy_comp.arff");

            Classifier j48 = helper.getJ48();

            System.out.println("=== J48: Percentage split (66%) ===");
            Evaluation evalSplit = helper.percentageSplit(j48, 66);
            helper.printEvaluation(evalSplit);


            System.out.println("=== J48: Supplied test set (buy_comp_extra.arff) ===");
            Evaluation evalSupplied = helper.suppliedTestSet(j48, "data/buy_comp_extra.arff");
            helper.printEvaluation(evalSupplied);

            System.out.println("Predictions on supplied test set:");
            helper.printPredictions(j48, "data/buy_comp_extra.arff", 1001);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
