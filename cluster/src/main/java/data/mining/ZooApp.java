package data.mining;

import weka.core.Instances;

public class ZooApp {
    public static void main(String[] args) {
        try {
            // Load dataset
            PreProcessData preData = new PreProcessData();
            Instances dataSet = preData.loadDateSet("data/zoo.arff");
            System.out.println(dataSet);

            ClusterUtils cluster = new ClusterUtils();
            cluster.buildKMeans(dataSet, 7, true);
            System.out.println(cluster.outPutKMeans());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
