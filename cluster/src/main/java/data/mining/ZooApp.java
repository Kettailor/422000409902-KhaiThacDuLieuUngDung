package data.mining;

import weka.core.Instances;

public class ZooApp {
    public static void main(String[] args) {
        try {
            // Load dataset
            PreProcessData preData = new PreProcessData();
            Instances dataSet = preData.loadDateSet("data/zoo.arff");

            ClusterUtils cluster = new ClusterUtils();

            cluster.buildKMeans(dataSet, 7, true);
            System.out.println(cluster.outPutKMeans());

            String[] linkTypes = {
                "SINGLE",
                "COMPLETE",
                "AVERAGE",
                "MEAN",
                "CENTROID",
            };
            
            for (String link : linkTypes) {
                System.out.println("---------------------------------------------------");
                System.out.println("HierarchicalClusterer - " + link);
                cluster.buildHierarchical(dataSet, 2, link);
                System.out.println("---------------------------------------------------\n");
            }


        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
