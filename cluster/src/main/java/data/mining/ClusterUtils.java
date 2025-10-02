package data.mining;

import weka.clusterers.HierarchicalClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.ManhattanDistance;

public class ClusterUtils {
    SimpleKMeans kMeans;
    HierarchicalClusterer hierCluster;
    
    public void buildKMeans(Instances dataSet, int k, boolean isEuclide) throws Exception {
        kMeans = new SimpleKMeans();
        // số cụm
        kMeans.setNumClusters(k);
        //kMeans.setPreserveInstancesOrder(true);
        //kMeans.setSeed(10);
        if (isEuclide) {
            kMeans.setDistanceFunction(new EuclideanDistance(dataSet));
        } else {
            kMeans.setDistanceFunction(new ManhattanDistance(dataSet));
        }
        // build mô hình
        kMeans.buildClusterer(dataSet);
    }

    public String outPutKMeans() {
        return kMeans.toString();
    }
}
