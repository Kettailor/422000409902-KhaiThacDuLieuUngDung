package data.mining;

import weka.clusterers.HierarchicalClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.ManhattanDistance;

public class ClusterUtils {
    SimpleKMeans kMeans;
    HierarchicalClusterer hierarchical;
    private Instances m_instances;

    // ===== KMeans =====
    public void buildKMeans(Instances dataSet, int k, boolean isEuclide) throws Exception {
        kMeans = new SimpleKMeans();
        kMeans.setNumClusters(k);

        if (isEuclide)
            kMeans.setDistanceFunction(new EuclideanDistance(dataSet));
        else
            kMeans.setDistanceFunction(new ManhattanDistance(dataSet));

        kMeans.buildClusterer(dataSet);
    }

    public void buildHierarchical(Instances dataSet, int numClusters, String linkTypeName) throws Exception {
        this.m_instances = dataSet;
        hierarchical = new HierarchicalClusterer();

        // ✅ Thiết lập thông số qua setOptions()
        String[] options = new String[4];
        options[0] = "-L";
        options[1] = linkTypeName.toUpperCase(); // single / average / complete / ward ...
        options[2] = "-N";
        options[3] = Integer.toString(numClusters);
        hierarchical.setOptions(options);

        hierarchical.buildClusterer(m_instances);

        for (int i = 0; i < m_instances.numInstances(); i++) {
            try {
                int cluster = hierarchical.clusterInstance(m_instances.instance(i));
                if (cluster > 0)
                    System.out.println("Instance " + i + ": Cluster " + cluster);
            } catch (Exception e) {
                System.out.println("Instance " + i + " → không được gán cụm");
            }
        }
    }

    // ===== Output =====
    public String outPutKMeans() {
        return kMeans != null ? kMeans.toString() : "⚠️ KMeans chưa được khởi tạo!";
    }
}
