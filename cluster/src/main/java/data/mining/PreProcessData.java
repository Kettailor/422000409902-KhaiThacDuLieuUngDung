package data.mining;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class PreProcessData {
    DataSource source;
    Instances dataSet;

    public Instances loadDateSet(String filename) throws Exception {
        source = new DataSource(filename);
        dataSet = source.getDataSet();
        return dataSet;
    }
}
