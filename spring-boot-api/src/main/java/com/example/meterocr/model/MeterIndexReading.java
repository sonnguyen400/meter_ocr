package com.example.meterocr.model;

public class MeterIndexReading {
    private String indexName;
    private String value;

    public MeterIndexReading(String indexName, String value) {
        this.indexName = indexName;
        this.value = value;
    }

    public String getIndexName() {
        return indexName;
    }

    public void setIndexName(String indexName) {
        this.indexName = indexName;
    }

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }
}
