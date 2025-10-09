
package com.example.meterocr.model;

import java.util.List;

public class Box {
    private String text;
    private double conf;
    private List<List<Double>> polygon; // 4-point polygon from PaddleOCR

    public Box() {}
    public Box(String text, double conf) { this.text = text; this.conf = conf; }

    public String getText() { return text; }
    public void setText(String text) { this.text = text; }
    public double getConf() { return conf; }
    public void setConf(double conf) { this.conf = conf; }
    public List<List<Double>> getPolygon() { return polygon; }
    public void setPolygon(List<List<Double>> polygon) { this.polygon = polygon; }
}
