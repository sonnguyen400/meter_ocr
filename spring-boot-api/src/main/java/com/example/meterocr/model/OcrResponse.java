
package com.example.meterocr.model;

import java.util.List;
import java.util.Map;

public class OcrResponse {
    public String meter_reading;
    public String serial_number;
    public String unit;
    public Map<String,Object> engine;
    public List<Box> boxes;
    public Map<String,Object> preprocess;
    public long elapsed_ms;
}
