
package com.example.meterocr.model;

public class MeterProfile {
    public RoiConfig roi = new RoiConfig();
    public PreprocessConfig preprocess = new PreprocessConfig();
    public TessConfig tesseract = new TessConfig();

    public static class RoiConfig {
        public int[] blackhat_kernel = new int[]{25,7};
        public int[] close_kernel = new int[]{35,5};
        public double[] hrange = new double[]{0.01,0.12};
        public double hr_best = 0.06;
        public double expand_x = 0.10;
        public double expand_y = 0.05;
    }

    public static class PreprocessConfig {
        public double clahe_clip = 3.0;
        public int[] clahe_tile = new int[]{8,8};
        public int adaptive_block = 31;
        public int adaptive_C = 8;
    }

    public static class TessConfig {
        public int psm = 7;
        public String whitelist = "0123456789.,";
        public int dpi = 300;
        public boolean disable_dawg = true;
        public int resize_min_height = 60;
    }
}
