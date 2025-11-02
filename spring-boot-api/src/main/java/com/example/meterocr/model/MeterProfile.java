
package com.example.meterocr.model;

public class MeterProfile {
    public RoiConfig roi = new RoiConfig();
    public PreprocessConfig preprocess = new PreprocessConfig();
    public TessConfig tesseract = new TessConfig();

    public static class RoiConfig {
        public final int[] blackhat_kernel = new int[]{25,7};
        public final int[] close_kernel = new int[]{35,5};
        public final double[] hrange = new double[]{0.03,0.215};
        public final double[] wrange = new double[]{0.25, 1};
        public final double bestMinAspect = 3.2d;
        public final double bestMaxAspect = 7d;
        public final double hr_best = 0.085;
        public final double expand_x = 0.10;
        public final double expand_y = 0.05;
    }

    public static class PreprocessConfig {
        public final double clahe_clip = 3.0;
        public final int[] clahe_tile = new int[]{8,8};
        public final int adaptive_block = 31;
        public final int adaptive_C = 8;
    }

    public static class TessConfig {
        public final int psm = 7;
        public final String whitelist = "0123456789.,";
        public final int dpi = 300;
        public final boolean disable_dawg = true;
        public final int resize_min_height = 60;
    }
}
