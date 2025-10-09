
package com.example.meterocr.model;

import org.bytedeco.opencv.opencv_core.Mat;

public class PreprocessResult {
    private final Mat rotated;
    private final Mat bin;
    private final double angle;

    public PreprocessResult(Mat rotated, Mat bin, double angle) {
        this.rotated = rotated;
        this.bin = bin;
        this.angle = angle;
    }

    public Mat rotated() { return rotated; }
    public Mat bin() { return bin; }
    public double angle() { return angle; }
}
