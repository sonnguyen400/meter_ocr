package com.example.meterocr.model;

import org.bytedeco.opencv.opencv_core.Rect;

public class RoiResult {
    private Rect rect;
    private double score;
    private MeterType meterType;
    private MeterProfile meterProfile;

    public RoiResult(Rect rect, double score, MeterType meterType, MeterProfile meterProfile) {
        this.rect = rect;
        this.score = score;
        this.meterType = meterType;
        this.meterProfile = meterProfile;
    }

    public Rect getRect() {
        return rect;
    }

    public void setRect(Rect rect) {
        this.rect = rect;
    }

    public double getScore() {
        return score;
    }

    public void setScore(double score) {
        this.score = score;
    }

    public MeterType getMeterType() {
        return meterType;
    }

    public void setMeterType(MeterType meterType) {
        this.meterType = meterType;
    }

    public MeterProfile getMeterProfile() {
        return meterProfile;
    }

    public void setMeterProfile(MeterProfile meterProfile) {
        this.meterProfile = meterProfile;
    }

    @Override
    public String toString() {
        return "RoiResult{" +
                "rect=" + String.format("([x:%s, y:%s][x:%s, y:%s] width:%s, height:%s)", rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height(), rect.width(), rect.height()) +
                ", score=" + score +
                ", meterType=" + meterType +
                ", meterProfile=" + meterProfile +
                '}';
    }
}
