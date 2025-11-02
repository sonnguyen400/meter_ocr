package com.example.meterocr.model;

public enum MeterType {
    LCD("lcd_digital"),
    MECHANIC("flip_mechanical"),
    AUTO("auto");

    public final String value;

    MeterType(String value) {
        this.value = value;
    }
}
