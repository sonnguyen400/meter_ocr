package com.example.meterocr.service;

import com.example.meterocr.model.Box;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

public class SerialFinder {

    // Khoảng cách Euclide giữa hai điểm
    private static double calculateDistance(List<Double> p1, List<Double> p2) {
        return Math.sqrt(Math.pow(p1.getFirst() - p2.getFirst(), 2) + Math.pow(p1.getLast() - p2.getLast(), 2));
    }


    public static String findSerialNumber(List<Box> boxes) {
        List<String> serialKeywords = Arrays.asList("NO", "S/N", "SERIAL", "MÃ SỐ", "SERI");

        // 1. Phân loại Rects
        List<Box> serialLabels = new ArrayList<>();
        List<Box> potentialValues = new ArrayList<>();

        for (Box box : boxes) {
            String normalizedText = box.getText().toUpperCase(Locale.ROOT).replace(" ", "").toUpperCase();
            boolean isLabel = false;

            for (String keyword : serialKeywords) {
                if (normalizedText.contains(keyword)) {
                    serialLabels.add(box);
                    isLabel = true;
                    break;
                }
            }

            if (!isLabel) {
                potentialValues.add(box);
            }
        }

        if (serialLabels.isEmpty()) {
            return "";
        }

        // 2. Tìm kiếm giá trị lân cận gần nhất
        String bestSerial = null;
        double minDistance = Double.MAX_VALUE;

        // Ngưỡng khoảng cách tối đa theo trục Y (để đảm bảo cùng dòng)
        final int Y_THRESHOLD = 20;
        // Ngưỡng khoảng cách tối đa tổng thể (cần điều chỉnh thực tế)
        final int DISTANCE_THRESHOLD = 200;

        for (Box label : serialLabels) {
            List<Double> labelCenter = getCenter(label);

            for (Box valueRect : potentialValues) {
                String valueText = valueRect.getText().trim();

                if (valueText.length() < 3) continue;

                List<Double> valueCenter = getCenter(valueRect);

                double distance = calculateDistance(labelCenter, valueCenter);
                double distanceY = Math.abs(labelCenter.getLast() - valueCenter.getLast());

                if (distanceY < Y_THRESHOLD && distance < DISTANCE_THRESHOLD && distance < minDistance) {
                    minDistance = distance;
                    bestSerial = valueText;
                }
            }
        }

        // 3. Kết luận và Xử lý TH Serial nằm trong Label
        if (bestSerial != null) {
            return bestSerial;
        } else {
            // Fallback: Kiểm tra lại các Rect là Label ban đầu
            for (Box label : serialLabels) {
                // Tìm số/chuỗi dài sau dấu hai chấm (ví dụ: "Serial No: 999XYZ")
                String[] parts = label.getText().split(":.");
                if (parts.length > 1) {
                    String potentialValue = parts[parts.length - 1].trim();
                    // Lấy phần tử đầu tiên sau dấu hai chấm (nếu có dấu cách) và kiểm tra độ dài
                    String finalValue = potentialValue.split("\\s+")[0];
                    if (finalValue.length() >= 5) {
                        return finalValue;
                    }
                }
            }
            return "";
        }
    }

    public static List<Double> getCenter(Box box) {
        if (box == null || box.getPolygon().size() != 4) {
            // Trường hợp lỗi, trả về một điểm mặc định
            return List.of(0.0, 0.0);
        }

        double minX = box.getPolygon().stream().mapToDouble(p -> p.stream().findFirst().orElse(0d)).min().orElse(0);
        double maxX = box.getPolygon().stream().mapToDouble(p -> p.stream().findFirst().orElse(0d)).max().orElse(0);
        double minY = box.getPolygon().stream().mapToDouble(p -> p.isEmpty() ? 0d : p.getLast()).min().orElse(0);
        double maxY = box.getPolygon().stream().mapToDouble(p -> p.isEmpty() ? 0d : p.getLast()).max().orElse(0);

        double centerX = (minX + maxX) / 2;
        double centerY = (minY + maxY) / 2;

        return List.of(centerX, centerY);
    }
}
