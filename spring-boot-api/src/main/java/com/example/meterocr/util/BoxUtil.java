package com.example.meterocr.util;

import com.example.meterocr.model.Box;

import java.util.List;

public class BoxUtil {
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

    public static double calculateDistance(Box box1, Box box2) {
        List<Double> center1 = getCenter(box1);
        List<Double> center2 = getCenter(box2);

        double dx = center1.getFirst() - center2.getFirst();
        double dy = center1.getLast() - center2.getLast();

        return Math.sqrt(dx * dx + dy * dy);
    }
}
