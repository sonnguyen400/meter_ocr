package com.example.meterocr.service;

import com.example.meterocr.model.Box;
import com.example.meterocr.model.MeterIndexReading;
import com.example.meterocr.util.BoxUtil;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class MultiIndexMeterExtractor {
    public static boolean isThreePhaseMeter(List<Box> boxes) {
        Set<String> foundPhases = new HashSet<>();
        Pattern phasePattern = Pattern.compile("\\bL[123]\\b", Pattern.CASE_INSENSITIVE);

        for (Box box : boxes) {
            String text = box.getText();
            Matcher matcher = phasePattern.matcher(text);

            while (matcher.find()) {
                // Chuẩn hóa về chữ hoa
                foundPhases.add(matcher.group().toUpperCase());
            }
        }

        // Kiểm tra có ít nhất 2 pha
        return foundPhases.size() >= 2;
    }

    public static List<MeterIndexReading> extractPhaseIndices(List<Box> boxes) {
        List<MeterIndexReading> results = new ArrayList<>();

        // Pattern để tìm tên pha (L1, L2, L3)
        Pattern phasePattern = Pattern.compile("\\b[LT][123]\\b", Pattern.CASE_INSENSITIVE);

        // Pattern để tìm số (có thể có dấu phẩy hoặc chấm thập phân)
        Pattern numberPattern = Pattern.compile("^\\d+[.,]?\\d*$");

        // Bước 1: Tìm các box chứa tên pha
        Map<String, Box> phaseBoxes = new HashMap<>();
        for (Box box : boxes) {
            Matcher matcher = phasePattern.matcher(box.getText());
            if (matcher.find()) {
                String phaseName = matcher.group().toUpperCase();
                phaseBoxes.put(phaseName, box);
            }
        }

        // Bước 2: Với mỗi pha, tìm box chứa giá trị gần nhất
        for (Map.Entry<String, Box> entry : phaseBoxes.entrySet()) {
            String phaseName = entry.getKey();
            Box phaseBox = entry.getValue();

            Box closestValueBox = null;
            double minDistance = Double.MAX_VALUE;

            // Tìm box chứa số gần nhất với box của pha
            for (Box box : boxes) {
                // Bỏ qua chính box của pha
                if (box == phaseBox) continue;

                Matcher numberMatcher = numberPattern.matcher(box.getText());
                if (numberMatcher.find()) {
                    double distance = BoxUtil.calculateDistance(phaseBox, box);

                    // Chỉ xét các box ở bên phải hoặc dưới box của pha
                    if (distance < minDistance && isReasonablePosition(phaseBox, box)) {
                        minDistance = distance;
                        closestValueBox = box;
                    }
                }
            }

            // Trích xuất giá trị số
            if (closestValueBox != null) {
                String valueText = closestValueBox.getText();
                Matcher numberMatcher = numberPattern.matcher(valueText);
                if (numberMatcher.find()) {
                    String numberStr = numberMatcher.group().replace(",", ".");
                    results.add(new MeterIndexReading(phaseName, numberStr));
                }
            }
        }

        // Sắp xếp theo tên pha
        results.sort(Comparator.comparing(MeterIndexReading::getIndexName));

        return results;
    }

    private static boolean isReasonablePosition(Box phaseBox, Box valueBox) {
        List<Double> phaseCenter = BoxUtil.getCenter(phaseBox);
        List<Double> valueCenter = BoxUtil.getCenter(valueBox);

        // Giá trị phải ở bên phải hoặc không quá xa về bên trái
        double horizontalDiff = valueCenter.getFirst() - phaseCenter.getFirst();
        double verticalDiff = Math.abs(valueCenter.getLast() - phaseCenter.getLast());

        // Cho phép giá trị ở bên phải hoặc hơi chéo xuống
        // và không quá xa theo chiều dọc
        return horizontalDiff > -50 && verticalDiff < 200;
    }


}
