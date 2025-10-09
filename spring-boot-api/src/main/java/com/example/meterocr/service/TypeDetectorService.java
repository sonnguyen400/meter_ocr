
package com.example.meterocr.service;

import com.example.meterocr.model.MeterProfile;
import org.bytedeco.opencv.opencv_imgproc.Vec4iVector;
import org.springframework.stereotype.Service;
import org.bytedeco.opencv.opencv_core.*;

import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import org.bytedeco.javacpp.*;

import java.util.LinkedHashMap;
import java.util.Map;

@Service
public class TypeDetectorService {

    private final MeterProfileService profileService;
    private final LcdRoiDetector roiDetector;

    public TypeDetectorService(MeterProfileService profileService, LcdRoiDetector roiDetector) {
        this.profileService = profileService;
        this.roiDetector = roiDetector;
    }

    public static class DetectResult {
        public final String type;
        public final Map<String, Object> scores;
        public DetectResult(String type, Map<String, Object> scores) { this.type = type; this.scores = scores; }
    }

    public DetectResult detect(byte[] imageBytes) {
        try {
            Mat buf = new Mat(1, imageBytes.length, CV_8U, new BytePointer(imageBytes));
            Mat color = imdecode(buf, IMREAD_COLOR);
            if (color == null || color.empty()) {
                return new DetectResult("lcd_digital", Map.of("error", "decode_failed"));
            }


            Mat gray = new Mat();
            cvtColor(color, gray, COLOR_BGR2GRAY);
            Mat edges = new Mat();
            Canny(gray, edges, 50, 150);
            Vec4iVector lines = new Vec4iVector();
            HoughLinesP(edges, lines, 1, Math.PI/180, 60, 60, 10);
            double angle = estimateSkewAngle2(lines);
            Mat rotated = rotate(color, angle);

            Map<String, MeterProfile> all = profileService.all();
            double bestScore = -1; String bestType = null; Map<String,Object> scoreMap = new LinkedHashMap<>();
            for (String name : all.keySet()) {
                MeterProfile p = all.get(name);
                LcdRoiDetector.RoiResult roi = roiDetector.detect(rotated, p);
                double s = (roi != null ? roi.score : 0.0);
                scoreMap.put(name, Map.of(
                    "score", s,
                    "has_roi", (roi != null && roi.rect != null),
                    "angle", angle
                ));
                if (s > bestScore) { bestScore = s; bestType = name; }
            }

            if (bestType == null) bestType = "lcd_digital";
            return new DetectResult(bestType, scoreMap);
        } catch (Exception e) {
            return new DetectResult("lcd_digital", Map.of("error", e.getMessage()));
        }
    }

    private Mat rotate(Mat src, double angle) {
        Point2f center = new Point2f(src.cols()/2f, src.rows()/2f);
        Mat rotMat = getRotationMatrix2D(center, angle, 1.0);
        Mat dst = new Mat();
        warpAffine(src, dst, rotMat, src.size(), INTER_LINEAR, BORDER_REPLICATE, new Scalar(255,255,255,0));
        return dst;
    }

    private double estimateSkewAngle(Mat lines) {
        if (lines == null || lines.empty()) return 0.0;
        double sum = 0; int cnt = 0;
        for (int i = 0; i < lines.rows(); i++) {
            IntPointer p = new IntPointer(lines.row(i).data());
            int x1 = p.get(0), y1 = p.get(1), x2 = p.get(2), y2 = p.get(3);
            double dx = x2 - x1; double dy = y2 - y1;
            if (Math.hypot(dx, dy) < 30) continue;
            double a = Math.atan2(dy, dx) * 180.0 / Math.PI;
            if (Math.abs(a) < 30) { sum += a; cnt++; }
        }
        return cnt == 0 ? 0.0 : sum / cnt;
    }

    private double estimateSkewAngle2(Vec4iVector lines) {
        if (lines == null || lines.size() == 0) return 0.0;
        double sum = 0;
        int cnt = 0;
        for (int i = 0; i < lines.size(); i++) {
            IntPointer p = new IntPointer(lines.get(i));
            int x1 = p.get(0), y1 = p.get(1), x2 = p.get(2), y2 = p.get(3);
            double dx = x2 - x1;
            double dy = y2 - y1;
            if (Math.hypot(dx, dy) < 30) continue;
            double a = Math.atan2(dy, dx) * 180.0 / Math.PI;
            if (Math.abs(a) < 30) {
                sum += a;
                cnt++;
            }
        }
        return cnt == 0 ? 0.0 : sum / cnt;
    }
}
