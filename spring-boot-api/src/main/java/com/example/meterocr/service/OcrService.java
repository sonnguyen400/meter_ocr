
package com.example.meterocr.service;

import com.example.meterocr.model.Box;
import com.example.meterocr.model.MeterProfile;
import com.example.meterocr.model.PreprocessResult;
import com.example.meterocr.util.ImageUtils;
import com.example.meterocr.util.RegexUtils;
import org.springframework.stereotype.Service;

import java.awt.image.BufferedImage;
import java.util.*;

import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.javacpp.BytePointer;

@Service
public class OcrService {
    private final PreprocessService preprocess;
    private final PaddleClient paddle;
    private final TesseractService tesseract;
    private final LcdRoiDetector lcdDetector;
    private final MeterProfileService profileService;
    private final TypeDetectorService typeDetector;
    private final RectifyService rectifier;
    private final ClassifierClient classifier;

    public OcrService(PreprocessService preprocess, PaddleClient paddle, TesseractService tesseract,
                      LcdRoiDetector lcdDetector, MeterProfileService profileService,
                      TypeDetectorService typeDetector, RectifyService rectifier, ClassifierClient classifier) {
        this.preprocess = preprocess;
        this.paddle = paddle;
        this.tesseract = tesseract;
        this.lcdDetector = lcdDetector;
        this.profileService = profileService;
        this.typeDetector = typeDetector;
        this.rectifier = rectifier;
        this.classifier = classifier;
    }

    public Map<String, Object> process(byte[] imageBytes, String type, boolean debug) throws Exception {
        long t0 = System.currentTimeMillis();
        String resolvedType; Map<String,Object> detectScores = null;
        if (type == null || type.isBlank() || type.equalsIgnoreCase("auto")) {
            String clsType = null; java.util.Map<String,Double> clsProbs = null; String used = null;
            try {
                var cls = classifier.classify(imageBytes);
                clsType = cls.type; clsProbs = cls.probs; used = cls.used;
            } catch (Exception ignore) { }
            if (clsProbs != null) {
                double maxp = 0.0; for (Double v: clsProbs.values()) maxp = Math.max(maxp, v);
                if (maxp >= 0.55) { resolvedType = clsType; } else { resolvedType = null; }
                java.util.Map<String,Object> clsMap = new java.util.LinkedHashMap<>();
                clsMap.put("probs", clsProbs); clsMap.put("used", used);
                if (clsMap.get("probs") != null) { if (detectScores == null) detectScores = new java.util.LinkedHashMap<>(); detectScores.put("classifier", clsMap); }
            } else {
                resolvedType = null;
            }
            if (resolvedType == null) {
                var det = typeDetector.detect(imageBytes);
                resolvedType = det.type; detectScores = det.scores;
            }
        } else { resolvedType = type; }

        MeterProfile profile = profileService.get(resolvedType);

        PreprocessResult pp = preprocess.preprocess(imageBytes, profile);

        BytePointer out = new BytePointer();
        imencode(".jpg", pp.rotated(), out);
        byte[] rotatedBytes = new byte[(int) out.limit()];
        out.get(rotatedBytes);

        // 1) PaddleOCR on full image
        List<Box> boxes = paddle.ocr(rotatedBytes);

        // 2) Reading/Serial from Paddle full image
        String readingPaddle = boxes.stream().map(Box::getText).map(RegexUtils::normalize)
                .filter(RegexUtils::looksLikeReading).max(Comparator.comparingInt(String::length)).orElse(null);
        double confReadingPaddle = boxes.stream()
                .filter(b -> RegexUtils.looksLikeReading(RegexUtils.normalize(b.getText())))
                .mapToDouble(Box::getConf).max().orElse(0.0);
        String serialPaddle = boxes.stream().map(Box::getText).map(RegexUtils::normalize)
                .filter(RegexUtils::looksLikeSerial).findFirst().orElse(null);

        // 3) LCD ROI detection
        LcdRoiDetector.RoiResult roiRes = lcdDetector.detect(pp.rotated(), profile);
        Rect lcdRect = roiRes != null ? roiRes.rect : null;

        // 3.5) Rectify ROI (smart)
        Mat rectifiedColor = null; Mat rectifiedBin = null; org.bytedeco.opencv.opencv_core.Size rectifiedSize = null;
        if (lcdRect != null && lcdRect.width() > 0 && lcdRect.height() > 0) {
            var rect = rectifier.rectifySmart(pp.rotated(), pp.bin(), lcdRect, boxes);
            if (rect != null) { rectifiedColor = rect.color; rectifiedBin = rect.bin; rectifiedSize = rect.size; }
        }

        // 4) PaddleOCR again on ROI (prefer rectified)
        if (lcdRect != null && lcdRect.width() > 0 && lcdRect.height() > 0) {
            Mat roiColor = (rectifiedColor != null ? rectifiedColor : new Mat(pp.rotated(), lcdRect).clone());
            BytePointer out2 = new BytePointer();
            imencode(".jpg", roiColor, out2);
            byte[] roiBytes = new byte[(int) out2.limit()];
            out2.get(roiBytes);
            List<Box> roiBoxes = paddle.ocr(roiBytes);
            String readingPaddleRoi = roiBoxes.stream().map(Box::getText).map(RegexUtils::normalize)
                    .filter(RegexUtils::looksLikeReading).max(Comparator.comparingInt(String::length)).orElse(null);
            if (readingPaddleRoi != null) {
                readingPaddle = readingPaddleRoi;
                confReadingPaddle = roiBoxes.stream()
                        .filter(b -> RegexUtils.looksLikeReading(RegexUtils.normalize(b.getText())))
                        .mapToDouble(Box::getConf).max().orElse(confReadingPaddle);
            }
        }

        // 5) Tesseract on ROI bin (with resize)
        String readingTess = null; double confReadingTess = 0.0;
        if (lcdRect != null && lcdRect.width() > 0 && lcdRect.height() > 0) {
            Mat roiBin = (rectifiedBin != null ? rectifiedBin : new Mat(pp.bin(), lcdRect).clone());
            Mat roiBinScaled = ImageUtils.resizeToMinHeight(roiBin, profile.tesseract.resize_min_height);
            BufferedImage binBI = ImageUtils.matToBufferedImage(roiBinScaled);
            String tessRaw = tesseract.ocrDigits(binBI, profile);
            readingTess = RegexUtils.extractBestReading(tessRaw);
            confReadingTess = (readingTess != null && !readingTess.isBlank()) ? Math.min(0.99, 0.7 + 0.3*Math.min(1.0, roiRes.score)) : 0.0;
        }

        // 6) Ensemble
        String finalReading; String primary; double finalConf;
        if (confReadingPaddle >= confReadingTess) {
            finalReading = (readingPaddle != null) ? readingPaddle : readingTess;
            primary = "paddleocr"; finalConf = Math.max(confReadingPaddle, confReadingTess);
        } else {
            finalReading = (readingTess != null) ? readingTess : readingPaddle;
            primary = "tesseract"; finalConf = Math.max(confReadingTess, confReadingPaddle);
        }

        Map<String, Object> res = new LinkedHashMap<>();
        res.put("meter_reading", finalReading);
        res.put("serial_number", serialPaddle);
        res.put("unit", "kWh");
        Map<String,Object> engine = new LinkedHashMap<>();
        engine.put("primary", primary);
        engine.put("fallback", primary.equals("paddleocr") ? "tesseract" : "paddleocr");
        engine.put("confidence", finalConf);
        res.put("engine", engine);
        res.put("boxes", boxes);
        Map<String,Object> ppinfo = new LinkedHashMap<>();
        ppinfo.put("rotation_deg", pp.angle());
        ppinfo.put("binarized", true);
        if (lcdRect != null) {
            ppinfo.put("lcd_roi", Map.of("x", lcdRect.x(), "y", lcdRect.y(), "w", lcdRect.width(), "h", lcdRect.height(), "score", roiRes.score));
        }
        ppinfo.put("profile", resolvedType);
        res.put("preprocess", ppinfo);
        res.put("elapsed_ms", System.currentTimeMillis() - t0);

        if (debug) {
            try {
                Mat dbg = pp.rotated().clone();
                if (lcdRect != null) {
                    com.example.meterocr.util.DebugOverlayUtils.drawPaddleBoxes(dbg, boxes);
                    rectangle(dbg, lcdRect, new Scalar(0,255,0,0), 2, LINE_AA, 0);
                }
                putText(dbg, "type="+resolvedType, new Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0,255,0,0), 2, LINE_AA, false);
                BytePointer dbgBuf = new BytePointer();
                imencode(".png", dbg, dbgBuf);
                byte[] b = new byte[(int)dbgBuf.limit()];
                dbgBuf.get(b);
                String b64 = java.util.Base64.getEncoder().encodeToString(b);
                res.put("debug_overlay_b64", b64);
            } catch (Throwable t) { /* ignore debug errors */ }
        }

        if (detectScores != null) res.put("detect_scores", detectScores);
        return res;
    }
}
