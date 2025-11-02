package com.example.meterocr.service;

import com.example.meterocr.model.MeterProfile;
import com.example.meterocr.model.MeterType;
import com.example.meterocr.model.RoiResult;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.CHAIN_APPROX_SIMPLE;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.GaussianBlur;
import static org.bytedeco.opencv.global.opencv_imgproc.LINE_8;
import static org.bytedeco.opencv.global.opencv_imgproc.MORPH_BLACKHAT;
import static org.bytedeco.opencv.global.opencv_imgproc.MORPH_CLOSE;
import static org.bytedeco.opencv.global.opencv_imgproc.MORPH_OPEN;
import static org.bytedeco.opencv.global.opencv_imgproc.MORPH_RECT;
import static org.bytedeco.opencv.global.opencv_imgproc.RETR_TREE;
import static org.bytedeco.opencv.global.opencv_imgproc.THRESH_BINARY;
import static org.bytedeco.opencv.global.opencv_imgproc.THRESH_BINARY_INV;
import static org.bytedeco.opencv.global.opencv_imgproc.THRESH_OTSU;
import static org.bytedeco.opencv.global.opencv_imgproc.boundingRect;
import static org.bytedeco.opencv.global.opencv_imgproc.contourArea;
import static org.bytedeco.opencv.global.opencv_imgproc.createCLAHE;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.findContours;
import static org.bytedeco.opencv.global.opencv_imgproc.getStructuringElement;
import static org.bytedeco.opencv.global.opencv_imgproc.morphologyEx;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;
import static org.bytedeco.opencv.global.opencv_imgproc.threshold;

@Service
public class MeterDisplayDetectionService {
    private final MeterProfileService meterProfileService;

    public MeterDisplayDetectionService(MeterProfileService meterProfileService) {
        this.meterProfileService = meterProfileService;
    }

    public RoiResult detectMeterDisplay(Mat image, MeterType meterType, boolean debug) {
        if (meterType == MeterType.AUTO) {
            // Thử cả hai phương pháp và chọn kết quả tốt nhất
            RoiResult lcdROI = this.detectLCDDisplay(image, this.meterProfileService.get(MeterType.LCD), debug);
            RoiResult mechROI = this.detectMechanicalDisplay(image, this.meterProfileService.get(MeterType.MECHANIC), debug);
            if (Objects.isNull(lcdROI) || Objects.isNull(mechROI)) return Optional.ofNullable(lcdROI).orElse(mechROI);
            else return lcdROI.getScore() > mechROI.getScore() ? lcdROI : mechROI;
        } else if (meterType == MeterType.LCD) {
            return this.detectLCDDisplay(image, this.meterProfileService.get(MeterType.LCD), debug);
        } else {
            return this.detectMechanicalDisplay(image, this.meterProfileService.get(MeterType.MECHANIC), debug);
        }
    }


    private RoiResult detectLCDDisplay(Mat image, MeterProfile profile, boolean debug) {
        Mat gray = new Mat();
        cvtColor(image, gray, COLOR_BGR2GRAY);

        // Black Hat: làm nổi vùng tối trên nền sáng
        Mat blackhat = applyBlackHat(gray, profile.roi.blackhat_kernel);

        // Threshold
        Mat thresh = new Mat();
        threshold(blackhat, thresh, 0, 255, THRESH_BINARY + THRESH_OTSU);

        // Morphological Close
        Mat closed = applyClose(thresh, profile.roi.close_kernel);
        imwrite("test45.jpg", thresh);

        // Tìm contours
        MatVector contours = new MatVector();
        Mat hierarchy = new Mat();
        findContours(closed, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        // Tìm ROI tốt nhất
        RoiResult bestROI = this.findBestRoi(contours, image, MeterType.LCD, profile);

        if (debug) imwrite("debug_lcd_closed.png", closed);
        if (bestROI != null && debug) {
            rectangle(image, bestROI.getRect(), new Scalar(0, 0, 255, 0), 1, LINE_8, 0);
            imwrite("debug_lcd_final.png", image);
        }

        // Giải phóng bộ nhớ
        gray.release();
        blackhat.release();
        thresh.release();
        closed.release();
        hierarchy.release();

        return bestROI;
    }

    private RoiResult detectMechanicalDisplay(Mat image, MeterProfile profile, boolean debug) {
        Mat gray = new Mat();
        cvtColor(image, gray, COLOR_BGR2GRAY);

        // Dùng CLAHE để cân bằng ánh sáng
        Mat clahe = new Mat();
        createCLAHE(2.0, new Size(8, 8)).apply(gray, clahe);

        // Dùng adaptiveThreshold (ổn định hơn Otsu với ảnh có ánh sáng không đều)
        Mat binary = new Mat();
        threshold(clahe, binary, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);

        // Làm mượt nhẹ để loại bỏ nhiễu nhỏ
        Mat blurred = new Mat();
        GaussianBlur(binary, blurred, new Size(3, 3), 0);

        // Morphology close ngang để nối các vùng tối liền kề (khung dãy số)
        Mat kernelH = getStructuringElement(MORPH_RECT, new Size(20, 1));
        Mat closed1 = new Mat();
        morphologyEx(blurred, closed1, MORPH_CLOSE, kernelH);

        // Cắt caác vùng thừa
        Mat kernelV = getStructuringElement(MORPH_RECT, new Size(1, 30));
        Mat closed = new Mat();
        morphologyEx(closed1, closed, MORPH_OPEN, kernelV);

        // Debug trung gian
        if (debug) imwrite("debug_step2_closed.png", closed);

        // Tìm contour
        MatVector contours = new MatVector();
        Mat hierarchy = new Mat();
        findContours(closed, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        RoiResult bestROI = this.findBestRoi(contours, image, MeterType.MECHANIC, profile);

        // Nếu không tìm thấy ROI, thử mở rộng tìm lại vùng có độ phủ lớn
        if (bestROI == null && contours.size() > 0) {
            double maxArea = 0;
            for (int i = 0; i < contours.size(); i++) {
                Rect rect = boundingRect(contours.get(i));
                double area = contourArea(contours.get(i));
                if (area > maxArea) {
                    maxArea = area;
                    bestROI = this.calculateRoiResult(rect, MeterType.MECHANIC, image, profile);
                }
            }
        }

        if (bestROI != null && debug) {
            // Vẽ ROI để debug trực quan
            Mat cloned = image.clone();
            rectangle(cloned, bestROI.getRect(), new Scalar(0, 0, 255, 0), 1, LINE_8, 0);
            imwrite("debug_final_withROI.png", cloned);
            cloned.release();
        }

        // Giải phóng
        gray.release();
        clahe.release();
        binary.release();
        blurred.release();
        kernelH.release();
        kernelV.release();
        closed.release();
        hierarchy.release();

        return bestROI;
    }

    private RoiResult findBestRoi(MatVector contours, Mat image, MeterType meterType, MeterProfile profile) {
        List<RoiResult> candidates = new ArrayList<>();
        for (int i = 0; i < contours.size(); i++) {
            Mat contour = contours.get(i);
            Rect rect = boundingRect(contour);
            RoiResult roiResult = this.calculateRoiResult(rect, meterType, image, profile);
            if (Objects.nonNull(roiResult)) {
                candidates.add(roiResult);
            }
        }
        candidates.sort((a, b) -> Double.compare(b.getScore(), a.getScore()));
        return candidates.isEmpty() ? null : candidates.get(0);
    }

    private RoiResult calculateRoiResult(Rect roi, MeterType meterType, Mat image, MeterProfile profile) {
        if (Objects.isNull(roi)) return null;
        double hr = (double) roi.height() / image.rows();     // height ratio
        double wr = (double) roi.width() / image.cols();       // width ratio
        double ratio = roi.width() / (double) roi.height();

        if (hr >= profile.roi.hrange[0] && hr <= profile.roi.hrange[1] &&
                wr >= profile.roi.wrange[0] && wr <= profile.roi.wrange[1] &&
                ratio >= profile.roi.bestMinAspect && ratio <= profile.roi.bestMaxAspect &&
                roi.y() <= image.rows() / 2
        ) {
            // Tính điểm dựa trên các yếu tố
            double heightScore = Math.abs(hr - profile.roi.hr_best);
            double widthScore = (wr >= profile.roi.wrange[0] && wr <= profile.roi.wrange[1]) ? 1.0 : 0.5;
            double ratioScore = (ratio > profile.roi.bestMinAspect &&
                    roi.width() / (double) roi.height() < profile.roi.bestMaxAspect) ? 1.0 : 0.3; // tỷ lệ hợp lý
            double posScore = Math.abs((roi.x() + roi.width()) / 2.0 - image.cols() / 2.0);

            double score = heightScore * 0.09 + widthScore * 0.1 + ratioScore * 0.2 + posScore * 0.1;
            return new RoiResult(roi, score, meterType, profile);
        } else {
            return null;
        }
    }

    /**
     * Áp dụng Black Hat transform
     */
    private Mat applyBlackHat(Mat gray, int[] kernelSize) {
        Mat kernel = getStructuringElement(MORPH_RECT,
                new Size(kernelSize[0], kernelSize[1]));
        Mat blackhat = new Mat();
        morphologyEx(gray, blackhat, MORPH_BLACKHAT, kernel);
        kernel.release();
        return blackhat;
    }

    /**
     * Áp dụng Morphological Close
     */
    private Mat applyClose(Mat binary, int[] kernelSize) {
        Mat kernel = getStructuringElement(MORPH_RECT,
                new Size(kernelSize[0], kernelSize[1]));
        Mat closed = new Mat();
        morphologyEx(binary, closed, MORPH_CLOSE, kernel);
        kernel.release();
        return closed;
    }

}
