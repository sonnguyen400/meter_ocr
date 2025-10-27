package com.example.meterocr.service;

import com.example.meterocr.model.MeterProfile;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.Function;

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
    public enum MeterType {
        LCD,        // Màn hình LCD điện tử
        MECHANICAL, // Hộp số cơ
        AUTO        // Tự động phát hiện
    }

    public Rect detectMeterDisplay(Mat image, MeterProfile profile, MeterType meterType, boolean debug) {
        if (meterType == MeterType.AUTO) {
            // Thử cả hai phương pháp và chọn kết quả tốt nhất
            Rect lcdROI = detectLCDDisplay(image, profile, debug);
            Rect mechROI = detectMechanicalDisplay2(image, profile, debug);

            return chooseBestROI(lcdROI, mechROI, image, profile);
        } else if (meterType == MeterType.LCD) {
            return detectLCDDisplay(image, profile, debug);
        } else {
            return detectMechanicalDisplay2(image, profile, debug);
        }
    }


    private Rect detectLCDDisplay(Mat image, MeterProfile profile, boolean debug) {
        Mat gray = new Mat();
        cvtColor(image, gray, COLOR_BGR2GRAY);

        // Black Hat: làm nổi vùng tối trên nền sáng
        Mat blackhat = applyBlackHat(gray, profile.roi.blackhat_kernel);

        // Threshold
        Mat thresh = new Mat();
        threshold(blackhat, thresh, 0, 255, THRESH_BINARY + THRESH_OTSU);

        // Morphological Close
        Mat closed = applyClose(thresh, profile.roi.close_kernel);

        // Tìm contours
        MatVector contours = new MatVector();
        Mat hierarchy = new Mat();
        findContours(closed, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        // Tìm ROI tốt nhất
        Rect bestROI = findBestROILCD(contours, image, profile);

        if (debug) imwrite("debug_lcd_closed.png", closed);
        if (bestROI != null && debug) {
            rectangle(image, bestROI, new Scalar(0, 0, 255, 0), 3, LINE_8, 0);
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

    private Rect findBestROILCD(MatVector contours, Mat image, MeterProfile profile) {
        int imageHeight = image.rows();
        int imageWidth = image.cols();
        int imageArea = imageHeight * imageWidth;

        List<ROICandidate> candidates = new ArrayList<>();

        for (int i = 0; i < contours.size(); i++) {
            Mat contour = contours.get(i);
            Rect boundingRect = boundingRect(contour);

            double heightRatio = (double) boundingRect.height() / imageHeight;

            if (heightRatio >= profile.roi.hrange[0] &&
                    heightRatio <= profile.roi.hrange[1] &&
                    boundingRect.y() < imageHeight * 0.5) {

                double distance = Math.abs(heightRatio - profile.roi.hr_best);
                double aspectRatio = (double) boundingRect.width() / boundingRect.height();
                double area = contourArea(contour);

                if (area > imageArea * 0.005 &&
                        aspectRatio >= profile.roi.bestMinAspect && aspectRatio <= profile.roi.bestMaxAspect &&
                        boundingRect.width() > 50) {

                    candidates.add(new ROICandidate(boundingRect, distance, heightRatio, 0));
                }
            }
        }

        candidates.sort(Comparator.comparingDouble(c -> c.distance));

        return candidates.isEmpty() ? null : candidates.get(0).rect;
    }

    private Rect detectMechanicalDisplay2(Mat image, MeterProfile profile, boolean debug) {
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
        Mat kernelH = getStructuringElement(MORPH_RECT, new Size(15, 1));
        Mat closed1 = new Mat();
        morphologyEx(blurred, closed1, MORPH_CLOSE, kernelH);

        // Cắt caác vùng thừa
        Mat kernelV = getStructuringElement(MORPH_RECT, new Size(1, 15));
        Mat closed = new Mat();
        morphologyEx(closed1, closed, MORPH_OPEN, kernelV);

        // Debug trung gian
        if (debug) imwrite("debug_step2_closed.png", closed);

        // Tìm contour
        MatVector contours = new MatVector();
        Mat hierarchy = new Mat();
        findContours(closed, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        // Dùng logic lọc lỏng hơn để phù hợp cho đồng hồ GELEX
        Rect bestROI = findBestROIMechanicalRelaxed(contours, image, profile);

        // Nếu không tìm thấy ROI, thử mở rộng tìm lại vùng có độ phủ lớn
        if (bestROI == null && contours.size() > 0) {
            double maxArea = 0;
            for (int i = 0; i < contours.size(); i++) {
                Rect rect = boundingRect(contours.get(i));
                double area = contourArea(contours.get(i));
                if (area > maxArea) {
                    maxArea = area;
                    bestROI = rect;
                }
            }
        }

        if (bestROI != null && debug) {
            bestROI = expandROI(bestROI, image, 0.05, 0.1);
            // Vẽ ROI để debug trực quan
            Mat cloned = image.clone();
            rectangle(cloned, bestROI, new Scalar(0, 0, 255, 0), 3, LINE_8, 0);
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

    private Rect findBestROIMechanicalRelaxed(MatVector contours, Mat image, MeterProfile profile) {
        int h = image.rows();
        int w = image.cols();
        int areaImage = h * w;

        List<ROICandidate> candidates = new ArrayList<>();

        for (int i = 0; i < contours.size(); i++) {
            Mat contour = contours.get(i);
            Rect rect = boundingRect(contour);
            double heightRatio = rect.height() / (double) h;
            double aspectRatio = rect.width() / (double) rect.height();

//            System.out.println(String.format("x: %s, y: %s, width: %s, height: %s , hRa: %s, as: %s", rect.x(), rect.y(), rect.width(), rect.height(), heightRatio, aspectRatio));
            // Nới điều kiện một chút cho trường hợp vùng hiển thị nhỏ
            if (heightRatio >= profile.roi.hrange[0] && heightRatio <= profile.roi.hrange[1] &&
                    aspectRatio >= profile.roi.bestMinAspect && aspectRatio <= profile.roi.bestMaxAspect
                    && (rect.y() < image.rows()*0.5)) {

                double centerScore = image.cols()-(rect.x() + rect.width()) / 2.0;

                double aspectScore = Math.min(aspectRatio / 10.0, 1.0);

                double score = aspectScore * 0.2 + centerScore * 0.1;
                candidates.add(new ROICandidate(rect, 0, heightRatio, score));
            }
        }

        candidates.sort((a, b) -> Double.compare(b.density, a.density));
        return candidates.isEmpty() ? null : candidates.get(0).rect;
    }

    private Rect chooseBestROI(Rect lcdROI, Rect mechROI, Mat image, MeterProfile profile) {
        if (lcdROI == null && mechROI == null) {
            return null;
        }
        if (lcdROI == null) {
            return mechROI;
        }
        if (mechROI == null) {
            return lcdROI;
        }

        // So sánh dựa trên height ratio gần hr_best hơn

        Function<Rect, Double> scoreROI = (Rect r) -> {
            double hr = (double) r.height() / image.rows();     // height ratio
            double wr = (double) r.width() / image.cols();       // width ratio
            double ar = (double) r.area() / image.size().area();         // area ratio

            // Tính điểm dựa trên các yếu tố
            double heightScore = (hr - profile.roi.hr_best);
            double widthScore = (wr > 0.2 && wr < 0.9) ? 1.0 : 0.5;
            double ratioScore = (r.width() / (double) r.height() > profile.roi.bestMinAspect &&
                    r.width() / (double) r.height() < profile.roi.bestMaxAspect) ? 1.0 : 0.3; // tỷ lệ hợp lý
            double posScore = Math.abs((double) (r.x() + r.width() / 2.0) - image.cols());

            return heightScore * 0.4 + widthScore * 0.1 + ratioScore * 0.2 + posScore * 0.2;
        };
        double lcdDistance = scoreROI.apply(lcdROI);
        double mechDistance = scoreROI.apply(mechROI);

        if (lcdDistance < mechDistance) {
            System.out.println(String.format("Chose lcd %s %s %s %s", lcdROI.x(), lcdROI.y(), lcdROI.x() + lcdROI.width(), lcdROI.y() + lcdROI.height()));
        } else {
            System.out.println(String.format("Chose merch %s %s %s %s", mechROI.x(), mechROI.y(), mechROI.x() + mechROI.width(), mechROI.y() + mechROI.height()));
        }
        return lcdDistance < mechDistance ? lcdROI : mechROI;
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

    /**
     * Mở rộng ROI
     */
    private Rect expandROI(Rect roi, Mat image, double expandX, double expandY) {
        int expandWidth = (int) (roi.width() * expandX);
        int expandHeight = (int) (roi.height() * expandY);

        int x = Math.max(0, roi.x() - expandWidth);
        int y = Math.max(0, roi.y() - expandHeight);
        int width = Math.min(image.cols() - x, roi.width() + 2 * expandWidth);
        int height = Math.min(image.rows() - y, roi.height() + 2 * expandHeight);

        return new Rect(x, y, width, height);
    }

    private static class ROICandidate {
        Rect rect;
        double distance;
        double heightRatio;
        double density;

        ROICandidate(Rect rect, double distance, double heightRatio, double density) {
            this.rect = rect;
            this.distance = distance;
            this.heightRatio = heightRatio;
            this.density = density;
        }
    }

}
