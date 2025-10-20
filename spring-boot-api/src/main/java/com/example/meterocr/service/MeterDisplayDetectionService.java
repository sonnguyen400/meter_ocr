package com.example.meterocr.service;

import com.example.meterocr.model.MeterProfile;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_imgproc.CLAHE;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.BORDER_DEFAULT;
import static org.bytedeco.opencv.global.opencv_core.CV_16S;
import static org.bytedeco.opencv.global.opencv_core.addWeighted;
import static org.bytedeco.opencv.global.opencv_core.convertScaleAbs;
import static org.bytedeco.opencv.global.opencv_core.countNonZero;
import static org.bytedeco.opencv.global.opencv_core.meanStdDev;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.ADAPTIVE_THRESH_GAUSSIAN_C;
import static org.bytedeco.opencv.global.opencv_imgproc.CHAIN_APPROX_SIMPLE;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.Canny;
import static org.bytedeco.opencv.global.opencv_imgproc.FONT_HERSHEY_SIMPLEX;
import static org.bytedeco.opencv.global.opencv_imgproc.GaussianBlur;
import static org.bytedeco.opencv.global.opencv_imgproc.INTER_CUBIC;
import static org.bytedeco.opencv.global.opencv_imgproc.LINE_8;
import static org.bytedeco.opencv.global.opencv_imgproc.LINE_AA;
import static org.bytedeco.opencv.global.opencv_imgproc.MORPH_BLACKHAT;
import static org.bytedeco.opencv.global.opencv_imgproc.MORPH_CLOSE;
import static org.bytedeco.opencv.global.opencv_imgproc.MORPH_RECT;
import static org.bytedeco.opencv.global.opencv_imgproc.RETR_EXTERNAL;
import static org.bytedeco.opencv.global.opencv_imgproc.Sobel;
import static org.bytedeco.opencv.global.opencv_imgproc.THRESH_BINARY;
import static org.bytedeco.opencv.global.opencv_imgproc.THRESH_OTSU;
import static org.bytedeco.opencv.global.opencv_imgproc.adaptiveThreshold;
import static org.bytedeco.opencv.global.opencv_imgproc.bilateralFilter;
import static org.bytedeco.opencv.global.opencv_imgproc.boundingRect;
import static org.bytedeco.opencv.global.opencv_imgproc.contourArea;
import static org.bytedeco.opencv.global.opencv_imgproc.createCLAHE;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.dilate;
import static org.bytedeco.opencv.global.opencv_imgproc.findContours;
import static org.bytedeco.opencv.global.opencv_imgproc.getStructuringElement;
import static org.bytedeco.opencv.global.opencv_imgproc.morphologyEx;
import static org.bytedeco.opencv.global.opencv_imgproc.putText;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.bytedeco.opencv.global.opencv_imgproc.threshold;

@Service
public class MeterDisplayDetectionService {

    /**
     * Enum để xác định loại công tơ
     */
    public enum MeterType {
        LCD,        // Màn hình LCD điện tử
        MECHANICAL, // Hộp số cơ
        AUTO        // Tự động phát hiện
    }

    /**
     * Phát hiện vùng chứa chỉ số công tơ tự động (cả LCD và hộp số cơ)
     * @param imagePath đường dẫn đến file ảnh
     * @param profile cấu hình phát hiện
     * @return vùng ROI tốt nhất chứa chỉ số
     */
    public Rect detectMeterDisplay(String imagePath, MeterProfile profile) {
        Mat image = imread(imagePath);
        if (image.empty()) {
            throw new RuntimeException("Không thể đọc ảnh: " + imagePath);
        }
        
        Rect roi = detectMeterDisplay(image, profile, MeterType.AUTO);
        image.release();
        return roi;
    }

    /**
     * Phát hiện vùng chứa chỉ số công tơ từ Mat
     * @param image ảnh đầu vào
     * @param profile cấu hình phát hiện
     * @param meterType loại công tơ (LCD, MECHANICAL, AUTO)
     * @return vùng ROI tốt nhất chứa chỉ số
     */
    public Rect detectMeterDisplay(Mat image, MeterProfile profile, MeterType meterType) {
        if (meterType == MeterType.AUTO) {
            // Thử cả hai phương pháp và chọn kết quả tốt nhất
            Rect lcdROI = detectLCDDisplay(image, profile);
            Rect mechROI = detectMechanicalDisplay(image, profile);
            
            return chooseBestROI(lcdROI, mechROI, image, profile);
        } else if (meterType == MeterType.LCD) {
            return detectLCDDisplay(image, profile);
        } else {
            return detectMechanicalDisplay(image, profile);
        }
    }

    /**
     * Phát hiện màn hình LCD (chữ số tối trên nền sáng)
     * Sử dụng Black Hat transform
     */
    private Rect detectLCDDisplay(Mat image, MeterProfile profile) {
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
        findContours(closed, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        // Tìm ROI tốt nhất
        Rect bestROI = findBestROI(contours, image, profile);
        
        // Mở rộng ROI
        if (bestROI != null) {
            bestROI = expandROI(bestROI, image, profile.roi.expand_x, profile.roi.expand_y);
        }
        
        // Giải phóng bộ nhớ
        gray.release();
        blackhat.release();
        thresh.release();
        closed.release();
        hierarchy.release();
        
        return bestROI;
    }

    /**
     * Phát hiện hộp số cơ (chữ số nổi, có viền, hoặc nền tối)
     * Sử dụng phương pháp kết hợp Canny và Morphology
     */
    private Rect detectMechanicalDisplay(Mat image, MeterProfile profile) {
        Mat gray = new Mat();
        cvtColor(image, gray, COLOR_BGR2GRAY);
        
        // Phương pháp 1: Phát hiện cạnh (hộp số có nhiều cạnh)
        Mat edges = detectEdges(gray);
        
        // Phương pháp 2: Phát hiện vùng có texture đặc trưng
        Mat texture = detectTexture(gray);
        
        // Kết hợp cả hai phương pháp
        Mat combined = new Mat();
        addWeighted(edges, 0.6, texture, 0.4, 0, combined);
        
        // Morphological operations
        Mat closed = applyClose(combined, profile.roi.close_kernel);
        
        // Tìm contours
        MatVector contours = new MatVector();
        Mat hierarchy = new Mat();
        findContours(closed, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        // Tìm ROI với tiêu chí khác cho hộp số cơ
        Rect bestROI = findBestROIMechanical(contours, image, profile);
        
        // Mở rộng ROI
        if (bestROI != null) {
            bestROI = expandROI(bestROI, image, profile.roi.expand_x, profile.roi.expand_y);
        }
        
        // Giải phóng bộ nhớ
        gray.release();
        edges.release();
        texture.release();
        combined.release();
        closed.release();
        hierarchy.release();
        
        return bestROI;
    }

    /**
     * Phát hiện cạnh cho hộp số cơ
     */
    private Mat detectEdges(Mat gray) {
        Mat blurred = new Mat();
        GaussianBlur(gray, blurred, new Size(3, 3), 0);
        
        Mat edges = new Mat();
        Canny(blurred, edges, 30, 100);
        
        // Dilate để kết nối các cạnh gần nhau
        Mat kernel = getStructuringElement(MORPH_RECT, new Size(3, 3));
        Mat dilated = new Mat();
        dilate(edges, dilated, kernel);
        
        blurred.release();
        edges.release();
        kernel.release();
        
        return dilated;
    }

    /**
     * Phát hiện texture (hộp số cơ có texture đặc trưng với các rãnh phân cách)
     */
    private Mat detectTexture(Mat gray) {
        // Sử dụng Sobel để phát hiện gradient
        Mat gradX = new Mat();
        Mat gradY = new Mat();
        
        Sobel(gray, gradX, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
        Sobel(gray, gradY, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
        
        Mat absGradX = new Mat();
        Mat absGradY = new Mat();
        convertScaleAbs(gradX, absGradX);
        convertScaleAbs(gradY, absGradY);
        
        Mat grad = new Mat();
        addWeighted(absGradX, 0.5, absGradY, 0.5, 0, grad);
        
        // Threshold
        Mat thresh = new Mat();
        threshold(grad, thresh, 0, 255, THRESH_BINARY + THRESH_OTSU);
        
        // Giải phóng
        gradX.release();
        gradY.release();
        absGradX.release();
        absGradY.release();
        grad.release();
        
        return thresh;
    }

    /**
     * Tìm ROI tốt nhất cho hộp số cơ
     * Hộp số cơ thường có:
     * - Nhiều cạnh và chi tiết hơn LCD
     * - Có thể có khung viền rõ ràng
     * - Aspect ratio hẹp hơn LCD (2:1 đến 5:1)
     */
    private Rect findBestROIMechanical(MatVector contours, Mat image, MeterProfile profile) {
        int imageHeight = image.rows();
        int imageWidth = image.cols();
        int imageArea = imageHeight * imageWidth;
        
        List<ROICandidate> candidates = new ArrayList<>();
        
        for (int i = 0; i < contours.size(); i++) {
            Mat contour = contours.get(i);
            Rect boundingRect = boundingRect(contour);
            
            double heightRatio = (double) boundingRect.height() / imageHeight;
            
            // Hộp số cơ có thể có height ratio khác LCD
            if (heightRatio >= profile.roi.hrange[0] && 
                heightRatio <= profile.roi.hrange[1]) {
                
                double distance = Math.abs(heightRatio - profile.roi.hr_best);
                double aspectRatio = (double) boundingRect.width() / boundingRect.height();
                double area = contourArea(contour);
                
                // Hộp số cơ: aspect ratio 2:1 đến 6:1, có thể hẹp hơn LCD
                if (area > imageArea * 0.003 && 
                    aspectRatio > 1.5 && aspectRatio < 7.0 &&
                    boundingRect.width() > 40 &&
                    boundingRect.height() > 15) {
                    
                    // Tính điểm dựa trên mật độ cạnh (hộp số cơ có nhiều chi tiết)
                    double rectArea = boundingRect.width() * boundingRect.height();
                    double density = area / rectArea;
                    
                    candidates.add(new ROICandidate(boundingRect, distance, heightRatio, density));
                }
            }
        }
        
        // Sắp xếp: ưu tiên mật độ cao (nhiều chi tiết) và gần hr_best
        candidates.sort((c1, c2) -> {
            double score1 = c1.density * 0.6 - c1.distance * 0.4;
            double score2 = c2.density * 0.6 - c2.distance * 0.4;
            return Double.compare(score2, score1);
        });
        
        return candidates.isEmpty() ? null : candidates.get(0).rect;
    }

    /**
     * Tìm ROI tốt nhất cho LCD
     */
    private Rect findBestROI(MatVector contours, Mat image, MeterProfile profile) {
        int imageHeight = image.rows();
        int imageWidth = image.cols();
        int imageArea = imageHeight * imageWidth;
        
        List<ROICandidate> candidates = new ArrayList<>();
        
        for (int i = 0; i < contours.size(); i++) {
            Mat contour = contours.get(i);
            Rect boundingRect = boundingRect(contour);
            
            double heightRatio = (double) boundingRect.height() / imageHeight;
            
            if (heightRatio >= profile.roi.hrange[0] && 
                heightRatio <= profile.roi.hrange[1]) {
                
                double distance = Math.abs(heightRatio - profile.roi.hr_best);
                double aspectRatio = (double) boundingRect.width() / boundingRect.height();
                double area = contourArea(contour);
                
                if (area > imageArea * 0.005 &&
                    aspectRatio > 2.0 && aspectRatio < 8.0 &&
                    boundingRect.width() > 50) {
                    
                    candidates.add(new ROICandidate(boundingRect, distance, heightRatio, 0));
                }
            }
        }
        
        candidates.sort(Comparator.comparingDouble(c -> c.distance));
        
        return candidates.isEmpty() ? null : candidates.get(0).rect;
    }

    /**
     * Chọn ROI tốt nhất giữa LCD và Mechanical
     */
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
        double lcdDistance = Math.abs((double) lcdROI.height() / image.rows() - profile.roi.hr_best);
        double mechDistance = Math.abs((double) mechROI.height() / image.rows() - profile.roi.hr_best);
        
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

    /**
     * Tiền xử lý cho LCD
     */
    public Mat preprocessLCD(Mat roi, MeterProfile profile) {
        Mat processed = roi.clone();
        
        if (processed.channels() > 1) {
            Mat gray = new Mat();
            cvtColor(processed, gray, COLOR_BGR2GRAY);
            processed.release();
            processed = gray;
        }
        
        Mat clahe = applyCLAHE(processed, 
            profile.preprocess.clahe_clip, 
            profile.preprocess.clahe_tile);
        processed.release();
        
        Mat thresh = new Mat();
        adaptiveThreshold(clahe, thresh, 255, 
            ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,
            profile.preprocess.adaptive_block,
            profile.preprocess.adaptive_C);
        clahe.release();
        
        return thresh;
    }

    /**
     * Tiền xử lý cho hộp số cơ
     * Hộp số cơ cần xử lý khác: làm nổi bật chi tiết và giảm bóng
     */
    public Mat preprocessMechanical(Mat roi, MeterProfile profile) {
        Mat processed = roi.clone();
        
        if (processed.channels() > 1) {
            Mat gray = new Mat();
            cvtColor(processed, gray, COLOR_BGR2GRAY);
            processed.release();
            processed = gray;
        }
        
        // CLAHE mạnh hơn cho hộp số cơ
        Mat clahe = applyCLAHE(processed, 
            profile.preprocess.clahe_clip * 1.5, 
            profile.preprocess.clahe_tile);
        processed.release();
        
        // Bilateral filter để giảm nhiễu nhưng giữ cạnh
        Mat filtered = new Mat();
        bilateralFilter(clahe, filtered, 5, 50, 50);
        clahe.release();
        
        // Adaptive threshold với block size nhỏ hơn
        Mat thresh = new Mat();
        adaptiveThreshold(filtered, thresh, 255, 
            ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,
            Math.max(11, profile.preprocess.adaptive_block - 10),
            profile.preprocess.adaptive_C);
        filtered.release();
        
        return thresh;
    }

    /**
     * Tiền xử lý tự động
     */
    public Mat preprocessROI(Mat roi, MeterProfile profile, MeterType meterType) {
        if (meterType == MeterType.MECHANICAL) {
            return preprocessMechanical(roi, profile);
        } else {
            return preprocessLCD(roi, profile);
        }
    }

    /**
     * Áp dụng CLAHE
     */
    private Mat applyCLAHE(Mat gray, double clipLimit, int[] tileSize) {
        CLAHE clahe = createCLAHE(clipLimit, new Size(tileSize[0], tileSize[1]));
        Mat result = new Mat();
        clahe.apply(gray, result);
        clahe.close();
        return result;
    }

    /**
     * Resize cho OCR
     */
    public Mat resizeForOCR(Mat roi, MeterProfile profile) {
        int minHeight = profile.tesseract.resize_min_height;
        
        if (roi.rows() < minHeight) {
            double scale = (double) minHeight / roi.rows();
            int newWidth = (int) (roi.cols() * scale);
            int newHeight = minHeight;
            
            Mat resized = new Mat();
            resize(roi, resized, new Size(newWidth, newHeight), 0, 0, INTER_CUBIC);
            return resized;
        }
        
        return roi.clone();
    }

    /**
     * Pipeline đầy đủ với tự động phát hiện loại công tơ
     */
    public Mat detectAndPreprocess(String imagePath, MeterProfile profile) {
        Mat image = imread(imagePath);
        if (image.empty()) {
            throw new RuntimeException("Không thể đọc ảnh: " + imagePath);
        }
        
        Rect roi = detectMeterDisplay(image, profile, MeterType.AUTO);
        
        if (roi == null) {
            image.release();
            throw new RuntimeException("Không tìm thấy vùng hiển thị công tơ");
        }
        
        Mat roiMat = new Mat(image, roi);
        Mat roiClone = roiMat.clone();
        image.release();
        
        // Phát hiện loại công tơ để chọn phương pháp tiền xử lý
        MeterType detectedType = detectMeterType(roiClone);
        
        Mat preprocessed = preprocessROI(roiClone, profile, detectedType);
        roiClone.release();
        
        Mat resized = resizeForOCR(preprocessed, profile);
        preprocessed.release();
        
        return resized;
    }

    /**
     * Phát hiện loại công tơ dựa trên đặc điểm ảnh
     */
    private MeterType detectMeterType(Mat roi) {
        Mat gray = new Mat();
        if (roi.channels() > 1) {
            cvtColor(roi, gray, COLOR_BGR2GRAY);
        } else {
            gray = roi.clone();
        }
        
        // Tính variance - hộp số cơ có variance cao hơn (nhiều chi tiết)
        Mat mean = new Mat();
        Mat stddev = new Mat();
        meanStdDev(gray, mean, stddev);

        // Lấy giá trị stddev từ Mat buffer
        org.bytedeco.javacpp.indexer.DoubleIndexer indexer = stddev.createIndexer();
        double variance = indexer.get(0);
        

        // Đếm số cạnh - hộp số cơ có nhiều cạnh hơn
        Mat edges = new Mat();
        Canny(gray, edges, 50, 150);
        int edgeCount = countNonZero(edges);
        double edgeDensity = (double) edgeCount / (roi.rows() * roi.cols());
        
        gray.release();
        edges.release();
        mean.release();
        stddev.release();
        
        // Hộp số cơ: variance cao và edge density cao
        if (variance > 40 && edgeDensity > 0.15) {
            return MeterType.MECHANICAL;
        } else {
            return MeterType.LCD;
        }
    }

    /**
     * Lưu debug images với cả hai phương pháp
     */
    public void saveDebugImages(String inputPath, String outputDir, MeterProfile profile) {
        Mat image = imread(inputPath);
        if (image.empty()) {
            throw new RuntimeException("Không thể đọc ảnh: " + inputPath);
        }
        
        // Phát hiện LCD
        Rect lcdROI = detectLCDDisplay(image, profile);
        if (lcdROI != null) {
            Mat imageLCD = image.clone();
            rectangle(imageLCD, lcdROI, new Scalar(0, 255, 0, 0), 2, LINE_8, 0);
            putText(imageLCD, "LCD", new Point(lcdROI.x(), lcdROI.y() - 10),
                    FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 255, 0, 0), 2, LINE_AA, false);
            imwrite(outputDir + "/lcd_detection.jpg", imageLCD);
            imageLCD.release();
        }
        
        // Phát hiện Mechanical
        Rect mechROI = detectMechanicalDisplay(image, profile);
        if (mechROI != null) {
            Mat imageMech = image.clone();
            rectangle(imageMech, mechROI, new Scalar(255, 0, 0, 0), 2, LINE_8, 0);
            putText(imageMech, "MECHANICAL", new Point(mechROI.x(), mechROI.y() - 10),
                    FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 0, 0, 0), 2, LINE_AA, false);
            imwrite(outputDir + "/mechanical_detection.jpg", imageMech);
            imageMech.release();
        }
        
        // ROI tốt nhất
        Rect bestROI = chooseBestROI(lcdROI, mechROI, image, profile);
        if (bestROI != null) {
            Mat imageBest = image.clone();
            rectangle(imageBest, bestROI, new Scalar(0, 0, 255, 0), 3, LINE_8, 0);
            imwrite(outputDir + "/best_roi.jpg", imageBest);
            
            // Tiền xử lý
            Mat roiMat = new Mat(image, bestROI);
            MeterType type = detectMeterType(roiMat);
            Mat preprocessed = preprocessROI(roiMat, profile, type);
            imwrite(outputDir + "/preprocessed_" + type + ".jpg", preprocessed);
            
            preprocessed.release();
            imageBest.release();
        }
        
        image.release();
    }

    /**
     * Class helper cho ROI candidate
     */
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
