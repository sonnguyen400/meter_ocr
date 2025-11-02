
package com.example.meterocr.service;

import com.example.meterocr.model.MeterProfile;
import com.example.meterocr.model.PreprocessResult;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point2f;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_imgproc.CLAHE;
import org.bytedeco.opencv.opencv_imgproc.Vec4iVector;
import org.springframework.stereotype.Service;

import static org.bytedeco.opencv.global.opencv_core.BORDER_REPLICATE;
import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import static org.bytedeco.opencv.global.opencv_core.addWeighted;
import static org.bytedeco.opencv.global.opencv_core.bitwise_and;
import static org.bytedeco.opencv.global.opencv_core.minMaxLoc;
import static org.bytedeco.opencv.global.opencv_core.subtract;
import static org.bytedeco.opencv.global.opencv_core.normalize;
import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_COLOR;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imdecode;
import static org.bytedeco.opencv.global.opencv_imgproc.ADAPTIVE_THRESH_GAUSSIAN_C;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.Canny;
import static org.bytedeco.opencv.global.opencv_imgproc.GaussianBlur;
import static org.bytedeco.opencv.global.opencv_imgproc.HoughLinesP;
import static org.bytedeco.opencv.global.opencv_imgproc.INTER_LINEAR;
import static org.bytedeco.opencv.global.opencv_imgproc.MORPH_CLOSE;
import static org.bytedeco.opencv.global.opencv_imgproc.MORPH_ELLIPSE;
import static org.bytedeco.opencv.global.opencv_imgproc.MORPH_OPEN;
import static org.bytedeco.opencv.global.opencv_imgproc.MORPH_RECT;
import static org.bytedeco.opencv.global.opencv_imgproc.THRESH_BINARY;
import static org.bytedeco.opencv.global.opencv_imgproc.adaptiveThreshold;
import static org.bytedeco.opencv.global.opencv_imgproc.bilateralFilter;
import static org.bytedeco.opencv.global.opencv_imgproc.createCLAHE;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.dilate;
import static org.bytedeco.opencv.global.opencv_imgproc.erode;
import static org.bytedeco.opencv.global.opencv_imgproc.getRotationMatrix2D;
import static org.bytedeco.opencv.global.opencv_imgproc.getStructuringElement;
import static org.bytedeco.opencv.global.opencv_imgproc.morphologyEx;
import static org.bytedeco.opencv.global.opencv_imgproc.warpAffine;
import static org.bytedeco.opencv.global.opencv_photo.fastNlMeansDenoising;
import static org.opencv.core.Core.NORM_MINMAX;

@Service
public class PreprocessService {

    public PreprocessResult preprocess(Mat mat, MeterProfile profile) {
        Mat img = imdecode(mat, IMREAD_COLOR);
        if (img == null || img.empty()) {
            throw new IllegalArgumentException("Cannot decode image");
        }

        Mat gray = new Mat();
        cvtColor(img, gray, COLOR_BGR2GRAY);

        Mat edges = new Mat();
        Canny(gray, edges, 50, 150);
        Mat lines = new Mat();
        Vec4iVector linesVec = new Vec4iVector();
        HoughLinesP(edges, linesVec, 1, Math.PI / 180, 60, 60, 10);
        double angle = estimateSkewAngle(lines);

        Mat rotated = rotate(img, angle);

        Mat gray2 = new Mat();
        cvtColor(rotated, gray2, COLOR_BGR2GRAY);

        // Denoise
        Mat denoised = new Mat();
        fastNlMeansDenoising(gray2, denoised, 3, 7, 21);

        // Sharpen using unsharp masking
        Mat blurred = new Mat();
        GaussianBlur(denoised, blurred, new Size(0, 0), 3);
        Mat sharpened = new Mat();
        addWeighted(denoised, 1.5, blurred, -0.5, 0, sharpened);


        double clip = profile.preprocess.clahe_clip;
        int tileX = profile.preprocess.clahe_tile != null && profile.preprocess.clahe_tile.length > 0 ? profile.preprocess.clahe_tile[0] : 8;
        int tileY = profile.preprocess.clahe_tile != null && profile.preprocess.clahe_tile.length > 1 ? profile.preprocess.clahe_tile[1] : 8;
        CLAHE clahe = createCLAHE(clip, new Size(tileX, tileY));
        Mat claheGray = new Mat();
        clahe.apply(sharpened, claheGray);

        int block = Math.max(3, profile.preprocess.adaptive_block | 1);
        int C = profile.preprocess.adaptive_C;
        Mat bin = new Mat();
        adaptiveThreshold(claheGray, bin, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, block, C);

        return new PreprocessResult(rotated, bin, angle);
    }

    public PreprocessResult preprocess(byte[] imageBytes, MeterProfile profile) {
        Mat buf = new Mat(1, imageBytes.length, CV_8U, new BytePointer(imageBytes));
        return preprocess(buf, profile);
    }

    public Mat preprocess(Mat mat){
        Mat processed = mat.clone();

        if (processed.channels() > 1) {
            Mat gray = new Mat();
            cvtColor(processed, gray, COLOR_BGR2GRAY);
            processed.release();
            processed = gray;
        }

        // Bước 1: Cân bằng histogram cục bộ mạnh
        Mat clahe = applyCLAHE(processed, 4.0, new int[]{8, 8});
        processed.release();

        // Bước 2: Bilateral filter - giảm nhiễu nhưng giữ cạnh sắc nét
        Mat filtered = new Mat();
        bilateralFilter(clahe, filtered, 7, 75, 75);
        clahe.release();

        // Bước 3: Tăng độ tương phản
        Mat contrasted = new Mat();
        filtered.convertTo(contrasted, -1, 1.5, -50); // alpha=1.5, beta=-50
        filtered.release();

        // Đảm bảo giá trị trong khoảng [0, 255]
        // Tìm min/max
        double minVal[] = new double[1];
        double maxVal[] = new double[1];
        minMaxLoc(contrasted, minVal, maxVal, null, null, null);

        // Normalize thủ công bằng convertTo
        Mat normalized = new Mat();
        contrasted.convertTo(normalized, CV_8U,
                255.0 / (maxVal[0] - minVal[0]),           // alpha (scale)
                -minVal[0] * 255.0 / (maxVal[0] - minVal[0])  // beta (shift)
        );

        // Bước 4: Adaptive threshold với 2 pass
        // Pass 1: Block size lớn để bắt cấu trúc tổng thể
        Mat thresh1 = new Mat();
        adaptiveThreshold(normalized, thresh1, 255,
                ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,
                51, 10);

        // Pass 2: Block size nhỏ để bắt chi tiết
        Mat thresh2 = new Mat();
        adaptiveThreshold(normalized, thresh2, 255,
                ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,
                21, 8);
        normalized.release();

        // Kết hợp 2 threshold
        Mat combined = new Mat();
        bitwise_and(thresh1, thresh2, combined);
        thresh1.release();
        thresh2.release();

        // Bước 5: Morphological cleaning
        // Loại bỏ nhiễu nhỏ
        Mat kernelOpen = getStructuringElement(MORPH_RECT, new Size(2, 2));
        Mat opened = new Mat();
        morphologyEx(combined, opened, MORPH_OPEN, kernelOpen);
        combined.release();
        kernelOpen.release();

        // Kết nối các phần chữ số bị đứt
        Mat kernelClose = getStructuringElement(MORPH_ELLIPSE, new Size(3, 3));
        Mat closed = new Mat();
        morphologyEx(opened, closed, MORPH_CLOSE, kernelClose);
        opened.release();
        kernelClose.release();

        // Bước 6: Xóa các đường thẳng dọc (rãnh phân cách)
        Mat noLines = removeVerticalLines(closed);
        closed.release();

        return noLines;
    }
    private Mat applyCLAHE(Mat gray, double clipLimit, int[] tileSize) {
        CLAHE clahe = createCLAHE(clipLimit, new Size(tileSize[0], tileSize[1]));
        Mat result = new Mat();
        clahe.apply(gray, result);
        clahe.close();
        return result;
    }

    private Mat removeVerticalLines(Mat binary) {
        // Tạo kernel dọc dài để phát hiện đường thẳng dọc
        Mat verticalKernel = getStructuringElement(MORPH_RECT, new Size(1, 15));

        // Erode và dilate để tìm đường thẳng dọc
        Mat verticalLines = new Mat();
        erode(binary, verticalLines, verticalKernel);
        dilate(verticalLines, verticalLines, verticalKernel);

        // Xóa đường thẳng dọc khỏi ảnh gốc
        Mat result = new Mat();
        subtract(binary, verticalLines, result);

        verticalKernel.release();
        verticalLines.release();

        return result;
    }


    private Mat rotate(Mat src, double angle) {
        Point2f center = new Point2f(src.cols() / 2f, src.rows() / 2f);
        Mat rotMat = getRotationMatrix2D(center, angle, 1.0);
        Mat dst = new Mat();
        warpAffine(src, dst, rotMat, src.size(), INTER_LINEAR, BORDER_REPLICATE, new Scalar(255, 255, 255, 0));
        return dst;
    }

    private double estimateSkewAngle(Mat lines) {
        if (lines == null || lines.empty()) return 0.0;
        double sum = 0;
        int cnt = 0;
        for (int i = 0; i < lines.rows(); i++) {
            IntPointer p = new IntPointer(lines.row(i).data());
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
