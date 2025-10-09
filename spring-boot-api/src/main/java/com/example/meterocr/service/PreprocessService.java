
package com.example.meterocr.service;

import com.example.meterocr.model.MeterProfile;
import com.example.meterocr.model.PreprocessResult;
import org.bytedeco.opencv.opencv_imgproc.CLAHE;
import org.bytedeco.opencv.opencv_imgproc.Vec4iVector;
import org.springframework.stereotype.Service;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.javacpp.*;

@Service
public class PreprocessService {

    public PreprocessResult preprocess(byte[] imageBytes, MeterProfile profile) {
        Mat buf = new Mat(1, imageBytes.length, CV_8U, new BytePointer(imageBytes));
        Mat img = imdecode(buf, IMREAD_COLOR);
        if (img == null || img.empty()) {
            throw new IllegalArgumentException("Cannot decode image");
        }

        Mat gray = new Mat();
        cvtColor(img, gray, COLOR_BGR2GRAY);

        Mat edges = new Mat();
        Canny(gray, edges, 50, 150);
        Mat lines = new Mat();
        Vec4iVector linesVec = new Vec4iVector();
        HoughLinesP(edges, linesVec, 1, Math.PI/180, 60, 60, 10);
        double angle = estimateSkewAngle(lines);

        Mat rotated = rotate(img, angle);

        Mat gray2 = new Mat();
        cvtColor(rotated, gray2, COLOR_BGR2GRAY);

        double clip = profile.preprocess.clahe_clip;
        int tileX = profile.preprocess.clahe_tile != null && profile.preprocess.clahe_tile.length>0 ? profile.preprocess.clahe_tile[0] : 8;
        int tileY = profile.preprocess.clahe_tile != null && profile.preprocess.clahe_tile.length>1 ? profile.preprocess.clahe_tile[1] : 8;
        CLAHE clahe = createCLAHE(clip, new Size(tileX, tileY));
        Mat claheGray = new Mat();
        clahe.apply(gray2, claheGray);

        int block = Math.max(3, profile.preprocess.adaptive_block | 1);
        int C = profile.preprocess.adaptive_C;
        Mat bin = new Mat();
        adaptiveThreshold(claheGray, bin, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, block, C);

        return new PreprocessResult(rotated, bin, angle);
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
}
