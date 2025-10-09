
package com.example.meterocr.service;

import com.example.meterocr.model.MeterProfile;
import org.springframework.stereotype.Service;
import org.bytedeco.opencv.opencv_core.*;

import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_core.*;

@Service
public class LcdRoiDetector {

    public static class RoiResult {
        public final Rect rect;
        public final double score;
        public RoiResult(Rect rect, double score) { this.rect = rect; this.score = score; }
    }

    public RoiResult detect(Mat color, MeterProfile profile) {
        if (color == null || color.empty()) return new RoiResult(null, 0.0);
        int H = color.rows();
        int W = color.cols();

        Mat gray = new Mat();
        cvtColor(color, gray, COLOR_BGR2GRAY);
        Mat blur = new Mat();
        bilateralFilter(gray, blur, 7, 40, 20);

        int bhx = profile.roi.blackhat_kernel != null && profile.roi.blackhat_kernel.length>0 ? profile.roi.blackhat_kernel[0] : Math.max(17, W/80);
        int bhy = profile.roi.blackhat_kernel != null && profile.roi.blackhat_kernel.length>1 ? profile.roi.blackhat_kernel[1] : Math.max(5, H/240);
        Mat kernelBH = getStructuringElement(MORPH_RECT, new Size(bhx, bhy));
        Mat blackhat = new Mat();
        morphologyEx(blur, blackhat, MORPH_BLACKHAT, kernelBH);

        Mat gradX32 = new Mat();
        Scharr(blackhat, gradX32, CV_32F, 1, 0, 1, 0, BORDER_DEFAULT);
        Mat gradX = new Mat();
        convertScaleAbs(gradX32, gradX);
        normalize(gradX, gradX, 0, 255, NORM_MINMAX, CV_8U, new Mat());

        Mat thresh = new Mat();
        threshold(gradX, thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);
        int cx = profile.roi.close_kernel != null && profile.roi.close_kernel.length>0 ? profile.roi.close_kernel[0] : Math.max(25, W/48);
        int cy = profile.roi.close_kernel != null && profile.roi.close_kernel.length>1 ? profile.roi.close_kernel[1] : Math.max(3, H/400);
        Mat kernelClose = getStructuringElement(MORPH_RECT, new Size(cx, cy));
        morphologyEx(thresh, thresh, MORPH_CLOSE, kernelClose);
//        morphologyEx(thresh, thresh, MORPH_CLOSE, kernelClose, new Point(-1, -1), 2);
        erode(thresh, thresh, new Mat(), new Point(-1, -1), 1, BORDER_CONSTANT, morphologyDefaultBorderValue());
        dilate(thresh, thresh, new Mat(), new Point(-1, -1), 1, BORDER_CONSTANT, morphologyDefaultBorderValue());

        MatVector contours = new MatVector();
        Mat hierarchy = new Mat();
        findContours(thresh.clone(), contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Rect bestRect = null; double bestScore = 0.0;
        for (long i = 0; i < contours.size(); i++) {
            Mat c = contours.get(i);
            Rect r = boundingRect(c);
            if (r.width() < 40 || r.height() < 12 || r.width() > W*0.95 || r.height() > H*0.6) continue;
            double ar = r.width() / (double) r.height();
            if (ar < 2.0 || ar > 12.0) continue;

            Rect rsafe = clampRect(r, W, H);
            if (rsafe.width() <= 0 || rsafe.height() <= 0) continue;

            Mat maskRoi = new Mat(thresh, rsafe);
            double fill = countNonZero(maskRoi) / (double)(rsafe.width() * rsafe.height());

            double hr = rsafe.height() / (double) H;
            double hrMin = profile.roi.hrange != null && profile.roi.hrange.length>0 ? profile.roi.hrange[0] : 0.01;
            double hrMax = profile.roi.hrange != null && profile.roi.hrange.length>1 ? profile.roi.hrange[1] : 0.12;
            double hrBest = profile.roi.hr_best;
            double hrScore = (hr < hrMin || hr > hrMax) ? 0.0 : 1.0 - Math.min(1.0, Math.abs(hr - hrBest) / Math.max(1e-6, (hrMax-hrMin)/2));

            double arScore = 1.0 - Math.min(1.0, Math.abs(ar - 5.0) / 5.0);
            double fillScore = 1.0 - Math.min(1.0, Math.abs(fill - 0.45) / 0.45);

            Mat gradRoi = new Mat(gradX, rsafe);
            Scalar meanGrad = mean(gradRoi);
            double edgeScore = Math.min(1.0, meanGrad.get(0) / 64.0);

            double score = 0.35*arScore + 0.30*hrScore + 0.20*fillScore + 0.15*edgeScore;

            if (score > bestScore) { bestScore = score; bestRect = rsafe; }
        }

        if (bestRect != null) {
            bestRect = expand(bestRect, W, H, profile.roi.expand_x, profile.roi.expand_y);
        }
        return new RoiResult(bestRect, bestScore);
    }

    private Rect clampRect(Rect r, int W, int H) {
        int x = Math.max(0, r.x());
        int y = Math.max(0, r.y());
        int w = Math.min(r.width(), W - x);
        int h = Math.min(r.height(), H - y);
        return new Rect(x, y, Math.max(0, w), Math.max(0, h));
    }

    private Rect expand(Rect r, int W, int H, double rx, double ry) {
        int mx = (int) Math.round(r.width() * rx);
        int my = (int) Math.round(r.height() * ry);
        int x = Math.max(0, r.x() - mx);
        int y = Math.max(0, r.y() - my);
        int w = Math.min(W - x, r.width() + 2*mx);
        int h = Math.min(H - y, r.height() + 2*my);
        return new Rect(x, y, w, h);
    }
}
