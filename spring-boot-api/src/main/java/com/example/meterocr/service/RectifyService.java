
package com.example.meterocr.service;

import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.springframework.stereotype.Service;
import org.bytedeco.opencv.opencv_core.*;

import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_core.*;

@Service
public class RectifyService {

    public static class RectifyResult {
        public final Mat color;
        public final Mat bin;
        public final Size size;
        public RectifyResult(Mat color, Mat bin, Size size) { this.color = color; this.bin = bin; this.size = size; }
    }

    public RectifyResult rectifySmart(Mat rotatedColor, Mat rotatedBin, Rect roi, java.util.List<com.example.meterocr.model.Box> boxes) {
        if (rotatedColor == null || rotatedColor.empty() || roi == null || boxes == null) return rectify(rotatedColor, rotatedBin, roi);
        java.util.List<double[]> tops = new java.util.ArrayList<>();
        java.util.List<double[]> bots = new java.util.ArrayList<>();
        double minX = Double.POSITIVE_INFINITY, maxX = Double.NEGATIVE_INFINITY;
        for (com.example.meterocr.model.Box b : boxes) {
            var poly = b.getPolygon(); if (poly == null || poly.size() < 4) continue;
            double cx=0, cy=0; for (var pt: poly){ cx += pt.get(0); cy += pt.get(1);} cx/=poly.size(); cy/=poly.size();
            if (cx < roi.x() || cx > roi.x()+roi.width() || cy < roi.y() || cy > roi.y()+roi.height()) continue;
            java.util.List<Double> ys = new java.util.ArrayList<>();
            java.util.List<Double> xs = new java.util.ArrayList<>();
            for (var pt: poly){ xs.add(pt.get(0)); ys.add(pt.get(1)); }
            java.util.Collections.sort(ys); java.util.Collections.sort(xs);
            double topY = 0.5*(ys.get(0)+ys.get(1));
            double botY = 0.5*(ys.get(ys.size()-1)+ys.get(ys.size()-2));
            double leftX = xs.get(0); double rightX = xs.get(xs.size()-1);
            tops.add(new double[]{(leftX+rightX)/2.0, topY});
            bots.add(new double[]{(leftX+rightX)/2.0, botY});
            minX = Math.min(minX, leftX); maxX = Math.max(maxX, rightX);
        }
        if (tops.size() < 3 || bots.size() < 3) {
            return rectify(rotatedColor, rotatedBin, roi);
        }
        double[] topAB = fitLineAB(tops); double[] botAB = fitLineAB(bots);
        if (topAB == null || botAB == null) return rectify(rotatedColor, rotatedBin, roi);
        double aT = topAB[0], bT = topAB[1];
        double aB = botAB[0], bB = botAB[1];
        double xL = Math.max(roi.x(), Math.floor(minX));
        double xR = Math.min(roi.x()+roi.width(), Math.ceil(maxX));
        double yTL = aT*xL + bT, yTR = aT*xR + bT;
        double yBL = aB*xL + bB, yBR = aB*xR + bB;
        if (!(yBL>yTL && yBR>yTR)) {
            return rectify(rotatedColor, rotatedBin, roi);
        }
        Point2f src = new Point2f(4);
        src.position(0).x((float)xL); src.position(0).y((float)yTL);
        src.position(1).x((float)xR); src.position(1).y((float)yTR);
        src.position(2).x((float)xR); src.position(2).y((float)yBR);
        src.position(3).x((float)xL); src.position(3).y((float)yBL);
        int outW = (int)Math.max(50, Math.round(xR - xL));
        int outH = (int)Math.max(20, Math.round(((yBL - yTL) + (yBR - yTR))/2.0));
        Point2f dst = new Point2f(4);
        dst.position(0).x(0f);      dst.position(0).y(0f);
        dst.position(1).x(outW-1f); dst.position(1).y(0f);
        dst.position(2).x(outW-1f); dst.position(2).y(outH-1f);
        dst.position(3).x(0f);      dst.position(3).y(outH-1f);

        Mat M = getPerspectiveTransform(src, dst);
        Mat warpedColor = new Mat();
        warpPerspective(rotatedColor, warpedColor, M, new Size(outW, outH), INTER_CUBIC, BORDER_REPLICATE, new Scalar(255,255,255,0));
        Mat warpedBin = new Mat();
        warpPerspective(rotatedBin, warpedBin, M, new Size(outW, outH), INTER_NEAREST, BORDER_REPLICATE, new Scalar(255,255,255,0));
        return new RectifyResult(warpedColor, warpedBin, new Size(outW, outH));
    }

    public RectifyResult rectify(Mat rotatedColor, Mat rotatedBin, Rect roi) {
        if (rotatedColor == null || rotatedColor.empty() || roi == null) return null;
        Rect r = expand(roi, rotatedColor.cols(), rotatedColor.rows(), 0.03, 0.01);
        Mat binCrop = new Mat(rotatedBin, r).clone();
        Mat colorCrop = new Mat(rotatedColor, r).clone();

        Mat cntSrc = binCrop.clone();
        MatVector contours = new MatVector();
        Mat hierarchy = new Mat();
        findContours(cntSrc, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if (contours.size() == 0) {
            return new RectifyResult(colorCrop, binCrop, colorCrop.size());
        }

        Mat allPts = new Mat();
        for (long i=0;i<contours.size();i++) {
            Mat c = contours.get(i);
            allPts.push_back(c);
        }
        RotatedRect rr = minAreaRect(allPts);
        Point2f points = new Point2f(4);
        rr.points(points);

        Point2f tl = points.position(0);
        Point2f tr = points.position(1);
        Point2f br = points.position(2);
        Point2f bl = points.position(3);

        Point2f[] pts = new Point2f[]{tl,tr,br,bl};
        java.util.Arrays.sort(pts, (a,b) -> {
            if (a.y() == b.y()) return Float.compare(a.x(), b.x());
            return Float.compare(a.y(), b.y());
        });
        Point2f top1 = pts[0], top2 = pts[1], bot1 = pts[2], bot2 = pts[3];
        Point2f topLeft = (top1.x() <= top2.x()) ? top1 : top2;
        Point2f topRight = (top1.x() > top2.x()) ? top1 : top2;
        Point2f botLeft = (bot1.x() <= bot2.x()) ? bot1 : bot2;
        Point2f botRight = (bot1.x() > bot2.x()) ? bot1 : bot2;

        double widthA = Math.hypot(botRight.x() - botLeft.x(), botRight.y() - botLeft.y());
        double widthB = Math.hypot(topRight.x() - topLeft.x(), topRight.y() - topLeft.y());
        double maxWidth = Math.max(widthA, widthB);
        double heightA = Math.hypot(topRight.x() - botRight.x(), topRight.y() - botRight.y());
        double heightB = Math.hypot(topLeft.x() - botLeft.x(), topLeft.y() - botLeft.y());
        double maxHeight = Math.max(heightA, heightB);

        if (maxWidth < 10 || maxHeight < 10) {
            return new RectifyResult(colorCrop, binCrop, colorCrop.size());
        }

        Mat srcMat = new Mat(4, 1, CV_32FC2);
        FloatIndexer indexer1 = srcMat.createIndexer();
        indexer1.put(0,0, new float[]{topLeft.x(), topLeft.y()});
        FloatIndexer indexer2 = srcMat.createIndexer();
        indexer2.put(1,0, new float[]{topRight.x(), topRight.y()});
        FloatIndexer indexer3 = srcMat.createIndexer();
        indexer3.put(2,0, new float[]{botRight.x(), botRight.y()});
        FloatIndexer indexer4 = srcMat.createIndexer();
        indexer4.put(3,0, new float[]{botLeft.x(), botLeft.y()});

        Mat dstMat = new Mat(4, 1, CV_32FC2);
        float W = (float)maxWidth;
        float H = (float)maxHeight;
        FloatIndexer dstIndexes1 = dstMat.createIndexer();dstIndexes1.put(0,0, new float[]{0f, 0f});
        FloatIndexer dstIndexes2 = dstMat.createIndexer();dstIndexes2.put(1,0, new float[]{W-1f, 0f});
        FloatIndexer dstIndexes3 = dstMat.createIndexer();dstIndexes3.put(2,0, new float[]{W-1f, H-1f});
        FloatIndexer dstIndexes4 = dstMat.createIndexer();dstIndexes4.put(3,0, new float[]{0f, H-1f});

        Mat M = getPerspectiveTransform(srcMat, dstMat);
        Mat warpedColor = new Mat();
        warpPerspective(colorCrop, warpedColor, M, new Size((int)W, (int)H), INTER_CUBIC, BORDER_REPLICATE, new Scalar(255,255,255,0));
        Mat warpedBin = new Mat();
        warpPerspective(binCrop, warpedBin, M, new Size((int)W, (int)H), INTER_NEAREST, BORDER_REPLICATE, new Scalar(255,255,255,0));

        return new RectifyResult(warpedColor, warpedBin, new Size((int)W,(int)H));
    }

    private Rect expand(Rect r, int W, int H, double rx, double ry) {
        int mx = (int)Math.round(r.width()*rx);
        int my = (int)Math.round(r.height()*ry);
        int x = Math.max(0, r.x()-mx), y = Math.max(0, r.y()-my);
        int w = Math.min(W - x, r.width()+2*mx);
        int h = Math.min(H - y, r.height()+2*my);
        return new Rect(x,y,w,h);
    }

    private double[] fitLineAB(java.util.List<double[]> pts){
        int n = pts.size(); if (n < 2) return null; double sx=0, sy=0, sxx=0, sxy=0; for (double[] p: pts){ sx+=p[0]; sy+=p[1]; sxx+=p[0]*p[0]; sxy+=p[0]*p[1]; }
        double denom = n*sxx - sx*sx; if (Math.abs(denom) < 1e-6) return null; double a = (n*sxy - sx*sy)/denom; double b = (sy - a*sx)/n; return new double[]{a,b};
    }
}
