
package com.example.meterocr.util;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import org.bytedeco.javacpp.BytePointer;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;

public class ImageUtils {
    public static BufferedImage matToBufferedImage(Mat mat) throws Exception {
        BytePointer buf = new BytePointer();
        imencode(".png", mat, buf);
        byte[] bytes = new byte[(int) buf.limit()];
        buf.get(bytes);
        return ImageIO.read(new ByteArrayInputStream(bytes));
    }

    public static Mat cropMat(Mat src, Rect r) {
        if (src == null || src.empty() || r == null || r.width() <= 0 || r.height() <= 0) return null;
        return new Mat(src, r).clone();
    }

    public static Mat resizeToMinHeight(Mat src, int minH) {
        if (src == null || src.empty()) return src;
        if (src.rows() >= minH) return src;
        double scale = (double)minH / Math.max(1, src.rows());
        Mat dst = new Mat();
        resize(src, dst, new org.bytedeco.opencv.opencv_core.Size(
                (int)Math.round(src.cols()*scale), minH), 0, 0, INTER_CUBIC);
        return dst;
    }
}
