
package com.example.meterocr.util;

import com.example.meterocr.model.Box;
import org.bytedeco.opencv.opencv_core.*;

import static org.bytedeco.opencv.global.opencv_imgproc.*;

import java.util.List;

public class DebugOverlayUtils {
    public static void drawPaddleBoxes(Mat img, List<Box> boxes) {
        if (boxes == null) return;
        for (Box b : boxes) {
            if (b.getPolygon() != null && b.getPolygon().size() >= 4) {
                Point pts = new Point(4);
                for (int i = 0; i < 4; i++) {
                    double x = b.getPolygon().get(i).get(0);
                    double y = b.getPolygon().get(i).get(1);
                    pts.position(i).x((int) Math.round(x));
                    pts.position(i).y((int) Math.round(y));
                }

//                polylines(img, new Point[]{pts.position(0)}, new int[]{4}, 1, true, new Scalar(255,0,0,0), 2, LINE_AA, 0);
                polylines(img, pts.position(0), new int[]{4}, 1, true, new Scalar(255, 0, 0, 0), 2, LINE_AA, 0);
                String label = (b.getText() != null ? b.getText() : "") + String.format(" (%.2f)", b.getConf());
                putText(img, label, new Point(pts.position(0).x(), Math.max(0, pts.position(0).y() - 5)), FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 0, 0, 0), 1, LINE_AA, false);
            }
        }
    }
}
