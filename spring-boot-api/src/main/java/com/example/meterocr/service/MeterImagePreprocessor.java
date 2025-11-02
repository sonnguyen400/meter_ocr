package com.example.meterocr.service;

import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_imgproc.CLAHE;
import org.springframework.stereotype.Service;

@Service
public class MeterImagePreprocessor {

    public Mat enhance(Mat src) {
        Mat gray = new Mat();

        // 1️⃣ Chuyển sang grayscale nếu cần
        if (src.channels() == 3) {
            opencv_imgproc.cvtColor(src, gray, opencv_imgproc.COLOR_BGR2GRAY);
        } else {
            gray = src.clone();
        }

        // 2️⃣ Giảm phản sáng nhẹ bằng CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Mat claheDst = new Mat();
        CLAHE clahe = opencv_imgproc.createCLAHE(1.2, new Size(8, 8));
        clahe.apply(gray, claheDst);

        // 3️⃣ Lọc nhiễu & làm mượt bằng Bilateral Filter (giữ cạnh chữ)
        Mat smooth = new Mat();
        opencv_imgproc.bilateralFilter(claheDst, smooth, 5, 50, 50);

        // 4️⃣ Tăng tương phản nhẹ nhàng bằng linear scaling (alpha ≈ 1.2)
        Mat contrasted = new Mat();
        smooth.convertTo(contrasted, -1, 1.2, 0);  // alpha=1.2, beta=0

        // 5️⃣ (Tuỳ chọn) Giảm sáng nhẹ nếu vẫn còn vùng phản sáng
        Mat result = new Mat();
        contrasted.convertTo(result, -1, 1.0, -10);

        return result;
    }
}
