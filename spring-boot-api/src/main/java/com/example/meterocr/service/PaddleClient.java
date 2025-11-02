
package com.example.meterocr.service;

import com.example.meterocr.model.Box;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

@Service
public class PaddleClient {
    private final OkHttpClient client;
    private final ObjectMapper mapper = new ObjectMapper();

    @Value("${app.paddle-url}")
    private String paddleUrl;

    public PaddleClient(){
        this.client = new OkHttpClient().newBuilder()
                .callTimeout(Duration.ofSeconds(20))
                .build();
    }
    public List<Box> ocr(byte[] imageBytes) throws IOException {
        MultipartBody body = new MultipartBody.Builder().setType(MultipartBody.FORM)
                .addFormDataPart("file", "image.jpg",
                        RequestBody.create(imageBytes, MediaType.parse("image/jpeg")))
                .build();
        Request request = new Request.Builder().url(paddleUrl).post(body).build();
        try (Response resp = client.newCall(request).execute()) {
            if (!resp.isSuccessful()) {
                throw new IOException("PaddleOCR error: " + resp.code());
            }
            String json = resp.body().string();
            JsonNode root = mapper.readTree(json).get("boxes");
            List<Box> boxes = new ArrayList<>();
            if (root != null && root.isArray()) {
                for (JsonNode n : root) {
                    String text = n.path("text").asText("");
                    double conf = n.path("conf").asDouble(0.0);
                    Box b = new Box(text, conf);
                    if (n.has("polygon") && n.get("polygon").isArray()) {
                        java.util.List<java.util.List<Double>> poly = new java.util.ArrayList<>();
                        for (JsonNode p : n.get("polygon")) {
                            java.util.List<Double> pt = new java.util.ArrayList<>();
                            pt.add(p.get(0).asDouble());
                            pt.add(p.get(1).asDouble());
                            poly.add(pt);
                        }
                        b.setPolygon(poly);
                    }
                    boxes.add(b);
                }
            }
            return boxes;
        }
    }
}
