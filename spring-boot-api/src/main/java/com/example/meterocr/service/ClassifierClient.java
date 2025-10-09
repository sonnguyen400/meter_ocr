
package com.example.meterocr.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

@Service
public class ClassifierClient {
    private final OkHttpClient client = new OkHttpClient();
    private final ObjectMapper mapper = new ObjectMapper();

    @Value("${app.classifier-url:http://localhost:8601/classify}")
    private String classifierUrl;

    public static class ClsResult {
        public final String type; public final Map<String,Double> probs; public final String used;
        public ClsResult(String t, Map<String,Double> p, String u){ type=t; probs=p; used=u; }
    }

    public ClsResult classify(byte[] imageBytes) throws IOException {
        MultipartBody body = new MultipartBody.Builder().setType(MultipartBody.FORM)
                .addFormDataPart("file", "image.jpg", RequestBody.create(imageBytes, MediaType.parse("image/jpeg")))
                .build();
        Request req = new Request.Builder().url(classifierUrl).post(body).build();
        try(Response resp = client.newCall(req).execute()){
            if(!resp.isSuccessful()) throw new IOException("Classifier error: "+resp.code());
            JsonNode r = mapper.readTree(resp.body().string());
            String type = r.path("type").asText("lcd_digital");
            String used = r.path("used").asText("");
            Map<String,Double> probs = new HashMap<>();
            JsonNode p = r.path("probs");
            p.fieldNames().forEachRemaining(k -> probs.put(k, p.get(k).asDouble(0.0)));
            return new ClsResult(type, probs, used);
        }
    }
}
