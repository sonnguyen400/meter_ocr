
package com.example.meterocr.controller;

import com.example.meterocr.model.MeterType;
import com.example.meterocr.service.OcrService;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.net.URL;
import java.util.Map;

@RestController
@RequestMapping("/ocr")
public class MeterOcrController {

    private final OcrService ocrService;

    public MeterOcrController(OcrService ocrService) {
        this.ocrService = ocrService;
    }

    @PostMapping(value = "/meter", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<Map<String, Object>> ocrMeter(
            @RequestPart("file") MultipartFile file,
            @RequestParam(value = "type", required = false, defaultValue = "AUTO") MeterType type,
            @RequestParam(value = "debug", required = false, defaultValue = "false") boolean debug) throws Exception {
        byte[] bytes = file.getBytes();
        return ResponseEntity.ok(ocrService.process(bytes, type, debug));
    }

    @PostMapping("/meter/url")
    public ResponseEntity<Map<String, Object>> ocrMeterFromUrl(
            @RequestBody Map<String, String> body,
            @RequestParam(value = "type", required = false, defaultValue = "AUTO") MeterType type,
            @RequestParam(value = "debug", required = false, defaultValue = "false") boolean debug) throws Exception {
        String url = body.get("url");
        byte[] bytes = new URL(url).openStream().readAllBytes();
        return ResponseEntity.ok(ocrService.process(bytes, type, debug));
    }

    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("OK");
    }
}
